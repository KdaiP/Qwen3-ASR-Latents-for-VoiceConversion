# coding=utf-8
"""Audio embedding extraction with the Qwen3-ASR audio tower only."""

from __future__ import annotations

import base64
import gc
import io
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModel, WhisperFeatureExtractor

from .core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRConfig
from .core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRForConditionalGeneration,
    _get_feat_extract_output_lengths,
)

SAMPLE_RATE = 16000
AudioArray = Union[np.ndarray, torch.Tensor]
AudioInput = Union[str, Path, AudioArray, tuple[AudioArray, int]]
LoadMode = Literal["audio_tower", "full"]


@dataclass
class AudioEmbeddingResult:
    """Embedding output for one audio item."""

    source: Optional[str]
    duration_seconds: float
    num_frames: int
    embedding: torch.Tensor


def _is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _is_probably_base64(value: str) -> bool:
    if value.startswith("data:audio"):
        return True
    return "/" not in value and "\\" not in value and len(value) > 256


def _decode_audio_string(value: str) -> tuple[np.ndarray, int]:
    if _is_url(value):
        with urllib.request.urlopen(value) as response:
            audio_bytes = response.read()
        with io.BytesIO(audio_bytes) as handle:
            audio, sample_rate = sf.read(handle, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    if _is_probably_base64(value):
        payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
        with io.BytesIO(base64.b64decode(payload)) as handle:
            audio, sample_rate = sf.read(handle, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    audio, sample_rate = librosa.load(value, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _is_audio_array(value: Any) -> bool:
    return isinstance(value, (np.ndarray, torch.Tensor))


def _as_numpy_audio(value: AudioArray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if value.layout != torch.strided:
        raise ValueError("Unsupported torch tensor layout; expected a dense strided audio tensor.")
    if value.is_complex():
        raise ValueError("Unsupported torch tensor dtype; expected real-valued audio samples.")

    tensor = value.detach().cpu()
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor.numpy()


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}; expected mono or stereo audio.")


def _normalize_range(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if audio.size == 0:
        raise ValueError("Audio is empty.")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)


def load_audio(audio: AudioInput) -> tuple[np.ndarray, Optional[str]]:
    """Load one audio item as 16 kHz mono float32 PCM."""

    source = None
    if isinstance(audio, (str, Path)):
        source = str(audio)
        waveform, sample_rate = _decode_audio_string(str(audio))
    elif isinstance(audio, tuple) and len(audio) == 2 and _is_audio_array(audio[0]):
        waveform, sample_rate = _as_numpy_audio(audio[0]), int(audio[1])
    elif _is_audio_array(audio):
        waveform, sample_rate = _as_numpy_audio(audio), SAMPLE_RATE
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    waveform = _to_mono(np.asarray(waveform))
    if sample_rate != SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return _normalize_range(waveform), source


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: Union[str, torch.dtype], device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[dtype_name]


def _iter_unique(items: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for item in items:
        resolved = item.resolve()
        if resolved not in seen and item.exists() and item.stat().st_size > 0:
            seen.add(resolved)
            output.append(item)
    return output


def _candidate_weight_files(model_path: Path, model_keys: set[str]) -> list[Path]:
    prefixes = ("thinker.audio_tower.", "audio_tower.", "model.thinker.audio_tower.")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        selected = []
        for key, filename in index.get("weight_map", {}).items():
            if key in model_keys or any(key.startswith(prefix) for prefix in prefixes):
                selected.append(model_path / filename)
        selected_files = _iter_unique(selected)
        if selected_files:
            return selected_files

    safetensors_files = sorted(model_path.glob("*.safetensors"))
    bin_files = sorted(model_path.glob("pytorch_model*.bin")) + sorted(model_path.glob("*.pt"))
    return _iter_unique([*safetensors_files, *bin_files])


def _read_checkpoint_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        return load_safetensors(str(path), device="cpu")
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def _load_audio_tower_weights(audio_tower: Qwen3ASRAudioEncoder, model_path: Path) -> list[Path]:
    model_state = audio_tower.state_dict()
    model_keys = set(model_state)
    parameter_keys = {name for name, _ in audio_tower.named_parameters()}
    prefixes = ("thinker.audio_tower.", "audio_tower.", "model.thinker.audio_tower.")
    files = _candidate_weight_files(model_path, model_keys)
    if not files:
        raise FileNotFoundError(
            f"No checkpoint weights found under {model_path}. "
            "Expected model.safetensors, model-*.safetensors, or pytorch_model*.bin."
        )

    filtered: dict[str, torch.Tensor] = {}
    for path in files:
        state = _read_checkpoint_file(path)
        for key, value in state.items():
            target_key = key if key in model_keys else None
            if target_key is None:
                for prefix in prefixes:
                    if key.startswith(prefix):
                        target_key = key[len(prefix) :]
                        break
            if target_key in model_keys and isinstance(value, torch.Tensor):
                filtered[target_key] = value
        del state
        gc.collect()

    if not filtered:
        raise RuntimeError(
            "Checkpoint files were found, but no audio tower weights were matched. "
            "Expected keys like 'thinker.audio_tower.conv2d1.weight'."
        )

    missing, _ = audio_tower.load_state_dict(filtered, strict=False)
    missing_parameters = [name for name in missing if name in parameter_keys]
    if missing_parameters:
        preview = ", ".join(missing_parameters[:10])
        more = "" if len(missing_parameters) <= 10 else f", ... ({len(missing_parameters)} total)"
        raise RuntimeError(f"Audio tower checkpoint is incomplete. Missing: {preview}{more}")
    return files


def _load_audio_config(model_path: Path) -> Qwen3ASRAudioEncoderConfig:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {model_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    try:
        audio_config = raw_config["thinker_config"]["audio_config"]
    except KeyError as exc:
        raise KeyError("config.json does not contain thinker_config.audio_config") from exc
    return Qwen3ASRAudioEncoderConfig(**audio_config)


def _load_feature_extractor(
    model_path: Path,
    audio_config: Qwen3ASRAudioEncoderConfig,
) -> WhisperFeatureExtractor:
    try:
        return WhisperFeatureExtractor.from_pretrained(str(model_path))
    except Exception:
        return WhisperFeatureExtractor(
            feature_size=audio_config.num_mel_bins,
            sampling_rate=SAMPLE_RATE,
            hop_length=160,
            n_fft=400,
            chunk_length=30,
            padding_value=0.0,
            return_attention_mask=True,
        )


class Qwen3ASREmbeddingExtractor:
    """Extract frame-level audio embeddings from Qwen3-ASR."""

    def __init__(
        self,
        audio_tower: Qwen3ASRAudioEncoder,
        feature_extractor: WhisperFeatureExtractor,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        checkpoint_files: Optional[list[Path]] = None,
    ):
        self.audio_tower = audio_tower.eval()
        for parameter in self.audio_tower.parameters():
            parameter.requires_grad_(False)
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)
        self.dtype = dtype
        self.checkpoint_files = checkpoint_files or []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path] = "pretrained/qwen3_asr",
        device: Optional[Union[str, torch.device]] = "auto",
        dtype: Union[str, torch.dtype] = "auto",
        load_mode: LoadMode = "audio_tower",
        attn_implementation: str = "eager",
        **model_kwargs: Any,
    ) -> "Qwen3ASREmbeddingExtractor":
        """Load an extractor from a Qwen3-ASR checkpoint directory."""

        resolved_path = Path(model_path)
        resolved_device = _resolve_device(device)
        resolved_dtype = _resolve_dtype(dtype, resolved_device)

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)

        audio_config = _load_audio_config(resolved_path)
        feature_extractor = _load_feature_extractor(resolved_path, audio_config)

        if load_mode == "full":
            config = Qwen3ASRConfig.from_pretrained(str(resolved_path))
            model = AutoModel.from_pretrained(
                str(resolved_path),
                config=config,
                dtype=resolved_dtype,
                attn_implementation=attn_implementation,
                **model_kwargs,
            )
            model.to(resolved_device)
            audio_tower = model.thinker.audio_tower
            checkpoint_files: list[Path] = []
        elif load_mode == "audio_tower":
            audio_config._attn_implementation = attn_implementation
            audio_tower = Qwen3ASRAudioEncoder(audio_config)
            checkpoint_files = _load_audio_tower_weights(audio_tower, resolved_path)
            audio_tower.to(device=resolved_device, dtype=resolved_dtype)
        else:
            raise ValueError(f"Unsupported load_mode: {load_mode!r}")

        return cls(
            audio_tower=audio_tower,
            feature_extractor=feature_extractor,
            device=resolved_device,
            dtype=resolved_dtype,
            checkpoint_files=checkpoint_files,
        )

    @torch.inference_mode()
    def extract_batch(
        self,
        audios: Sequence[AudioInput],
    ) -> list[AudioEmbeddingResult]:
        loaded = [load_audio(audio) for audio in audios]
        waveforms = [item[0] for item in loaded]
        sources = [item[1] for item in loaded]
        durations = [float(len(waveform) / SAMPLE_RATE) for waveform in waveforms]

        features = self.feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = features["input_features"].to(device=self.device, dtype=self.dtype)
        feature_attention_mask = features["attention_mask"].to(device=self.device)
        feature_lengths = feature_attention_mask.sum(dim=1).long()

        results: list[AudioEmbeddingResult] = []
        for input_feature, feature_len, source, duration in zip(
            input_features,
            feature_lengths,
            sources,
            durations,
        ):
            output = self.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            frames = output.last_hidden_state
            expected_frames = int(_get_feat_extract_output_lengths(feature_len).item())
            frames = frames[:expected_frames]

            results.append(
                AudioEmbeddingResult(
                    source=source,
                    duration_seconds=duration,
                    num_frames=int(frames.shape[0]),
                    embedding=frames.detach().cpu(),
                )
            )

        return results

    def extract(
        self,
        audio: Union[AudioInput, Sequence[AudioInput]],
    ) -> Union[AudioEmbeddingResult, list[AudioEmbeddingResult]]:
        is_single_tuple = isinstance(audio, tuple) and len(audio) == 2 and _is_audio_array(audio[0])
        is_batch = isinstance(audio, Sequence) and not isinstance(audio, (str, bytes, Path, np.ndarray, torch.Tensor))
        if is_batch and not is_single_tuple:
            return self.extract_batch(list(audio))
        return self.extract_batch([audio])[0]

__all__ = [
    "AudioEmbeddingResult",
    "Qwen3ASREmbeddingExtractor",
    "load_audio",
]
