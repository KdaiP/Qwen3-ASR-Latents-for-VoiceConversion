import sys
import json

from pathlib import Path
from dataclasses import asdict, dataclass

import torch
import torchaudio
import transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_asr.wrapper import Qwen3ASRWrapper


@dataclass(frozen=True)
class Config:
    model_path: Path = REPO_ROOT / "pretrained" / "qwen3_asr"
    audio_dir: Path = REPO_ROOT / "examples" / "audios"
    feature_dir: Path = REPO_ROOT / "examples" / "features"
    audio_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
    add_extra_padding: bool = True
    atol: float = 1e-3
    rtol: float = 1e-3
    fail_on_mismatch: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class CompareResult:
    audio_name: str
    passed: bool
    shape: tuple[int, ...]
    reference_shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    cosine_similarity: float


def iter_audio_files(config: Config) -> list[Path]:
    return sorted(
        path
        for path in config.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.audio_extensions
    )


def load_reference(feature_path: Path) -> torch.Tensor:
    embedding = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"{feature_path} does not contain a tensor. Regenerate it with save_example_features.py.")
    return embedding.detach().cpu()


def load_reference_versions(feature_path: Path) -> dict[str, str]:
    metadata_path = feature_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("versions"), dict):
        return {str(key): str(value) for key, value in payload["versions"].items()}
    return {}


def infer_feature(model: Qwen3ASRWrapper, audio_path: Path, config: Config) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(str(audio_path))
    return model.forward(audio, sample_rate, add_extra_padding=config.add_extra_padding).detach().cpu()


def compare_embedding(audio_path: Path, reference: torch.Tensor, current: torch.Tensor, config: Config) -> CompareResult:
    reference_float = reference.float()
    current_float = current.float()

    shape_matches = tuple(reference.shape) == tuple(current.shape)
    if shape_matches:
        diff = (reference_float - current_float).abs()
        max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
        cosine_similarity = float(
            torch.nn.functional.cosine_similarity(
                reference_float.reshape(1, -1),
                current_float.reshape(1, -1),
                dim=1,
            ).item()
        )
        values_match = bool(torch.allclose(reference_float, current_float, atol=config.atol, rtol=config.rtol))
    else:
        max_abs_diff = float("inf")
        mean_abs_diff = float("inf")
        cosine_similarity = float("nan")
        values_match = False

    return CompareResult(
        audio_name=audio_path.name,
        passed=shape_matches and values_match,
        shape=tuple(current.shape),
        reference_shape=tuple(reference.shape),
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        cosine_similarity=cosine_similarity,
    )


def print_result(result: CompareResult) -> None:
    status = "pass" if result.passed else "fail"
    print(
        f"[{status}] {result.audio_name} "
        f"shape={result.shape} ref_shape={result.reference_shape} "
        f"max_abs={result.max_abs_diff:.8g} mean_abs={result.mean_abs_diff:.8g} "
        f"cos={result.cosine_similarity:.8g}"
    )


def main() -> None:
    config = Config()
    audio_files = iter_audio_files(config)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {config.audio_dir}")

    print("Feature comparison config:")
    print(json.dumps(asdict(config), indent=2, default=str))
    print("Current dependency versions:")
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "transformers": transformers.__version__,
            },
            indent=2,
        )
    )

    model = Qwen3ASRWrapper(str(config.model_path)).to(config.device).eval()
    results: list[CompareResult] = []
    missing_features: list[Path] = []

    for audio_path in audio_files:
        feature_path = config.feature_dir / f"{audio_path.stem}.pt"
        if not feature_path.exists():
            missing_features.append(feature_path)
            print(f"[missing] {feature_path.relative_to(REPO_ROOT)}")
            continue

        reference_versions = load_reference_versions(feature_path)
        if reference_versions:
            print(f"[ref] {feature_path.name} versions={reference_versions}")

        reference = load_reference(feature_path)
        current = infer_feature(model, audio_path, config)
        result = compare_embedding(audio_path, reference, current, config)
        results.append(result)
        print_result(result)

    failed = [result for result in results if not result.passed]
    print(f"Compared {len(results)} file(s): {len(results) - len(failed)} passed, {len(failed)} failed")

    if missing_features:
        print(f"Missing {len(missing_features)} feature file(s)")

    if config.fail_on_mismatch and (failed or missing_features):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
