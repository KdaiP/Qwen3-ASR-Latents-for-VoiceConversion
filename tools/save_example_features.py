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
    overwrite: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def iter_audio_files(config: Config) -> list[Path]:
    return sorted(
        path
        for path in config.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.audio_extensions
    )


def save_feature(model: Qwen3ASRWrapper, audio_path: Path, output_path: Path, config: Config) -> None:
    audio, sample_rate = torchaudio.load(str(audio_path))
    embedding = model.forward(audio, sample_rate, add_extra_padding=config.add_extra_padding).detach().cpu()

    payload = {
        "audio_path": str(audio_path.relative_to(REPO_ROOT)),
        "sample_rate": int(sample_rate),
        "embedding_shape": tuple(embedding.shape),
        "embedding_dtype": str(embedding.dtype),
        "versions": {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "transformers": transformers.__version__,
        },
    }
    torch.save(embedding, output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    config = Config()
    config.feature_dir.mkdir(parents=True, exist_ok=True)

    audio_files = iter_audio_files(config)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {config.audio_dir}")

    print("Feature export config:")
    print(json.dumps(asdict(config), indent=2, default=str))

    model = Qwen3ASRWrapper(str(config.model_path)).to(config.device).eval()

    for audio_path in audio_files:
        output_path = config.feature_dir / f"{audio_path.stem}.pt"
        if output_path.exists() and not config.overwrite:
            print(f"[skip] {output_path.relative_to(REPO_ROOT)} already exists")
            continue

        save_feature(model, audio_path, output_path, config)
        metadata_path = output_path.with_suffix(".json")
        print(
            f"[save] {audio_path.relative_to(REPO_ROOT)} -> "
            f"{output_path.relative_to(REPO_ROOT)}, {metadata_path.relative_to(REPO_ROOT)}"
        )

    print(f"Saved {len(audio_files)} feature file(s) to {config.feature_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
