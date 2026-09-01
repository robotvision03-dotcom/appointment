"""Download vhdm/whisper-large-fa-v1 and convert it to CTranslate2 for faster-whisper.

Usage: python -m src.download_whisper
"""

from __future__ import annotations

from pathlib import Path

from src.config import config
from src.utils import log


def _ct2_ready(path: Path) -> bool:
    return (path / "model.bin").is_file()


def download_and_convert(force: bool = False) -> Path:
    dest = config.whisper_model_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _ct2_ready(dest) and not force:
        log.info("Whisper CT2 already at %s", dest)
        return dest

    hf_id = config.whisper_model_id
    log.info("Downloading %s and converting to CTranslate2 int8 → %s", hf_id, dest)
    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(
        hf_id,
        copy_files=[
            "tokenizer.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
        ],
    )
    converter.convert(str(dest), quantization="int8", force=force or dest.exists())
    if not _ct2_ready(dest):
        raise RuntimeError(f"Conversion finished but {dest / 'model.bin'} is missing")
    log.info("Whisper large Farsi v1 ready at %s", dest)
    return dest


def main() -> int:
    download_and_convert()
    print("OK", config.whisper_model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
