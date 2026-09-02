"""Download Shenava-Koochik-v1.5 for local hearing (sherpa-onnx).

Source: Reza2kn/Shenava-Koochik-v1.5
Runtime weights: Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx

Usage:
  python -m src download-shenava
  python -m src download-whisper
  py scripts/download_shenava.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import config
from src.utils import log

V15 = "Reza2kn/Shenava-Koochik-v1.5"
V15_RNNT = "Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx"
FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)


def _ready(dest: Path) -> bool:
    return all((dest / name).is_file() for name in FILES)


def download(force: bool = False) -> Path:
    dest = config.shenava_model_path
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.txt").write_text(f"{V15}\n{V15_RNNT}\n", encoding="utf-8")
    if _ready(dest) and not force:
        log.info("Shenava-Koochik-v1.5 already at %s", dest)
        print("OK", dest)
        return dest

    from huggingface_hub import hf_hub_download

    log.info("Downloading %s (int8 RNNT for sherpa-onnx)", V15_RNNT)
    for name in FILES:
        cached = hf_hub_download(repo_id=V15_RNNT, filename=name)
        target = dest / name
        shutil.copy(cached, target)
        log.info("saved %s (%s bytes)", target.name, target.stat().st_size)
    if not _ready(dest):
        missing = [n for n in FILES if not (dest / n).is_file()]
        raise RuntimeError(f"Shenava download incomplete: {missing}")
    print("OK Shenava-Koochik-v1.5", dest)
    print("Then: python -m src")
    return dest


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
