"""Download Shenava models for local hearing (sherpa-onnx).

Primary (better WER): CTC head from Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx
  — same CTC as v1.5, the recommended deployed head.

Fallback: RNNT int8 from Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx

Usage:
  python -m src download-shenava
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import config
from src.utils import log

V15 = "Reza2kn/Shenava-Koochik-v1.5"
V15_RNNT = "Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx"
V10_CTC = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"
RNNT_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
CTC_FILES = ("model.onnx", "tokens.txt")


def _rnnt_ready(dest: Path) -> bool:
    return all((dest / name).is_file() for name in RNNT_FILES)


def _ctc_ready(dest: Path) -> bool:
    return all((dest / name).is_file() for name in CTC_FILES)


def _copy_hf(repo: str, name: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo, filename=name)
    target = dest / name
    shutil.copy(cached, target)
    log.info("saved %s (%s bytes)", target.name, target.stat().st_size)


def download_ctc(force: bool = False) -> Path:
    dest = config.shenava_ctc_path
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.txt").write_text(f"{V10_CTC}\nCTC head of {V15}\n", encoding="utf-8")
    if _ctc_ready(dest) and not force:
        log.info("Shenava CTC already at %s", dest)
        print("OK CTC", dest)
        return dest
    log.info("Downloading %s (~450 MB CTC, recommended hearing)", V10_CTC)
    for name in CTC_FILES:
        _copy_hf(V10_CTC, name, dest)
    if not _ctc_ready(dest):
        raise RuntimeError("Shenava CTC download incomplete")
    print("OK CTC", dest)
    return dest


def download_rnnt(force: bool = False) -> Path:
    dest = config.shenava_model_path
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.txt").write_text(f"{V15}\n{V15_RNNT}\n", encoding="utf-8")
    if _rnnt_ready(dest) and not force:
        log.info("Shenava RNNT already at %s", dest)
        print("OK RNNT", dest)
        return dest
    log.info("Downloading %s (int8 RNNT fallback)", V15_RNNT)
    for name in RNNT_FILES:
        _copy_hf(V15_RNNT, name, dest)
    if not _rnnt_ready(dest):
        raise RuntimeError("Shenava RNNT download incomplete")
    print("OK RNNT", dest)
    return dest


def download(force: bool = False) -> Path:
    ctc = download_ctc(force=force)
    try:
        download_rnnt(force=force)
    except Exception as exc:  # noqa: BLE001
        log.warning("RNNT download skipped: %s", exc)
    print("Then: python -m src")
    return ctc


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
