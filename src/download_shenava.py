"""Download Shenava-Koochik-v1.5 for local hearing (sherpa-onnx).

Canonical model: Reza2kn/Shenava-Koochik-v1.5
Runtime: Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx (int8)
Plus CTC export (same CTC head as v1.5, 8.12% WER) when available.

Usage: python -m src download-shenava
"""

from __future__ import annotations

from src.config import config
from src.utils import log

V15_RNNT = "Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx"
V15_CTC = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"  # CTC head is identical to v1.5


def _pull(repo: str, names: tuple[str, ...], dest) -> None:
    from huggingface_hub import hf_hub_download

    for name in names:
        path = hf_hub_download(repo, name, local_dir=str(dest))
        log.info("saved %s", path)


def download(force: bool = False) -> None:
    dest = config.shenava_model_path
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.txt").write_text(
        "Reza2kn/Shenava-Koochik-v1.5\n" + V15_RNNT + "\n",
        encoding="utf-8",
    )

    rnnt_ok = all(
        (dest / n).is_file()
        for n in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    )
    if not rnnt_ok or force:
        log.info("Downloading Shenava-Koochik-v1.5 RNNT (sherpa-onnx int8)")
        _pull(
            V15_RNNT,
            ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
            dest,
        )
    else:
        log.info("v1.5 RNNT already at %s", dest)

    ctc_ok = (dest / "model.onnx").is_file()
    if not ctc_ok or force:
        log.info("Downloading Shenava-Koochik-v1.5 CTC head (stronger WER)")
        _pull(V15_CTC, ("model.onnx",), dest)
    else:
        log.info("CTC already at %s", dest / "model.onnx")

    print("OK Shenava-Koochik-v1.5", dest)
    print("Then: python -m src")


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
