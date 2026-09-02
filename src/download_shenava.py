"""Download Shenava-Koochik-v1.5 for local sherpa-onnx.

CTC head of v1.5 is identical to the published sherpa CTC export
(Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx) and is the stronger head (8.12% WER).
RNNT int8 from Reza2kn/Shenava-Koochik-v1.5-RNNT-sherpa-onnx is a fallback.

Usage: python -m src download-shenava
"""

from __future__ import annotations

from src.config import config
from src.utils import log


def download(force: bool = False) -> None:
    dest = config.shenava_model_path
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    ctc_ok = (dest / "model.onnx").is_file() and (dest / "tokens.txt").is_file()
    if not ctc_ok or force:
        log.info("Downloading Shenava-Koochik CTC (v1.5 CTC head, 8.12%% WER)")
        for name in ("model.onnx", "tokens.txt"):
            path = hf_hub_download(
                "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx",
                name,
                local_dir=str(dest),
            )
            log.info("saved %s", path)
    else:
        log.info("CTC already at %s", dest / "model.onnx")

    print("OK", dest)
    print("Then: python -m src")


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
