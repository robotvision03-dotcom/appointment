"""Download nezamisafa/whisper-persian-v4 for faster-whisper (CTranslate2).

Official weights are ~6GB float32. Runtime uses the CTranslate2 int8 export of
that same fine-tune so CPU hearing stays practical:

  python -m src download-whisper
"""

from __future__ import annotations

from pathlib import Path

from src.config import config
from src.utils import log

OFFICIAL_ID = "nezamisafa/whisper-persian-v4"
CT2_ID = "AlexAnoshka/fast-whisper-persian-v4"

NEEDED = ("model.bin", "config.json", "vocabulary.json", "tokenizer_config.json")


def _ready(path: Path) -> bool:
    return (path / "model.bin").is_file() and (
        (path / "vocabulary.json").is_file() or (path / "tokenizer.json").is_file()
    )


def download(force: bool = False) -> Path:
    from importlib.util import find_spec

    if find_spec("faster_whisper") is None:
        print("faster-whisper is not installed. Run first:")
        print("  pip install -r requirements.txt")
        raise SystemExit(2)

    dest = config.whisper_model_path
    dest.mkdir(parents=True, exist_ok=True)
    if _ready(dest) and not force:
        log.info("Whisper Persian v4 already at %s", dest)
        print(f"Ready: {dest}")
        return dest

    from huggingface_hub import snapshot_download

    log.info("Downloading CTranslate2 int8 of %s from %s → %s", OFFICIAL_ID, CT2_ID, dest)
    snapshot_download(
        repo_id=config.whisper_ct2_repo or CT2_ID,
        local_dir=str(dest),
    )
    if not _ready(dest):
        raise RuntimeError(f"Whisper download incomplete in {dest}")
    print(f"Official model: {OFFICIAL_ID}")
    print(f"Runtime (CT2 int8): {dest}")
    print("Then: python -m src")
    return dest


def main() -> int:
    download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
