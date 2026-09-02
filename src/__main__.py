"""Run the FastAPI app: python -m src

Download STT: python -m src download-shenava
"""

from __future__ import annotations

import sys


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in {"download-shenava", "download_shenava", "download-whisper", "download_whisper"}:
        from src.download_shenava import main

        raise SystemExit(main())

    from src.config import config
    from src.main import app
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())
