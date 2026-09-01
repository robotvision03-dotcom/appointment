"""Run the FastAPI app: python -m src

Also: python -m src download-whisper
"""

from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"download-whisper", "download_whisper"}:
        from src.download_whisper import main

        raise SystemExit(main())

    from src.config import config
    from src.main import app
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())
