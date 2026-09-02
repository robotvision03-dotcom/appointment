"""Alias so older docs still work: python -m src.download_whisper"""

from src.download_shenava import main

if __name__ == "__main__":
    raise SystemExit(main())
