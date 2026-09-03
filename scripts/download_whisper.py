"""Windows-friendly: py scripts/download_whisper.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download_whisper import main

if __name__ == "__main__":
    raise SystemExit(main())
