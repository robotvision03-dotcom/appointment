"""TTS disabled: replies are text-only in the frontend."""

from __future__ import annotations

from pathlib import Path

from src.config import ROOT_DIR
from src.utils import log, write_wav

GENERATED_DIR = ROOT_DIR / "static" / "generated"


class DisabledTTS:
    available = False
    sample_rate = 16000

    def __init__(self) -> None:
        log.info("TTS disabled — assistant replies as text only")

    def synthesize(self, text: str) -> bytes:
        return b""

    def synthesize_wav(self, text: str) -> bytes:
        return b""

    def synthesize_to_file(self, text: str, filename: str | None = None) -> Path:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        path = GENERATED_DIR / (filename or "empty.wav")
        write_wav(path, b"\x00\x00", sample_rate=self.sample_rate)
        return path


tts = DisabledTTS()
