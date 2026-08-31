"""Offline Persian speech-to-text via Vosk, with a graceful fallback."""

from __future__ import annotations

import json
from pathlib import Path

from src.config import config
from src.utils import log


class OfflineSTT:
    """Transcribe 16 kHz mono 16-bit PCM using a local Vosk Persian model."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = Path(model_path or config.vosk_model_path)
        self._model = None
        self.available = False
        self._load()

    def _load(self) -> None:
        model_ok = (self.model_path / "am" / "final.mdl").exists() or (
            self.model_path / "conf" / "model.conf"
        ).exists()
        if not model_ok:
            log.warning(
                "Vosk model not found at %s — STT will return empty transcripts. "
                "Run scripts/download_models.sh",
                self.model_path,
            )
            return
        try:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            self._model = Model(str(self.model_path))
            self.available = True
            log.info("Loaded Vosk model from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001 — model load must never crash the app
            log.error("Failed to load Vosk model: %s", exc)

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """Accept raw PCM (16 kHz, mono, 16-bit) and return a Persian transcript."""
        if not audio_data:
            return ""
        if not self.available or self._model is None:
            log.warning("STT unavailable; skipping transcription of %d bytes", len(audio_data))
            return ""
        try:
            from vosk import KaldiRecognizer

            rec = KaldiRecognizer(self._model, sample_rate)
            rec.SetWords(True)
            # Feed in chunks so Vosk can run its pipeline
            chunk = 4000
            for i in range(0, len(audio_data), chunk):
                rec.AcceptWaveform(audio_data[i : i + chunk])
            result = json.loads(rec.FinalResult())
            text = (result.get("text") or "").strip()
            log.info("STT transcript: %s", text)
            return text
        except Exception as exc:  # noqa: BLE001
            log.error("STT failed: %s", exc)
            return ""

    def transcribe_partial(self, recognizer, audio_chunk: bytes) -> tuple[str | None, str]:
        """
        Incremental API for a live stream.

        Returns (final_text_or_None, partial_text).
        """
        if recognizer is None:
            return None, ""
        try:
            if recognizer.AcceptWaveform(audio_chunk):
                result = json.loads(recognizer.Result())
                return (result.get("text") or "").strip(), ""
            partial = json.loads(recognizer.PartialResult()).get("partial") or ""
            return None, partial.strip()
        except Exception as exc:  # noqa: BLE001
            log.error("Incremental STT failed: %s", exc)
            return None, ""

    def make_recognizer(self, sample_rate: int = 16000):
        if not self.available or self._model is None:
            return None
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, sample_rate)
        rec.SetWords(True)
        return rec


# Process-wide singleton — Vosk models are heavy.
stt = OfflineSTT()
