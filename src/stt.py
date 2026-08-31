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
        resolved = _find_vosk_root(self.model_path)
        if resolved is None:
            log.warning(
                "Vosk model not found at %s — unzip vosk-model-small-fa-0.5.zip "
                "so that models\\vosk-model-fa\\conf\\model.conf exists. "
                "On Windows run: powershell -File scripts\\download_models.ps1",
                self.model_path,
            )
            return
        self.model_path = resolved
        try:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            self._model = Model(str(self.model_path))
            self.available = True
            log.info("Loaded Vosk model from %s", self.model_path)
        except Exception as exc:  # noqa: BLE001 — model load must never crash the app
            log.error("Failed to load Vosk model: %s", exc)


def _looks_like_vosk(path: Path) -> bool:
    return (path / "am" / "final.mdl").exists() or (path / "conf" / "model.conf").exists()


def _find_vosk_root(path: Path) -> Path | None:
    """Accept either the model dir itself or a nested unzip folder (vosk-model-small-fa-0.5)."""
    if _looks_like_vosk(path):
        return path
    if not path.exists() or not path.is_dir():
        return None
    for child in sorted(path.iterdir()):
        if child.is_dir() and _looks_like_vosk(child):
            return child
        if child.is_dir():
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and _looks_like_vosk(grandchild):
                    return grandchild
    return None

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
