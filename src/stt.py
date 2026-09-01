"""Persian speech-to-text via Whisper large Farsi v1 (vhdm), local CTranslate2."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src.config import config
from src.utils import log


def _looks_like_ct2(path: Path) -> bool:
    return (path / "model.bin").is_file()


class WhisperSTT:
    """Transcribe 16 kHz mono 16-bit PCM with vhdm/whisper-large-fa-v1."""

    def __init__(self) -> None:
        self.model_id = config.whisper_model_id
        self.model_path = config.whisper_model_path
        self.engine = "whisper-large-fa-v1"
        self._model = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        if _looks_like_ct2(self.model_path):
            self.last_error = None
        else:
            self.last_error = (
                f"Whisper CT2 model not found at {self.model_path}. "
                "Run: python -m src.download_whisper"
            )
            log.warning("%s", self.last_error)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def available(self) -> bool:
        return _looks_like_ct2(self.model_path) or self._model is not None

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            if not _looks_like_ct2(self.model_path):
                return False
            try:
                from faster_whisper import WhisperModel

                device = config.whisper_device
                compute = config.whisper_compute_type
                if device == "auto":
                    try:
                        import ctranslate2

                        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
                    except Exception:  # noqa: BLE001
                        device = "cpu"
                if device == "cpu" and compute in {"float16", "float32"}:
                    compute = "int8"
                log.info(
                    "Loading Whisper %s from %s device=%s compute=%s",
                    self.model_id,
                    self.model_path,
                    device,
                    compute,
                )
                self._model = WhisperModel(
                    str(self.model_path),
                    device=device,
                    compute_type=compute,
                )
                self.last_error = None
                log.info("Whisper large Farsi v1 ready")
                return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.error("Failed to load Whisper: %s", exc)
                self._model = None
                return False

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        if not audio_data:
            return ""
        if not self.ensure_loaded():
            log.warning("STT unavailable; skipping transcription of %d bytes", len(audio_data))
            return ""
        try:
            audio = _pcm16_to_float32(audio_data, sample_rate)
            if audio.size < 1600:
                return ""
            with self._lock:
                segments, _info = self._model.transcribe(
                    audio,
                    language="fa",
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                )
                text = "".join(seg.text for seg in segments).strip()
            log.info("STT transcript: %s", text)
            return text
        except Exception as exc:  # noqa: BLE001
            log.error("STT failed: %s", exc)
            return ""

    def transcribe_partial(self, recognizer, audio_chunk: bytes) -> tuple[str | None, str]:
        """Whisper is utterance-based; live path uses VAD then transcribe()."""
        return None, ""

    def make_recognizer(self, sample_rate: int = 16000):
        return None


def _pcm16_to_float32(audio_data: bytes, sample_rate: int) -> np.ndarray:
    pcm = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate == 16000 or pcm.size == 0:
        return pcm
    n = int(round(pcm.size * 16000 / sample_rate))
    if n <= 0:
        return pcm
    x_old = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


# Process-wide singleton — Whisper weights are heavy.
stt = WhisperSTT()
