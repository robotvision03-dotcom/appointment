"""nezamisafa/whisper-persian-v4 via faster-whisper (CTranslate2 int8)."""

from __future__ import annotations

import threading

import numpy as np

from src.cars import CATALOG
from src.config import config
from src.hearing import prepare_for_asr
from src.utils import log

ENGINE = "whisper-persian-v4"


def _car_prompt() -> str:
    names: list[str] = []
    seen: set[str] = set()
    for make, model, extra in CATALOG:
        for tok in (make, model, *extra.split()[:3]):
            t = tok.strip()
            if len(t) < 2 or t.isascii() and t.isupper():
                continue
            key = t.replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            names.append(t)
            if len(names) >= 40:
                return " ".join(names)
    return " ".join(names)


def _to_16k_float(audio_data: bytes, sample_rate: int) -> np.ndarray:
    import audioop

    raw = bytes(audio_data)
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return np.zeros(0, dtype=np.float32)
    rate = int(sample_rate or 16000)
    if rate != 16000 and rate >= 8000:
        raw, _ = audioop.ratecv(raw, 2, 1, rate, 16000, None)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperPersianSTT:
    """Fine-tuned Whisper large-v3 Persian (nezamisafa/whisper-persian-v4)."""

    engine = ENGINE

    def __init__(self) -> None:
        self.model_id = config.whisper_model_id
        self.model_path = config.whisper_model_path
        self.head = "faster-whisper-int8"
        self.last_error: str | None = None
        self._model = None
        self._lock = threading.Lock()
        self._prompt = _car_prompt()
        if not self.available:
            self.last_error = (
                f"Whisper Persian v4 not found at {self.model_path}. "
                "Run: python -m src download-whisper"
            )

    @property
    def available(self) -> bool:
        p = self.model_path
        return (p / "model.bin").is_file() and (
            (p / "vocabulary.json").is_file() or (p / "tokenizer.json").is_file()
        )

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if not self.available:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from faster_whisper import WhisperModel

                threads = max(1, int(config.whisper_threads))
                self._model = WhisperModel(
                    str(self.model_path),
                    device="cpu",
                    compute_type=config.whisper_compute or "int8",
                    cpu_threads=threads,
                    num_workers=1,
                )
                self.last_error = None
                log.info(
                    "Hearing: %s (%s) %s threads=%s",
                    self.engine,
                    self.model_id,
                    self.model_path,
                    threads,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.error("Failed to load Whisper Persian v4: %s", exc)
                self._model = None
                return False

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        if not audio_data:
            return ""
        if not self.ensure_loaded():
            log.warning("Whisper unavailable; %s", self.last_error)
            return ""
        try:
            audio = _to_16k_float(audio_data, sample_rate)
            if audio.size < 3200:
                return ""
            raw_rms = float(np.sqrt(np.mean(np.square(audio))))
            if raw_rms < 0.006:
                return ""
            samples = prepare_for_asr(audio, 16000).astype(np.float32, copy=False).ravel()
            with self._lock:
                segments, info = self._model.transcribe(
                    samples,
                    language="fa",
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                    without_timestamps=True,
                    initial_prompt=self._prompt,
                    temperature=0.0,
                )
                text = "".join(seg.text for seg in segments).strip()
            lang = getattr(info, "language", "fa")
            log.info(
                "Whisper v4 transcript=%r lang=%s samples=%s rms=%.4f",
                text,
                lang,
                int(audio.size),
                raw_rms,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            log.error("Whisper v4 failed: %s", exc)
            return ""
