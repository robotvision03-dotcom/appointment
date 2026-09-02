"""Persian speech-to-text via Shenava-Koochik-v1.5 (sherpa-onnx)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src.config import config
from src.hearing import prepare_for_asr
from src.utils import log


def _ctc_ready(path: Path) -> bool:
    return (path / "model.onnx").is_file() and (path / "tokens.txt").is_file()


def _rnnt_ready(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    )


def _sherpa_text(raw) -> str:
    """Only the transcript string — never stringify the whole sherpa result object."""
    if raw is None:
        return ""
    text = getattr(raw, "text", None)
    if text is None and isinstance(raw, dict):
        text = raw.get("text")
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith("{") and "ys_log_probs" in text:
        return ""
    return text


def _prepare_waveform(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    return prepare_for_asr(audio, sample_rate)


class ShenavaSTT:
    """16 kHz mono PCM → Persian text via Shenava-Koochik-v1.5."""

    def __init__(self) -> None:
        self.model_id = config.shenava_model_id
        self.model_path = config.shenava_model_path
        self.engine = "shenava-koochik-v1.5"
        self._recognizer = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.head = ""
        if (
            not _ctc_ready(config.shenava_ctc_path)
            and not _ctc_ready(self.model_path)
            and not _rnnt_ready(self.model_path)
        ):
            self.last_error = (
                f"Shenava not found at {self.model_path}. "
                "Run: python -m src download-shenava"
            )
            log.warning("%s", self.last_error)

    @property
    def loaded(self) -> bool:
        return self._recognizer is not None

    @property
    def available(self) -> bool:
        return (
            _ctc_ready(config.shenava_ctc_path)
            or _ctc_ready(self.model_path)
            or _rnnt_ready(self.model_path)
            or self._recognizer is not None
        )

    def ensure_loaded(self) -> bool:
        if self._recognizer is not None:
            return True
        with self._lock:
            if self._recognizer is not None:
                return True
            try:
                import sherpa_onnx
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"sherpa-onnx missing: {exc}"
                log.error("%s", self.last_error)
                return False
            try:
                threads = max(1, int(config.shenava_threads))
                prefer_ctc = config.shenava_head != "rnnt"
                ctc_dir = (
                    config.shenava_ctc_path
                    if _ctc_ready(config.shenava_ctc_path)
                    else self.model_path
                    if _ctc_ready(self.model_path)
                    else None
                )
                if prefer_ctc and ctc_dir is not None:
                    self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                        model=str(ctc_dir / "model.onnx"),
                        tokens=str(ctc_dir / "tokens.txt"),
                        num_threads=threads,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                    )
                    self.head = "ctc"
                    self.engine = "shenava-koochik-ctc"
                    log.info("Hearing: Shenava CTC (recommended) %s", ctc_dir)
                elif _rnnt_ready(self.model_path):
                    common = dict(
                        encoder=str(self.model_path / "encoder.int8.onnx"),
                        decoder=str(self.model_path / "decoder.int8.onnx"),
                        joiner=str(self.model_path / "joiner.int8.onnx"),
                        tokens=str(self.model_path / "tokens.txt"),
                        num_threads=threads,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                    )
                    # Official export requires nemo_transducer. "nemo" is invalid and
                    # sherpa then guesses — often decoding blank transcripts.
                    try:
                        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                            **common,
                            model_type="nemo_transducer",
                        )
                        self.head = "rnnt-nemo_transducer"
                    except Exception as exc:  # noqa: BLE001
                        log.warning("nemo_transducer load failed (%s); retrying without type", exc)
                        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**common)
                        self.head = "rnnt-auto"
                    log.info("Hearing: Shenava-Koochik-v1.5 %s %s", self.head, self.model_path)
                else:
                    self.last_error = (
                        f"Shenava files missing in {self.model_path}. "
                        "Run: python -m src download-shenava"
                    )
                    return False
                self.last_error = None
                return True
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.error("Failed to load Shenava: %s", exc)
                self._recognizer = None
                return False

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        if not audio_data:
            return ""
        if not self.ensure_loaded():
            log.warning("STT unavailable; skipping transcription of %d bytes", len(audio_data))
            return ""
        try:
            audio = _pcm16_to_float32(audio_data, sample_rate)
            if audio.size < 3200:
                log.info("STT skip short audio samples=%s", audio.size)
                return ""
            raw_rms = float(np.sqrt(np.mean(np.square(audio))))
            if raw_rms < 0.006:
                log.info("STT skip quiet rms=%.5f", raw_rms)
                return ""
            samples = _prepare_waveform(audio, 16000)
            waveform = samples.astype(np.float32, copy=False).ravel().tolist()
            with self._lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(16000, waveform)
                self._recognizer.decode_stream(stream)
                text = _sherpa_text(stream.result)
            log.info(
                "STT transcript=%r samples=%s rms=%.4f head=%s in_rate=%s",
                text,
                int(audio.size),
                raw_rms,
                self.head,
                sample_rate,
            )
            return text
        except Exception as exc:  # noqa: BLE001
            log.error("STT failed: %s", exc)
            return ""

    def transcribe_partial(self, recognizer, audio_chunk: bytes) -> tuple[str | None, str]:
        return None, ""

    def make_recognizer(self, sample_rate: int = 16000):
        return None


def _pcm16_to_float32(audio_data: bytes, sample_rate: int) -> np.ndarray:
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


stt = ShenavaSTT()
