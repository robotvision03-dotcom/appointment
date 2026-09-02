"""Persian speech-to-text via Shenava-Koochik-v1.5 (sherpa-onnx)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from src.config import config
from src.utils import log


def _ctc_ready(path: Path) -> bool:
    return (path / "model.onnx").is_file() and (path / "tokens.txt").is_file()


def _rnnt_ready(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
    )


class ShenavaSTT:
    """16 kHz mono PCM → Persian text. CTC head of Koochik v1.5 (8.12% WER)."""

    def __init__(self) -> None:
        self.model_id = config.shenava_model_id
        self.model_path = config.shenava_model_path
        self.engine = "shenava-koochik-v1.5"
        self._recognizer = None
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self.head = ""
        if not _ctc_ready(self.model_path) and not _rnnt_ready(self.model_path):
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
        return _ctc_ready(self.model_path) or _rnnt_ready(self.model_path) or self._recognizer is not None

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
                want_ctc = config.shenava_head == "ctc" and _ctc_ready(self.model_path)
                if want_ctc:
                    self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                        model=str(self.model_path / "model.onnx"),
                        tokens=str(self.model_path / "tokens.txt"),
                        num_threads=threads,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                    )
                    self.head = "ctc"
                    log.info("Hearing: Shenava-Koochik-v1.5 CTC %s", self.model_path)
                elif _rnnt_ready(self.model_path):
                    self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                        encoder=str(self.model_path / "encoder.int8.onnx"),
                        decoder=str(self.model_path / "decoder.int8.onnx"),
                        joiner=str(self.model_path / "joiner.int8.onnx"),
                        tokens=str(self.model_path / "tokens.txt"),
                        num_threads=threads,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                        model_type="nemo",
                    )
                    self.head = "rnnt"
                    log.info("Hearing: Shenava-Koochik-v1.5 RNNT %s", self.model_path)
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
            if audio.size < 2400:
                return ""
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms < 0.01:
                return ""
            with self._lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(16000, audio)
                self._recognizer.decode_stream(stream)
                raw = stream.result
                text = (getattr(raw, "text", None) or str(raw or "")).strip()
            log.info("STT transcript: %s", text)
            return text
        except Exception as exc:  # noqa: BLE001
            log.error("STT failed: %s", exc)
            return ""

    def transcribe_partial(self, recognizer, audio_chunk: bytes) -> tuple[str | None, str]:
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


stt = ShenavaSTT()
