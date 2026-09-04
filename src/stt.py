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


class HearingSTT:
    """Whisper Persian v4 first; Shenava if the checkpoint is missing; Gooya if licensed."""

    def __init__(self) -> None:
        from src.gooya import GooyaSTT
        from src.whisper_fa import WhisperPersianSTT

        self.gooya = GooyaSTT()
        self.whisper = WhisperPersianSTT()
        self.shenava = ShenavaSTT()
        self._last_engine = ""
        self.last_error: str | None = None
        prefer = config.stt_engine
        if prefer not in {"auto", "gooya", "shenava", "whisper"}:
            prefer = "whisper"
        self.mode = prefer
        if self.mode in {"whisper", "auto"} and not self.whisper.available:
            if not self.whisper.runtime_installed:
                log.warning(
                    "Whisper Persian v4 skipped: faster-whisper is not installed. "
                    "Run: pip install -r requirements.txt"
                )
            else:
                log.warning(
                    "Whisper Persian v4 weights missing. Run: python -m src download-whisper"
                )

    @property
    def engine(self) -> str:
        if self._last_engine:
            return self._last_engine
        if self.mode in {"whisper", "auto"} and self.whisper.available:
            return self.whisper.engine
        if self.mode in {"auto", "gooya"} and self.gooya.configured:
            return self.gooya.engine
        return self.shenava.engine

    @property
    def head(self) -> str:
        if self.engine.startswith("whisper"):
            return getattr(self.whisper, "head", "faster-whisper")
        if self.engine.startswith("gooya"):
            return "http-api"
        return getattr(self.shenava, "head", "") or ""

    @property
    def loaded(self) -> bool:
        return self.whisper.loaded or self.gooya.configured or self.shenava.loaded

    @property
    def available(self) -> bool:
        return self.whisper.available or self.gooya.available or self.shenava.available

    def ensure_loaded(self) -> bool:
        if self.mode in {"whisper", "auto"} and self.whisper.available:
            if self.whisper.ensure_loaded():
                self.last_error = None
                return True
        if self.mode in {"auto", "gooya"} and self.gooya.configured:
            if self.gooya.ensure_loaded():
                return True
        ok = self.shenava.ensure_loaded()
        self.last_error = (
            self.whisper.last_error or self.gooya.last_error or self.shenava.last_error
        )
        return ok

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        if not audio_data:
            return ""
        order: list[tuple[str, object]] = []
        if self.mode == "whisper":
            # Whisper only. Cascading into Shenava on an empty result doubled
            # latency (12s clip → ~90s) and overwrote a good-enough transcript
            # with CTC garbage like «پارس پو پارس».
            if self.whisper.available:
                order = [("whisper", self.whisper)]
            else:
                order = [("shenava", self.shenava)]
        elif self.mode == "gooya":
            order = [("gooya", self.gooya), ("whisper", self.whisper), ("shenava", self.shenava)]
        elif self.mode == "shenava":
            order = [("shenava", self.shenava)]
        else:
            order = [
                ("whisper", self.whisper),
                ("gooya", self.gooya),
                ("shenava", self.shenava),
            ]
        last_err = None
        for name, engine in order:
            if name == "gooya" and not getattr(engine, "configured", False):
                continue
            if name != "gooya" and not getattr(engine, "available", True):
                continue
            text = engine.transcribe(audio_data, sample_rate)
            if text:
                self._last_engine = getattr(engine, "engine", name)
                self.last_error = None
                return text
            last_err = getattr(engine, "last_error", None)
        self.last_error = last_err
        return ""

    def transcribe_partial(self, recognizer, audio_chunk: bytes) -> tuple[str | None, str]:
        return None, ""

    def make_recognizer(self, sample_rate: int = 16000):
        return None


stt = HearingSTT()
