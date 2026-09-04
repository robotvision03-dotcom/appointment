"""nezamisafa/whisper-persian-v4 via faster-whisper (CTranslate2 int8)."""

from __future__ import annotations

import os
import re
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


_PERSIAN = re.compile(r"[\u0600-\u06FF]")
# Any letter that is not Persian/Arabic. Whisper slips into Latin («Pژو») and
# Cyrillic («ежоپارس») mid-word; neither belongs in a spoken Persian name.
_FOREIGN_LETTER = re.compile(r"[^\W\d_\u0600-\u06FF]")
# Junk is punctuation with no letter or digit. Digits are real answers:
# a model year, a mileage, or a phone number is often all the seller says.
_JUNK = re.compile(r"^\W*$")


def _persian_only_tokens(tokenizer) -> list[int]:
    """Token ids that carry a Latin, Cyrillic, or Greek letter.

    This fine-tune sometimes transliterates a whole word — «پژو پارس» came back
    as «ежоپарс» — and no amount of post-editing recovers that. Suppressing the
    foreign tokens makes the decoder unable to leave Persian, and costs nothing.
    """
    foreign = re.compile(r"[A-Za-z\u0370-\u03FF\u0400-\u04FF]")
    ids = [-1]  # keep Whisper's own non-speech suppressions
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id])
        if text and foreign.search(text):
            ids.append(token_id)
    return ids


def _auto_threads() -> int:
    """Physical cores, capped. Oversubscribing CTranslate2 costs more than it buys."""
    cores = os.cpu_count() or 4
    return max(1, min(4, cores // 2 if cores > 4 else cores))


def clean_transcript(text: str) -> str:
    """Drop Whisper punctuation and Latin letters glued into Persian words.

    Whisper writes «Pژو پانس.» when it slips into the Latin alphabet mid-word;
    the Latin letter is never part of a spoken Persian car name.
    """
    if not text:
        return ""
    out: list[str] = []
    for token in text.replace("\u200c", " ").split():
        stripped = token.strip(".,!?;:،؛؟«»\"'()[]…")
        if not stripped:
            continue
        if _PERSIAN.search(stripped) and _FOREIGN_LETTER.search(stripped):
            stripped = _FOREIGN_LETTER.sub("", stripped)
        if stripped:
            out.append(stripped)
    joined = " ".join(out).strip()
    return "" if _JUNK.match(joined) else joined


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
        # Car-name bias is what made 7057a98 accurate on «پرس» → پژو پارس.
        self._prompt = (config.whisper_prompt or "").strip() or _car_prompt()
        self._suppress: list[int] | None = None
        if not self.available:
            self.last_error = (
                f"Whisper Persian v4 not found at {self.model_path}. "
                "Run: python -m src download-whisper"
            )

    @property
    def runtime_installed(self) -> bool:
        from importlib.util import find_spec

        return find_spec("faster_whisper") is not None

    @property
    def available(self) -> bool:
        p = self.model_path
        has_files = (p / "model.bin").is_file() and (
            (p / "vocabulary.json").is_file() or (p / "tokenizer.json").is_file()
        )
        return has_files and self.runtime_installed

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
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    self.last_error = (
                        "faster-whisper نصب نیست. اجرا کنید: "
                        "pip install -r requirements.txt"
                    )
                    log.error("%s", self.last_error)
                    return False

                threads = max(1, int(config.whisper_threads) or _auto_threads())
                kwargs = dict(
                    device="cpu",
                    compute_type=config.whisper_compute or "int8",
                    cpu_threads=threads,
                    num_workers=1,
                )
                # Skip the Hub round-trip that printed
                # "You are sending unauthenticated requests to the HF Hub"
                # on every cold start and delayed the first decode ~15s.
                try:
                    self._model = WhisperModel(
                        str(self.model_path), local_files_only=True, **kwargs
                    )
                except TypeError:
                    self._model = WhisperModel(str(self.model_path), **kwargs)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Whisper local-only load failed (%s); retrying", exc)
                    self._model = WhisperModel(str(self.model_path), **kwargs)
                if config.whisper_persian_only:
                    tokenizer = getattr(self._model, "hf_tokenizer", None)
                    if tokenizer is not None:
                        self._suppress = _persian_only_tokens(tokenizer)
                        log.warning(
                            "WHISPER_PERSIAN_ONLY=1 suppresses %s tokens — "
                            "this is slow on CPU. Leave it off unless needed.",
                            len(self._suppress) - 1,
                        )
                self.last_error = None
                log.info(
                    "Hearing: %s (%s) %s threads=%s persian_only=%s",
                    self.engine,
                    self.model_id,
                    self.model_path,
                    threads,
                    len(self._suppress) - 1 if self._suppress else 0,
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
            cap = max(3200, int(config.whisper_max_seconds) * 16000)
            if samples.size > cap:
                samples = samples[-cap:]
            decode = dict(
                language="fa",
                task="transcribe",
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                initial_prompt=self._prompt or None,
                temperature=0.0,
            )
            if self._suppress:
                decode["suppress_tokens"] = self._suppress
            with self._lock:
                segments, info = self._model.transcribe(samples, **decode)
                text = clean_transcript("".join(seg.text for seg in segments))
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
