"""Audio conversion helpers, Persian date/time parsing, and logging setup."""

from __future__ import annotations

import audioop
import json
import logging
import re
import struct
import wave
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import config

# Twilio Media Streams send 8-bit μ-law (G.711) at 8 kHz.
MULAW_RATE = 8000
PCM16_RATE = 16000


def setup_logging() -> logging.Logger:
    """Configure rotating file + console logging. Safe to call more than once."""
    config.log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("persian_voice_agent")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        config.log_dir / "agent.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger


log = setup_logging()


def mulaw_to_pcm16(mulaw_bytes: bytes, target_rate: int = PCM16_RATE) -> bytes:
    """Decode μ-law 8 kHz audio to 16-bit PCM at target_rate (default 16 kHz)."""
    pcm8k, _ = audioop.ulaw2lin(mulaw_bytes, 2)
    if target_rate == MULAW_RATE:
        return pcm8k
    converted, _ = audioop.ratecv(pcm8k, 2, 1, MULAW_RATE, target_rate, None)
    return converted


def pcm16_to_mulaw(pcm_bytes: bytes, source_rate: int = PCM16_RATE) -> bytes:
    """Encode 16-bit PCM to 8 kHz μ-law for Twilio Media Streams."""
    if source_rate != MULAW_RATE:
        pcm_bytes, _ = audioop.ratecv(pcm_bytes, 2, 1, source_rate, MULAW_RATE, None)
    return audioop.lin2ulaw(pcm_bytes, 2)


def pcm16_rms(pcm_bytes: bytes) -> int:
    """Root-mean-square energy of 16-bit mono PCM. Used for simple VAD."""
    if len(pcm_bytes) < 2:
        return 0
    return audioop.rms(pcm_bytes, 2)


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = PCM16_RATE) -> bytes:
    """Wrap raw PCM in a WAV container."""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int = PCM16_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


def silence_pcm(duration_ms: int, sample_rate: int = PCM16_RATE) -> bytes:
    n_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def generate_tone_pcm(
    duration_ms: int = 250, freq: int = 440, sample_rate: int = PCM16_RATE, amplitude: int = 4000
) -> bytes:
    """Tiny beep used when Piper is not installed, so the stream still has audio."""
    import math

    n = int(sample_rate * duration_ms / 1000)
    frames = [
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)
    ]
    return struct.pack("<" + "h" * n, *frames)


_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_ARABIC_YEH = str.maketrans("يىك", "ییک")


def normalize_persian(text: str) -> str:
    text = (text or "").translate(_PERSIAN_DIGITS).translate(_ARABIC_YEH)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_YES = {"بله", "آره", "اره", "باشه", "تایید", "تأیید", "درسته", "اوکی", "ok", "yes"}
_NO = {"نه", "خیر", "نمیخوام", "نمی‌خوام", "غلط", "no"}
_TRANSFER = (
    "منشی",
    "اپراتور",
    "انسان",
    "آدم",
    "صحبت کنم",
    "وصل کنید",
    "انتقال",
    "reception",
    "human",
)


def is_yes(text: str) -> bool:
    t = normalize_persian(text).lower()
    return any(w in t for w in _YES)


def is_no(text: str) -> bool:
    t = normalize_persian(text).lower()
    return any(w in t for w in _NO) and not is_yes(text)


def wants_transfer(text: str) -> bool:
    t = normalize_persian(text).lower()
    return any(w in t for w in _TRANSFER)


def parse_relative_date(text: str, today: date | None = None) -> str | None:
    """Return ISO date (YYYY-MM-DD) for امروز/فردا/پس‌فردا or numeric dates."""
    today = today or date.today()
    t = normalize_persian(text)

    if "پس فردا" in t or "پس‌فردا" in t or "پسفردا" in t:
        return (today + timedelta(days=2)).isoformat()
    if "فردا" in t:
        return (today + timedelta(days=1)).isoformat()
    if "امروز" in t:
        return today.isoformat()

    m = re.search(r"(20\d{2}|13\d{2}|14\d{2})[/-](\d{1,2})[/-](\d{1,2})", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Jalali years are 13xx/14xx — store as a display string, not Gregorian.
        if y >= 1300 and y < 1600:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None

    m = re.search(r"(\d{1,2})[/-](\d{1,2})", t)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        try:
            year = today.year
            parsed = date(year, mo, d)
            if parsed < today:
                parsed = date(year + 1, mo, d)
            return parsed.isoformat()
        except ValueError:
            return None
    return None


_HOUR_WORDS = {
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
    "ده": 10,
    "یازده": 11,
    "دوازده": 12,
}


def parse_time(text: str) -> str | None:
    """Return HH:MM (24h) from Persian/English time expressions."""
    t = normalize_persian(text).lower()
    evening = any(w in t for w in ("عصر", "شب", "بعدازظهر", "بعد از ظهر", "pm"))
    morning = any(w in t for w in ("صبح", "قبل از ظهر", "am"))

    m = re.search(r"(\d{1,2})[:\.](\d{2})", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if evening and h < 12:
            h += 12
        if 0 <= h <= 23 and 0 <= mi <= 59:
            return f"{h:02d}:{mi:02d}"

    m = re.search(r"(?:ساعت\s*)?(\d{1,2})", t)
    hour = None
    if m:
        hour = int(m.group(1))
    else:
        for word, val in _HOUR_WORDS.items():
            if word in t:
                hour = val
                break
    if hour is None:
        return None
    if evening and hour < 12:
        hour += 12
    if morning and hour == 12:
        hour = 0
    if hour > 23:
        return None
    return f"{hour:02d}:00"


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM reply."""
    if not text:
        return None
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
