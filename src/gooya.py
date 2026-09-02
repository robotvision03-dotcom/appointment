"""Gooya v1.4 Persian ASR — commercial HTTP API (weights are not public).

The official demo Space (navidved/gooya-asr) does not ship checkpoints; it
POSTs audio to ASR_API_URL with a Bearer token. We use the same contract:

  POST {GOOYA_API_URL}
  Authorization: Bearer {GOOYA_API_TOKEN}
  multipart file=<16 kHz wav>
  JSON { "transcription": "..." }

Get a license from the author (navidved@gmail.com / Hugging Face navidved),
then set those two env vars. Without them this client is inactive and Shenava
stays the on-device engine.
"""

from __future__ import annotations

import io
import threading
import wave
from typing import Any

import httpx
import numpy as np

from src.config import config
from src.hearing import prepare_for_asr
from src.utils import log


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


def _pcm16_wav_bytes(pcm16: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    n = len(pcm16) // 2
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16[: n * 2])
    return buf.getvalue()


def _extract_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        text = payload.strip()
        if text.lower().startswith("error"):
            return ""
        return text
    if isinstance(payload, dict):
        for key in ("transcription", "text", "result", "output"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip() and not val.lower().startswith("error"):
                return val.strip()
        if isinstance(payload.get("data"), list) and payload["data"]:
            return _extract_text(payload["data"][0])
    if isinstance(payload, list) and payload:
        return _extract_text(payload[0])
    return ""


class GooyaSTT:
    """Remote Gooya v1.4 — same request shape as navidved/gooya-asr."""

    engine = "gooya-v1.4"

    def __init__(self) -> None:
        self.api_url = (config.gooya_api_url or "").rstrip("/")
        self.token = config.gooya_api_token
        self.timeout = float(config.gooya_timeout_s)
        self.last_error: str | None = None
        self._lock = threading.Lock()
        if not self.configured:
            self.last_error = (
                "Gooya v1.4 weights are not public. Set GOOYA_API_URL and "
                "GOOYA_API_TOKEN from the vendor, then: python -m src download-gooya"
            )

    @property
    def configured(self) -> bool:
        return bool(self.api_url and self.token)

    @property
    def available(self) -> bool:
        return self.configured

    @property
    def loaded(self) -> bool:
        return self.configured

    def ensure_loaded(self) -> bool:
        if self.configured:
            self.last_error = None
            return True
        return False

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        if not audio_data or not self.ensure_loaded():
            return ""
        try:
            audio = _to_16k_float(audio_data, sample_rate)
            if audio.size < 3200:
                return ""
            raw_rms = float(np.sqrt(np.mean(np.square(audio))))
            if raw_rms < 0.006:
                return ""
            prepared = prepare_for_asr(audio, 16000)
            pcm = np.clip(prepared * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            wav = _pcm16_wav_bytes(pcm, 16000)
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {self.token}",
            }
            files = {"file": ("utterance.wav", wav, "audio/wav")}
            with self._lock:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.api_url, headers=headers, files=files)
            if response.status_code != 200:
                self.last_error = f"Gooya HTTP {response.status_code}: {response.text[:240]}"
                log.warning("%s", self.last_error)
                return ""
            try:
                payload: Any = response.json()
            except Exception:
                payload = response.text
            text = _extract_text(payload)
            self.last_error = None
            log.info("Gooya v1.4 transcript=%r samples=%s rms=%.4f", text, int(audio.size), raw_rms)
            return text
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            log.error("Gooya v1.4 failed: %s", exc)
            return ""
