"""Verify Gooya v1.4 API credentials. Weights cannot be downloaded.

  python -m src download-gooya
"""

from __future__ import annotations

import io
import math
import struct
import wave

from src.config import config
from src.gooya import GooyaSTT, _extract_text
from src.utils import log


def _tone_wav() -> bytes:
    rate = 16000
    n = rate
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = [
            int(0.12 * 32767 * math.sin(2 * math.pi * 220 * i / rate))
            for i in range(n)
        ]
        wf.writeframes(struct.pack("<" + "h" * n, *frames))
    return buf.getvalue()


def main() -> int:
    print("Gooya v1.4 is not on Hugging Face as open weights.")
    print("The published demo Space only forwards audio to a private ASR_API_URL.")
    print("License: navidved@gmail.com  ·  Space: https://huggingface.co/spaces/navidved/gooya-asr")
    print()
    if not config.gooya_api_url or not config.gooya_api_token:
        print("Not configured. In .env set:")
        print("  STT_ENGINE=auto")
        print("  GOOYA_API_URL=https://<vendor-host>/transcribe")
        print("  GOOYA_API_TOKEN=<bearer token>")
        print("Until then the office uses Shenava CTC on this machine.")
        return 2
    client = GooyaSTT()
    import httpx

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {config.gooya_api_token}",
    }
    files = {"file": ("probe.wav", _tone_wav(), "audio/wav")}
    try:
        response = httpx.post(
            config.gooya_api_url,
            headers=headers,
            files=files,
            timeout=config.gooya_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Gooya probe failed: %s", exc)
        print("Request failed:", exc)
        return 1
    print("HTTP", response.status_code)
    print(response.text[:800])
    if response.status_code != 200:
        return 1
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    text = _extract_text(payload)
    print("transcription:", text or "(empty)")
    print("client.configured:", client.configured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
