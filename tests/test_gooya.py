"""Gooya v1.4 HTTP client (vendor API; no public checkpoints)."""

from __future__ import annotations

import httpx

from src.gooya import GooyaSTT, _extract_text


def test_extract_text_skips_vendor_errors():
    assert _extract_text({"transcription": "پژو پارس"}) == "پژو پارس"
    assert _extract_text("Error: Missing ASR_API_URL or AUTH_TOKEN.") == ""
    assert _extract_text({"text": "سمند"}) == "سمند"


def test_gooya_posts_wav_and_reads_transcription(monkeypatch):
    import math
    import struct

    from src import gooya as g

    monkeypatch.setattr(g.config, "gooya_api_url", "https://asr.example.test/v1")
    monkeypatch.setattr(g.config, "gooya_api_token", "secret-token")
    monkeypatch.setattr(g.config, "gooya_timeout_s", 5.0)

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"transcription": "پژو پارس"})

    transport = httpx.MockTransport(handler)

    class FakeClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(g.httpx, "Client", FakeClient)
    engine = GooyaSTT()
    assert engine.configured
    n = 16000
    pcm = struct.pack(
        "<" + "h" * n,
        *[int(8000 * math.sin(2 * math.pi * 220 * i / n)) for i in range(n)],
    )
    text = engine.transcribe(pcm, sample_rate=16000)
    assert text == "پژو پارس"
    assert seen["auth"] == "Bearer secret-token"
    assert "asr.example.test" in seen["url"]


def test_gooya_inactive_without_secrets(monkeypatch):
    from src import gooya as g

    monkeypatch.setattr(g.config, "gooya_api_url", "")
    monkeypatch.setattr(g.config, "gooya_api_token", "")
    engine = GooyaSTT()
    assert not engine.configured
    assert engine.transcribe(b"\x00\x40" * 16000) == ""
