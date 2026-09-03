"""The browser voice socket must answer even when Whisper is slow."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import db, live_voice
from src.call_manager import call_manager
from src.main import app


def _drain_for_assistant(ws, needle: str, limit: int = 12) -> list[dict]:
    """Collect socket traffic until the receptionist answers with `needle`."""
    seen: list[dict] = []
    for _ in range(limit):
        msg = ws.receive_json()
        seen.append(msg)
        if msg.get("event") == "assistant" and needle in (msg.get("text") or ""):
            return seen
    raise AssertionError(f"{needle!r} never arrived; saw {seen}")


def _speech(seconds: float, amplitude: float = 0.35, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    t = np.arange(n, dtype=np.float32) / rate
    tone = amplitude * (np.sin(2 * np.pi * 190 * t) + 0.5 * np.sin(2 * np.pi * 420 * t))
    return (np.clip(tone, -1, 1) * 32767).astype(np.int16).tobytes()


def _silence(seconds: float, rate: int = 16000) -> bytes:
    return np.zeros(int(seconds * rate), dtype=np.int16).tobytes()


@pytest.fixture()
def fake_stt(monkeypatch):
    """Whisper is far too slow to load in a test; stand in for it."""
    calls: list[bytes] = []

    def transcribe(chunk: bytes, sample_rate: int = 16000) -> str:
        calls.append(chunk)
        return "پژو پارس"

    monkeypatch.setattr(live_voice.stt, "transcribe", transcribe)
    monkeypatch.setattr(type(live_voice.stt), "available", property(lambda self: True))
    return calls


def test_speech_produces_a_dictation_and_a_reply(fake_stt, tmp_path, monkeypatch):
    monkeypatch.setattr(db.config, "db_path", tmp_path / "live.db")
    db.init_db(tmp_path / "live.db")
    call_manager.end_call("live-test")
    client = TestClient(app)
    with client.websocket_connect("/voice/live") as ws:
        ws.send_json({"event": "start", "session_id": "live-test", "sample_rate": 16000})
        for _ in range(20):
            ws.send_bytes(_silence(0.05))
        for _ in range(24):
            ws.send_bytes(_speech(0.05))
        for _ in range(30):
            ws.send_bytes(_silence(0.05))
        seen = _drain_for_assistant(ws, "پژو پارس")

    assert "dictation" in [m.get("event") for m in seen], seen
    assert fake_stt, "Whisper was never handed any audio"


def test_slow_whisper_still_reaches_the_browser(tmp_path, monkeypatch):
    """A multi-second transcription must not land after the socket is gone."""
    import time as _time

    def slow_transcribe(chunk: bytes, sample_rate: int = 16000) -> str:
        _time.sleep(0.6)
        return "سمند"

    monkeypatch.setattr(live_voice.stt, "transcribe", slow_transcribe)
    monkeypatch.setattr(type(live_voice.stt), "available", property(lambda self: True))
    monkeypatch.setattr(db.config, "db_path", tmp_path / "slow.db")
    db.init_db(tmp_path / "slow.db")

    call_manager.end_call("slow-test")
    client = TestClient(app)
    with client.websocket_connect("/voice/live") as ws:
        ws.send_json({"event": "start", "session_id": "slow-test", "sample_rate": 16000})
        for _ in range(20):
            ws.send_bytes(_silence(0.05))
        for _ in range(24):
            ws.send_bytes(_speech(0.05))
        for _ in range(30):
            ws.send_bytes(_silence(0.05))
        _drain_for_assistant(ws, "سمند")


def test_send_is_skipped_after_hangup():
    class Closed:
        client_state = live_voice.WebSocketState.DISCONNECTED

        async def send_json(self, payload):  # pragma: no cover - must not run
            raise AssertionError("sent on a closed socket")

    import asyncio

    assert asyncio.run(live_voice._send(Closed(), {"event": "assistant"})) is False
