#!/usr/bin/env python3
"""Exercise STT → dialogue → TTS locally (models optional)."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402
from src.call_manager import call_manager  # noqa: E402
from src.stt import stt  # noqa: E402
from src.tts import tts  # noqa: E402
from src.utils import log  # noqa: E402


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "expected mono WAV"
        assert wf.getsampwidth() == 2, "expected 16-bit PCM"
        return wf.readframes(wf.getnframes()), wf.getframerate()


def run_dialogue() -> None:
    db.init_db()
    sid = "test-local-pipeline"
    call_manager.end_call(sid)
    call_manager.start_call(sid, from_number="+989120000002")
    print("AGENT:", call_manager.greeting())
    script = [
        "مکانیک",
        "اول",
        "۰۹۱۲۱۲۳۴۵۶۷",
    ]
    for line in script:
        print("USER:", line)
        result = call_manager.handle_user_text(sid, line)
        print("AGENT:", result["reply"], "| phase=", result["phase"], "| intent=", result["intent"])
        if result.get("connect"):
            print("CONNECT", result["connect"].get("provider", {}).get("name"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=Path, default=None, help="Optional 16 kHz mono WAV for STT")
    args = parser.parse_args()

    print("STT available:", stt.available)
    print("TTS available:", tts.available)

    if args.wav:
        if not args.wav.exists():
            print("WAV not found:", args.wav, file=sys.stderr)
            return 1
        pcm, rate = read_wav_pcm(args.wav)
        text = stt.transcribe(pcm, sample_rate=rate)
        print("TRANSCRIPT:", text or "(empty — is the Vosk Persian model installed?)")
        if text:
            db.init_db()
            sid = "wav-session"
            call_manager.end_call(sid)
            call_manager.start_call(sid)
            result = call_manager.handle_user_text(sid, text)
            print("REPLY:", result["reply"])
            out = Path("/tmp/persian_agent_reply.wav")
            tts.synthesize_to_file(result["reply"], filename=out.name)
            print("Wrote TTS to", out)

    run_dialogue()

    sample = tts.synthesize("نوبت شما ثبت شد.")
    print("TTS bytes:", len(sample))
    log.info("Local AI pipeline test finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
