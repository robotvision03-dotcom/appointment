"""HTTP turn API for an Iranian SIP/Asterisk PBX (local trunk, no Twilio)."""

from __future__ import annotations

from typing import Any

from src.call_manager import call_manager
from src.handoff import start_warm_transfer
from src.sms import send_booking_sms
from src.utils import log


def sip_turn(session_id: str, text: str, phone: str = "") -> dict[str, Any]:
    """
    One dialogue turn for Asterisk AGI / dialplan.

    Typical Iran setup: SIP trunk from MCI/Shatel/Respina into Asterisk,
    SpeechToText via local Shenava-Koochik-v1.5, then POST here. Replies are text.
    """
    if call_manager.get(session_id) is None:
        call_manager.start_call(session_id, from_number=phone)
        if not text:
            reply = call_manager.greeting()
            return {"reply": reply, "phase": "ask_service", "intent": "continue", "call_sid": session_id}
    result = call_manager.handle_user_text(session_id, text)
    if result.get("intent") == "book":
        send_booking_sms(session_id)
    if result.get("intent") == "transfer":
        result["transfer"] = start_warm_transfer(session_id)
        extra = (result["transfer"] or {}).get("reply_extra")
        if extra:
            result["reply"] = f"{result.get('reply', '')} {extra}".strip()
    log.info("SIP turn %s phase=%s", session_id, result.get("phase"))
    return result


def sip_greeting_wav_path(session_id: str, phone: str = "") -> bytes:
    if call_manager.get(session_id) is None:
        call_manager.start_call(session_id, from_number=phone)
    return b""
