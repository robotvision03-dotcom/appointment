"""Human handoff that works in Iran: click-to-call the receptionist (no Twilio)."""

from __future__ import annotations

from typing import Any

from src.call_manager import call_manager
from src.config import config
from src.utils import log


def start_warm_transfer(call_sid: str) -> dict[str, Any]:
    """
    Twilio conference bridging is not used: Twilio is blocked in Iran.

    Instead we return a tel: link and a spoken summary the patient/agent can
    read to the human receptionist after dialing RECEPTIONIST_NUMBER.
    """
    summary = call_manager.transfer_summary(call_sid)
    number = config.receptionist_number
    log.info("Handoff to receptionist %s for %s: %s", number, call_sid, summary)
    return {
        "ok": True,
        "method": "click_to_call",
        "receptionist_number": number,
        "tel_url": f"tel:{number}",
        "summary": summary,
        "reply_extra": f"شماره منشی: {number}. می‌توانید همین حالا تماس بگیرید.",
    }
