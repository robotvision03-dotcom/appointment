"""Iranian SMS via Kavenegar (works inside Iran). Optional Melipayamak-style HTTP fallback is not required."""

from __future__ import annotations

import re

import httpx

from src.config import config
from src.utils import log


def normalize_iran_mobile(raw: str) -> str | None:
    """Return 98912… form, or None if it does not look like an Iranian mobile."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("0098"):
        digits = digits[4:]
    if digits.startswith("98"):
        digits = digits[2:]
    if digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10 and digits.startswith("9"):
        return "98" + digits
    return None


def send_sms(receptor: str, message: str) -> dict:
    """
    Send an SMS through Kavenegar.

    Docs: https://kavenegar.com/rest.html
    Endpoint is hosted in Iran and does not depend on Twilio.
    """
    phone = normalize_iran_mobile(receptor)
    if not phone:
        log.warning("SMS skipped — invalid Iranian mobile: %s", receptor)
        return {"ok": False, "reason": "invalid_number"}
    if not config.kavenegar_api_key:
        log.info("SMS dry-run to %s: %s", phone, message)
        return {"ok": False, "reason": "kavenegar_not_configured", "dry_run": True, "to": phone, "message": message}

    url = f"https://api.kavenegar.com/v1/{config.kavenegar_api_key}/sms/send.json"
    params = {"receptor": phone, "message": message}
    if config.kavenegar_sender:
        params["sender"] = config.kavenegar_sender
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)
            data = resp.json()
        ok = resp.status_code == 200 and (data.get("return") or {}).get("status") == 200
        log.info("Kavenegar SMS to %s ok=%s", phone, ok)
        return {"ok": ok, "provider": "kavenegar", "to": phone, "response": data}
    except Exception as exc:  # noqa: BLE001
        log.error("Kavenegar SMS failed: %s", exc)
        return {"ok": False, "reason": str(exc), "to": phone}


def send_booking_sms(call_sid: str) -> dict | None:
    from src.call_manager import call_manager

    body = call_manager.sms_body(call_sid)
    state = call_manager.get(call_sid)
    if not body or not state:
        return None
    phone = state.from_number or state.patient_info.get("phone") or ""
    if not phone:
        log.info("SMS skipped — no patient phone on call %s", call_sid)
        return {"ok": False, "reason": "no_phone"}
    return send_sms(phone, body)
