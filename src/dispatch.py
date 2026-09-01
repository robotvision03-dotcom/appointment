"""Connect a customer to a local service provider (call + SMS fallback)."""

from __future__ import annotations

from typing import Any

from src import db
from src.sms import make_tts_call, normalize_iran_mobile, send_sms
from src.utils import log


def _spoken_digits(phone: str) -> str:
    mapping = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    digits = "".join(c for c in phone if c.isdigit())
    if digits.startswith("98") and len(digits) == 12:
        digits = "0" + digits[2:]
    return " ".join(digits.translate(mapping))


def _display_phone(phone: str) -> str:
    digits = "".join(c for c in (phone or "") if c.isdigit())
    if digits.startswith("98") and len(digits) >= 12:
        return "0" + digits[2:]
    return phone or ""


def connect_customer_to_provider(
    customer_phone: str,
    provider_id: int,
    customer_name: str = "",
) -> dict[str, Any]:
    """
    Call the provider right away. If the call cannot be placed (or we cannot
    know they answered), send an SMS that includes the customer's number.
    """
    provider = db.get_provider(provider_id)
    if not provider:
        return {"ok": False, "error": "سرویس‌دهنده پیدا نشد."}

    cust = _display_phone(customer_phone) or customer_phone
    who = customer_name.strip() or "یک مشتری"
    service = provider.get("service_name") or "سرویس"
    sms_text = (
        f"{who} برای «{service}» به شما نیاز دارد. "
        f"شماره تماس مشتری: {cust}"
    )
    tts_text = (
        f"سلام. {who} برای {service} به شما نیاز دارد. "
        f"شماره ایشان {_spoken_digits(cust) or cust} است. لطفاً تماس بگیرید."
    )

    call_result = make_tts_call(provider["phone"], tts_text)
    sms_needed = not call_result.get("ok") or call_result.get("reason") in {
        "kavenegar_not_configured",
        "emergency_number",
        "invalid_number",
    }
    # Always leave an SMS trail so a missed TTS call still reaches the seller.
    sms_result = send_sms(provider["phone"], sms_text)
    if sms_result.get("reason") == "invalid_number":
        sms_needed = True

    call_status = "placed" if call_result.get("ok") else (
        "dry_run" if call_result.get("dry_run") else f"failed:{call_result.get('reason', 'unknown')}"
    )
    sms_status = "sent" if sms_result.get("ok") else (
        "dry_run" if sms_result.get("dry_run") else f"failed:{sms_result.get('reason', 'unknown')}"
    )
    req_id = db.save_service_request(
        customer_name=customer_name,
        customer_phone=cust,
        provider_id=int(provider["id"]),
        call_status=call_status,
        sms_status=sms_status,
    )
    log.info(
        "Dispatch request=%s provider=%s call=%s sms=%s",
        req_id,
        provider["name"],
        call_status,
        sms_status,
    )
    provider_tel = "".join(c for c in provider["phone"] if c.isdigit() or c == "+")
    customer_tel = "".join(c for c in cust if c.isdigit() or c == "+")
    return {
        "ok": True,
        "request_id": req_id,
        "provider": provider,
        "customer_phone": cust,
        "call": call_result,
        "sms": sms_result,
        "sms_on_no_answer": True,
        "call_attempted": not call_result.get("dry_run"),
        "tel_provider": f"tel:{provider_tel}",
        "tel_customer": f"tel:{customer_tel}" if customer_tel else "",
        "fallback_sms": sms_needed or not call_result.get("ok"),
    }


def format_provider_list(service_id: int) -> str:
    providers = db.list_providers(service_id)
    if not providers:
        return "برای این سرویس هنوز ارائه‌دهنده‌ای ثبت نشده است."
    parts = []
    for i, p in enumerate(providers, start=1):
        area = f" — {p['area']}" if p.get("area") else ""
        parts.append(f"{i}) {p['name']}{area}")
    return "، ".join(parts)
