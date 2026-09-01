"""Connect a customer to a local service provider (call + SMS fallback)."""

from __future__ import annotations

from typing import Any

from src import db
from src.sms import make_tts_call, send_sms
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
    Call the seller so the customer can talk and book an appointment.

    If the seller does not answer (or the call cannot be placed), SMS the
    seller's number to the customer, and the customer's number to the seller.
    """
    provider = db.get_provider(provider_id)
    if not provider:
        return {"ok": False, "error": "سرویس‌دهنده پیدا نشد."}

    cust = _display_phone(customer_phone) or customer_phone
    seller = _display_phone(provider["phone"]) or provider["phone"]
    who = customer_name.strip() or "یک مشتری"
    service = provider.get("service_name") or "سرویس"

    tts_text = (
        f"سلام. {who} برای قرار ملاقات {service} با شما تماس می‌گیرد. "
        f"شماره مشتری {_spoken_digits(cust) or cust} است. لطفاً گوشی را بردارید."
    )
    sms_to_seller = (
        f"{who} برای قرار ملاقات «{service}» با شما کار دارد. "
        f"شماره مشتری: {cust}"
    )
    sms_to_customer = (
        f"{provider['name']} جواب نداد. برای صحبت و قرار ملاقات با ایشان تماس بگیرید: {seller}"
    )

    call_result = make_tts_call(provider["phone"], tts_text)
    no_answer = not call_result.get("ok")

    seller_sms = send_sms(provider["phone"], sms_to_seller)
    customer_sms = {"ok": False, "reason": "not_needed"}
    if no_answer:
        customer_sms = send_sms(cust, sms_to_customer)

    call_status = "placed" if call_result.get("ok") else (
        "dry_run" if call_result.get("dry_run") else f"failed:{call_result.get('reason', 'unknown')}"
    )
    sms_status = (
        f"seller:{_sms_label(seller_sms)};customer:{_sms_label(customer_sms)}"
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
    provider_tel = "".join(c for c in seller if c.isdigit() or c == "+")
    customer_tel = "".join(c for c in cust if c.isdigit() or c == "+")
    return {
        "ok": True,
        "request_id": req_id,
        "provider": provider,
        "customer_phone": cust,
        "provider_phone": seller,
        "call": call_result,
        "sms": seller_sms,
        "sms_to_seller": seller_sms,
        "sms_to_customer": customer_sms,
        "sms_on_no_answer": no_answer,
        "call_attempted": not call_result.get("dry_run"),
        "tel_provider": f"tel:{provider_tel}",
        "tel_customer": f"tel:{customer_tel}" if customer_tel else "",
        "fallback_sms": no_answer,
    }


def _sms_label(result: dict[str, Any]) -> str:
    if result.get("ok"):
        return "sent"
    if result.get("dry_run"):
        return "dry_run"
    return str(result.get("reason") or "skipped")


def format_provider_list(service_id: int) -> str:
    providers = db.list_providers(service_id)
    if not providers:
        return "برای این سرویس هنوز ارائه‌دهنده‌ای ثبت نشده است."
    parts = []
    for i, p in enumerate(providers, start=1):
        area = f"، {p['area']}" if p.get("area") else ""
        parts.append(f"{i}) {p['name']}{area} — {p['phone']}")
    return "؛ ".join(parts)
