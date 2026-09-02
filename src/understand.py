"""Map noisy ASR onto cars / km / year using lexicon then a constrained LLM."""

from __future__ import annotations

from typing import Any

from src.cars import parse_km, parse_year
from src.lexicon import resolve_car
from src.utils import log, normalize_persian


def understand(text: str, phase: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return canonical fields plus a display string (heard → corrected)."""
    heard = normalize_persian(text)
    out: dict[str, Any] = {"heard": heard, "text": heard, "source": "raw"}
    if not heard:
        return out

    if phase in {"ask_type", "ask_model"}:
        car = resolve_car(heard)
        if car and (car.get("model") or car.get("make")):
            out["car"] = car
            out["source"] = "lexicon"
            label = f"{car.get('make', '')} {car.get('model', '')}".strip()
            out["text"] = label
            out["corrected"] = label != heard
            return out
        car = _llm_map_car(heard, info or {})
        if car:
            out["car"] = car
            out["source"] = "llm"
            label = f"{car.get('make', '')} {car.get('model', '')}".strip()
            out["text"] = label or heard
            out["corrected"] = bool(label) and label != heard
            return out
        return out

    if phase == "ask_year":
        year = parse_year(heard)
        if year:
            out["year"] = year
            out["text"] = year
            out["source"] = "parser"
        return out

    if phase == "ask_km":
        km = parse_km(heard)
        if km is not None:
            out["km"] = km
            out["text"] = str(km)
            out["source"] = "parser"
        return out

    return out


def _llm_map_car(heard: str, info: dict[str, Any]) -> dict | None:
    from src.llm import llm

    if not llm.is_available():
        return None
    from src.cars import CATALOG

    names = "\n".join(f"- {make} {model}" for make, model, _ in CATALOG[:80])
    prompt = (
        "تو تصحیح‌کننده گفتار برای دفتر کارشناسی خودرو در ایران هستی.\n"
        "فقط از فهرست خودروهای زیر انتخاب کن. اگر گفته ناقص است نزدیک‌ترین نام یکتا را بده.\n"
        "مثال: سمن → سمند ؛ پرس → پژو پارس.\n"
        f"فهرست:\n{names}\n"
        f"گفته مشتری: {heard}\n"
        f"قبلاً ثبت شده: {info}\n"
        'فقط JSON: {"make":"...","model":"..."} یا {"make":null,"model":null} اگر هیچ خودرویی نیست.'
    )
    raw = llm.generate_response(prompt)
    from src.utils import extract_json_object

    parsed = extract_json_object(raw) if raw else None
    if not parsed:
        return None
    make = (parsed.get("make") or "").strip()
    model = (parsed.get("model") or "").strip()
    if not make and not model:
        return None
    hit = resolve_car(f"{make} {model}".strip())
    if hit:
        log.info("LLM car map %r -> %s %s", heard, hit.get("make"), hit.get("model"))
        return hit
    if make:
        return {"make": make, "model": model, "score": 0.7, "heard": heard}
    return None
