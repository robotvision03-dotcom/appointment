"""Iranian passenger-car catalog and fuzzy matching for spoken names."""

from __future__ import annotations

from difflib import SequenceMatcher

from src.utils import normalize_persian

# (make, model, extra keywords)
CATALOG: list[tuple[str, str, str]] = [
    ("پژو", "پارس", "پارس ELX پارس سال peugeot pars پرشیا"),
    ("پژو", "۴۰۵", "405 جی ال ایکس GLX"),
    ("پژو", "۲۰۶", "206 تیپ"),
    ("پژو", "۲۰۶ صندوق‌دار", "206 SD صندوقدار"),
    ("پژو", "۲۰۷", "207 207i"),
    ("پژو", "۲۰۷ آی", "207i"),
    ("پژو", "۲۰۷ صندوق‌دار", "207 SD"),
    ("پژو", "۲۰۰۸", "2008"),
    ("پژو", "۳۰۱", "301"),
    ("پژو", "روآ", "ROA روآ"),
    ("پژو", "RD", "آر دی"),
    ("ایران‌خودرو", "سمند", "سمند LX معمولی"),
    ("ایران‌خودرو", "سمند سورن", "سورن پلاس"),
    ("ایران‌خودرو", "دنا", "دنا معمولی"),
    ("ایران‌خودرو", "دنا پلاس", "دناپلاس توربو"),
    ("ایران‌خودرو", "تارا", "تارا V4 اتومات"),
    ("ایران‌خودرو", "رانا", "رانا پلاس"),
    ("ایران‌خودرو", "پیکان", "پيکان"),
    ("سایپا", "پراید", "پراید ۱۳۱ ۱۳۲ ۱۱۱"),
    ("سایپا", "تیبا", "تیبا ۲"),
    ("سایپا", "تیبا ۲", "تیبا2"),
    ("سایپا", "ساینا", "ساینا S"),
    ("سایپا", "کوییک", "کوییک آر"),
    ("سایپا", "شاهین", "شاهین پلاس"),
    ("سایپا", "اطلس", "اطلس سایپا"),
    ("سایپا", "نیسان آبی", "وانت نیسان"),
    ("رنو", "تندر ۹۰", "ال۹۰ ال 90 L90 پارس تندر"),
    ("رنو", "ساندرو", "ساندرو استپ‌وی"),
    ("رنو", "مگان", "مگان"),
    ("رنو", "کپچر", "کپچر"),
    ("رنو", "تالیسمان", "تالیسمان"),
    ("کیا", "سراتو", "سراتو سایپا"),
    ("کیا", "اسپورتیج", "sportage"),
    ("کیا", "اپتیما", "optima"),
    ("کیا", "ریو", "rio"),
    ("هیوندای", "آوانته", "آوانته"),
    ("هیوندای", "i20", "آی ۲۰"),
    ("هیوندای", "توسان", "tucson"),
    ("هیوندای", "سانتافه", "santafe"),
    ("هیوندای", "سوناتا", "sonata"),
    ("هیوندای", "النترا", "elantra"),
    ("مزدا", "مزدا ۳", "mazda 3"),
    ("مزدا", "مزدا ۲", "mazda 2"),
    ("تویوتا", "کمری", "camry"),
    ("تویوتا", "کرولا", "corolla"),
    ("تویوتا", "یاریس", "yaris"),
    ("تویوتا", "پرادو", "prado"),
    ("تویوتا", "راو۴", "rav4"),
    ("نیسان", "ماکسیما", "maxima"),
    ("نیسان", "جوک", "juke"),
    ("نیسان", "ایکس‌تریل", "xtrail"),
    ("میتسوبیشی", "اوتلندر", "outlander"),
    ("میتسوبیشی", "لنسر", "lancer"),
    ("چری", "تیگو ۵", "tiggo"),
    ("چری", "تیگو ۷", "tiggo 7"),
    ("چری", "آریزو ۵", "arrizo"),
    ("ام‌وی‌ام", "X22", "ایکس ۲۲"),
    ("ام‌وی‌ام", "X33", "ایکس ۳۳"),
    ("ام‌وی‌ام", "۳۱۵", "315"),
    ("ام‌وی‌ام", "۱۱۰", "110"),
    ("جک", "S3", "اس ۳"),
    ("جک", "S5", "اس ۵"),
    ("جک", "J4", "جی ۴"),
    ("هایما", "S5", "هایما اس ۵"),
    ("هایما", "S7", "هایما اس ۷"),
    ("هایما", "۸S", "8s"),
    ("بهمن", "فیدلیتی", "fidelity"),
    ("بهمن", "دیگنیتی", "dignity"),
    ("لاماری", "ایما", "lamari"),
    ("کی‌ام‌سی", "T8", "kmc"),
    ("کی‌ام‌سی", "J7", "kmc j7"),
    ("بنز", "C200", "مرسدس c کلاس"),
    ("بنز", "E250", "e کلاس"),
    ("بی‌ام‌و", "۳۱۸", "bmw 318"),
    ("بی‌ام‌و", "۵۲۰", "520"),
    ("لکسوس", "NX", "nx"),
    ("لکسوس", "RX", "rx"),
    ("ولوو", "XC60", "xc60"),
    ("پورشه", "ماکان", "macan"),
    ("هوندا", "سیویک", "civic"),
    ("هوندا", "آکورد", "accord"),
    ("جیلی", "امگرند", "emgrand"),
    ("برلیانس", "H330", "h330"),
    ("لیفان", "X60", "x60"),
    ("فاو", " بسترن", "b50"),
    ("زامیاد", "پادرا", "پادرا"),
    ("سوزوکی", "ویتارا", "vitara"),
    ("مینی", "کانتری‌من", "countryman"),
    ("آئودی", "A6", "a6"),
    ("فولکس", "گل", "golf"),
    ("سیتروئن", "زانتیا", "xantia"),
    ("داچیا", "داستر", "duster"),
    ("ام‌جی", "6", "mg"),
    ("هاوال", "H6", "h6"),
    ("جنتو", "جنتو", "jetour"),
    ("اکستریم", "VX", "exeed"),
    ("ریگان", "کوپا", "coupe"),
    ("دایون", "Y5", "dayun"),
    ("کاپرا", "۲", "capra"),
]


def _fold(text: str) -> str:
    t = normalize_persian(text).replace("‌", "").replace(" ", "").replace("-", "")
    t = t.replace("آ", "ا")
    return t.lower()


def match_car(text: str) -> dict | None:
    """Return {make, model} when the utterance names a catalog car."""
    raw = normalize_persian(text)
    if not raw:
        return None
    folded = _fold(raw)
    scored: list[tuple[float, str, str]] = []
    for make, model, extra in CATALOG:
        blob = _fold(f"{make} {model} {extra}")
        hit = 0.0
        if _fold(model) and _fold(model) in folded:
            hit = 1.0 + len(_fold(model)) / 40
        elif _fold(make) in folded:
            hit = 0.55
        else:
            ratio = SequenceMatcher(None, folded, blob[: max(len(folded), 4)]).ratio()
            if ratio > 0.84:
                hit = ratio
        if hit:
            scored.append((hit, make, model))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_hit, make, model = scored[0]
    # If they only said the make, keep model empty so the agent asks for it.
    said_model = _fold(model) in folded
    if best_hit < 0.55:
        return None
    return {"make": make, "model": model if said_model or best_hit >= 1.0 else ""}


def list_makes() -> list[str]:
    seen: list[str] = []
    for make, _model, _extra in CATALOG:
        if make not in seen:
            seen.append(make)
    return seen


def models_for(make: str) -> list[str]:
    return [model for m, model, _ in CATALOG if m == make]


def parse_year(text: str) -> str | None:
    t = normalize_persian(text)
    import re

    nums = [int(x) for x in re.findall(r"\d{2,4}", t)]
    for n in nums:
        if 1370 <= n <= 1410:
            return str(n)
        if 1990 <= n <= 2027:
            return str(n)
        if 70 <= n <= 99:
            return str(1300 + n)
        if 0 <= n <= 20:
            return str(1400 + n)
    return None


_KM_WORDS = {
    "ده": 10, "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50,
    "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
    "صد": 100, "دویست": 200,
}


def parse_km(text: str) -> int | None:
    t = normalize_persian(text)
    import re

    compact = t.replace(",", "").replace("٬", "").replace("،", "")
    m = re.search(r"(\d+(?:\.\d+)?)", compact)
    if m:
        value = float(m.group(1))
    else:
        value = None
        for word, num in _KM_WORDS.items():
            if word in compact:
                value = float(num)
                break
        if value is None:
            return None
    if any(w in compact for w in ("میلیون", "ملیون")):
        value *= 1_000_000
    elif "هزار" in compact:
        value *= 1_000
    km = int(value)
    if km < 0 or km > 2_000_000:
        return None
    return km
