"""Fast lexicon for Iranian car names, ASR typos, and short fragments.

Example: سمن → سمند, پرس → پارس (one-letter slip / missing alef).
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from src.cars import CATALOG
from src.utils import normalize_persian

# Letters Shenava/Whisper often confuse in Persian (homophones + similar glyphs).
_LETTER_PAIRS = (
    ("ژ", "ج"),
    ("ق", "غ"),
    ("ث", "س"),
    ("ص", "س"),
    ("ط", "ت"),
    ("ذ", "ز"),
    ("ض", "ز"),
    ("ظ", "ز"),
    ("ك", "ک"),
    ("ة", "ه"),
    ("أ", "ا"),
    ("إ", "ا"),
    ("ؤ", "و"),
)


def letter_swap_variants(folded: str) -> set[str]:
    """One-substitution neighbors used as extra ASR spellings of catalog names."""
    if len(folded) < 3:
        return set()
    out: set[str] = set()
    for a, b in _LETTER_PAIRS:
        if a in folded:
            out.add(folded.replace(a, b, 1))
        if b in folded and a != b:
            out.add(folded.replace(b, a, 1))
    out.discard(folded)
    return out


# Heard fragment → canonical spoken form (then matched against the catalog).
ASR_ALIASES = {
    "سمن": "سمند",
    "سمندد": "سمند",
    "سورن": "سمند سورن",
    "پرس": "پارس",
    "پانس": "پارس",
    "پانص": "پارس",
    "پارص": "پارس",
    "ژو": "پژو",
    "پارسس": "پارس",
    "پرشیا": "پارس",
    "پرش": "پارس",
    "پرا": "پراید",
    "پرایدد": "پراید",
    "پژ": "پژو",
    "پجو": "پژو",
    "پجوو": "پژو",
    "دناپ": "دنا پلاس",
    "دناپلاس": "دنا پلاس",
    "شاهی": "شاهین",
    "شاهینن": "شاهین",
    "تیب": "تیبا",
    "ساین": "ساینا",
    "کویک": "کوییک",
    "کوییکی": "کوییک",
    "تاراا": "تارا",
    "راناا": "رانا",
    "ال۹۰": "تندر ۹۰",
    "ال90": "تندر ۹۰",
    "ال نود": "تندر ۹۰",
    "ال‌نود": "تندر ۹۰",
    "تندر": "تندر ۹۰",
    "پیکان": "پیکان",
    "پیکن": "پیکان",
    "سرات": "سراتو",
    "توسن": "توسان",
    "سانتفه": "سانتافه",
    "کمری": "کمری",
    "کرلا": "کرولا",
    "هایما": "هایما",
    "فیدلیتی": "فیدلیتی",
    "دیگنیتی": "دیگنیتی",
    "پژوپارس": "پژو پارس",
    "پجو پارس": "پژو پارس",
    "پجوپارس": "پژو پارس",
    "دو شش": "۲۰۶",
    "دویست و شش": "۲۰۶",
    "چهارصد پنج": "۴۰۵",
    "چهارصد و پنج": "۴۰۵",
    "تیگو": "تیگو ۵",
    "آریزو": "آریزو ۵",
    "کوییک ار": "کوییک",
    "دنا توربو": "دنا پلاس",
    "سورن پلاس": "سمند سورن",
    "وانت نیسان": "نیسان آبی",
    "نیسان ابی": "نیسان آبی",
}


def fold(text: str) -> str:
    t = normalize_persian(text)
    t = t.replace("‌", "").replace(" ", "").replace("-", "").replace("ـ", "")
    t = t.replace("آ", "ا").replace("أ", "ا").replace("ة", "ه")
    # Whisper punctuation and stray Latin letters inside Persian words.
    if any("\u0600" <= ch <= "\u06ff" for ch in t):
        t = "".join(ch for ch in t if not ("a" <= ch.lower() <= "z"))
    t = "".join(ch for ch in t if ch.isalnum() or "\u0600" <= ch <= "\u06ff")
    return t.lower()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 3:
        return 9
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


class CarDictionary:
    """Prefix + edit-distance index over the office catalog (built once)."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str, str, str]] = []  # fold, make, model, kind
        self.prefix: dict[str, list[int]] = defaultdict(list)
        for make, model, extra in CATALOG:
            tokens = [make, model, *extra.split()]
            for tok in tokens:
                f = fold(tok)
                if len(f) < 2:
                    continue
                kind = "model" if fold(model) == f or fold(model).startswith(f) else "make"
                idx = len(self.entries)
                self.entries.append((f, make, model, kind))
                for n in range(3, min(len(f), 8) + 1):
                    self.prefix[f[:n]].append(idx)
        self.aliases = {fold(k): fold(v) for k, v in ASR_ALIASES.items()}
        for make, model, extra in CATALOG:
            for tok in (make, model, *extra.split()):
                f = fold(tok)
                if len(f) < 4:
                    continue
                for variant in letter_swap_variants(f):
                    self.aliases.setdefault(variant, f)

    def lookup(self, text: str) -> dict | None:
        raw = normalize_persian(text)
        folded = fold(raw)
        if len(folded) < 2:
            return None
        if folded in self.aliases:
            folded = self.aliases[folded]
            raw = folded

        scored: list[tuple[float, str, str, bool]] = []
        best_map: dict[tuple[str, str], tuple[float, str, str, bool]] = {}

        def consider(score: float, make: str, model: str, is_model: bool) -> None:
            key = (make, model)
            prev = best_map.get(key)
            if prev is None or score > prev[0]:
                best_map[key] = (score, make, model, is_model)

        # Exact / contains
        for f, make, model, kind in self.entries:
            if f == folded:
                is_model = kind == "model" or fold(model) == f
                # A bare make («پژو») must not drag in an arbitrary model.
                consider(1.0 if is_model else 0.75, make, model, is_model)
            elif len(folded) >= 3 and (folded in f or f in folded):
                model_in = fold(model) and fold(model) in folded
                prefixish = f.startswith(folded) or folded.startswith(f)
                if model_in or kind == "model":
                    consider(0.95 if model_in else (0.93 if prefixish else 0.86), make, model, True)
                else:
                    # Make-only substring (پژو inside پژوپارس) must not pick «RD».
                    consider(0.72, make, model, False)

        # Prefix index (سمن → سمند)
        for idx in self.prefix.get(folded[: min(len(folded), 8)], []):
            f, make, model, kind = self.entries[idx]
            if f.startswith(folded) and len(folded) >= 3:
                is_model = kind == "model" or fold(model).startswith(folded)
                score = 0.9 + min(len(folded), 6) / 80 if is_model else 0.74
                consider(score, make, model, is_model)

        # Edit distance 1 (پرس → پارس), and 2 for longer slips (ژوپانس → پژوپارس)
        if 3 <= len(folded) <= 14:
            for f, make, model, kind in self.entries:
                if abs(len(f) - len(folded)) > 2:
                    continue
                d = _lev(folded, f)
                if d == 0:
                    is_model = kind == "model" or fold(model) == f
                    consider(1.0 if is_model else 0.75, make, model, is_model)
                elif d == 1:
                    consider(0.84, make, model, kind != "make" or len(f) <= 4)
                elif d == 2 and len(folded) >= 6 and len(f) >= 6:
                    consider(0.83, make, model, kind == "model")

        # SequenceMatcher fallback for slightly longer phrases
        if not best_map and len(folded) >= 4:
            for f, make, model, _kind in self.entries:
                if abs(len(f) - len(folded)) > 3:
                    continue
                r = SequenceMatcher(None, folded, f).ratio()
                if r >= 0.78:
                    consider(r * 0.9, make, model, True)

        scored = list(best_map.values())
        if not scored:
            return None
        scored.sort(key=lambda x: (x[0], 1 if x[3] else 0, len(x[2])), reverse=True)
        best = scored[0]
        if best[0] < 0.82:
            return None
        make, model = best[1], best[2]
        said_model = best[3] or fold(model).startswith(fold(text)) or fold(text) in fold(model)
        alias = self.aliases.get(fold(text))
        if alias and alias in fold(model):
            said_model = True
        return {
            "make": make,
            "model": model if said_model else "",
            "score": round(best[0], 3),
            "heard": text,
        }


_DICT: CarDictionary | None = None


def car_dictionary() -> CarDictionary:
    global _DICT
    if _DICT is None:
        _DICT = CarDictionary()
    return _DICT


def _resolve_by_token(text: str) -> dict | None:
    """Recover «ژو پانس» when each word alone still points at the catalog."""
    words = [w for w in normalize_persian(text).split() if len(fold(w)) >= 2]
    if len(words) < 2:
        return None
    dictionary = car_dictionary()
    make = ""
    model = ""
    model_make = ""
    for word in words:
        hit = dictionary.lookup(word)
        if not hit:
            continue
        if hit.get("model") and not model:
            model = hit["model"]
            model_make = hit.get("make", "")
        elif hit.get("make") and not make:
            make = hit["make"]
    if model:
        return {"make": model_make or make, "model": model, "score": 0.82, "heard": text}
    if make:
        return {"make": make, "model": "", "score": 0.72, "heard": text}
    return None


def resolve_car(text: str) -> dict | None:
    hit = car_dictionary().lookup(text)
    if hit and hit.get("model"):
        return hit
    by_token = _resolve_by_token(text)
    if by_token and by_token.get("model"):
        return by_token
    if hit:
        return hit
    if by_token:
        return by_token
    from src.cars import match_car as _legacy

    return _legacy(text)
