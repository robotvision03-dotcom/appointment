"""Fast lexicon for Iranian car names, ASR typos, and short fragments.

Example: سمن → سمند, پرس → پارس (one-letter slip / missing alef).
"""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher

from src.cars import CATALOG
from src.utils import normalize_persian

# Heard fragment → canonical spoken form (then matched against the catalog).
ASR_ALIASES = {
    "سمن": "سمند",
    "سمندد": "سمند",
    "سورن": "سمند سورن",
    "پرس": "پارس",
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
}


def fold(text: str) -> str:
    t = normalize_persian(text)
    t = t.replace("‌", "").replace(" ", "").replace("-", "").replace("ـ", "")
    t = t.replace("آ", "ا").replace("أ", "ا").replace("ة", "ه")
    return t.lower()


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
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
                consider(1.0, make, model, kind == "model" or fold(model) == f)
            elif len(folded) >= 3 and (folded in f or f in folded):
                consider(0.93 if f.startswith(folded) or folded.startswith(f) else 0.86, make, model, True)

        # Prefix index (سمن → سمند)
        for idx in self.prefix.get(folded[: min(len(folded), 8)], []):
            f, make, model, kind = self.entries[idx]
            if f.startswith(folded) and len(folded) >= 3:
                consider(0.9 + min(len(folded), 6) / 80, make, model, True)

        # Edit distance 1 (پرس → پارس)
        if 3 <= len(folded) <= 12:
            for f, make, model, kind in self.entries:
                if abs(len(f) - len(folded)) > 1:
                    continue
                d = _lev(folded, f)
                if d == 1:
                    consider(0.84, make, model, kind != "make" or len(f) <= 4)
                elif d == 0:
                    consider(1.0, make, model, True)

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
        scored.sort(key=lambda x: (x[0], -len(x[2]), 1 if x[3] else 0), reverse=True)
        best = scored[0]
        if best[0] < 0.82:
            return None
        make, model = best[1], best[2]
        said_model = best[3] or fold(model).startswith(fold(text)) or fold(text) in fold(model)
        if fold(text) in self.aliases:
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


def resolve_car(text: str) -> dict | None:
    hit = car_dictionary().lookup(text)
    if hit:
        return hit
    from src.cars import match_car as _legacy

    return _legacy(text)
