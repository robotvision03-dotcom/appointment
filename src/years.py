"""Shamsi model years 1370–1410 spoken as words, digits, or loose fragments.

Sellers say a model year in many shapes:

  هزار و سیصد و هشتاد و هشت · یک هزار و سیصد و هشتاد و هشت
  سیصد و هشتاد و هشت · هشتاد و هشت · هشت و هشت · یک سه نه نه
  ۱۳۸۸ · ۳۸۸ · ۸۸

Only 41 years are possible, so every spoken form is generated once into a
lookup table. Anything the table misses falls back to a word-sum parser.
"""

from __future__ import annotations

import re

from src.utils import normalize_persian

MIN_YEAR = 1370
MAX_YEAR = 1410

UNITS: dict[int, str] = {
    0: "صفر",
    1: "یک",
    2: "دو",
    3: "سه",
    4: "چهار",
    5: "پنج",
    6: "شش",
    7: "هفت",
    8: "هشت",
    9: "نه",
}

TEENS: dict[int, str] = {
    10: "ده",
    11: "یازده",
    12: "دوازده",
    13: "سیزده",
    14: "چهارده",
    15: "پانزده",
    16: "شانزده",
    17: "هفده",
    18: "هجده",
    19: "نوزده",
}

TENS: dict[int, str] = {
    20: "بیست",
    30: "سی",
    40: "چهل",
    50: "پنجاه",
    60: "شصت",
    70: "هفتاد",
    80: "هشتاد",
    90: "نود",
}

HUNDREDS: dict[int, str] = {
    100: "صد",
    200: "دویست",
    300: "سیصد",
    400: "چهارصد",
}

# ASR slips and regional spellings, mapped onto the canonical number word.
TOKEN_ALIASES: dict[str, str] = {
    "نو": "نه",
    "نوه": "نه",
    "شیش": "شش",
    "شیشه": "شش",
    "چار": "چهار",
    "پونصد": "پانصد",
    "یکصد": "صد",
    "سد": "صد",
    "سیسد": "سیصد",
    "چارصد": "چهارصد",
    "هیجده": "هجده",
    "هژده": "هجده",
    "پانزده": "پانزده",
    "پونزده": "پانزده",
    "شونزده": "شانزده",
    "هفده": "هفده",
    "هیفده": "هفده",
    "نودو": "نود",
    "هشتادو": "هشتاد",
    "هفتادو": "هفتاد",
}

_WORD_VALUES: dict[str, int] = {}
for _table in (UNITS, TEENS, TENS, HUNDREDS):
    for _value, _word in _table.items():
        _WORD_VALUES[_word] = _value
_WORD_VALUES["پانصد"] = 500


def _tokens(text: str) -> list[str]:
    t = normalize_persian(text).replace("\u200c", " ")
    t = re.sub(r"[^\w\u0600-\u06FF]+", " ", t)
    out: list[str] = []
    for raw in t.split():
        tok = TOKEN_ALIASES.get(raw, raw)
        if tok == "و":
            continue
        out.append(tok)
    return out


def _key(tokens: list[str]) -> str:
    return "".join(tokens)


def _tens_units_words(n: int) -> list[str]:
    """Spoken words for 0–99, e.g. 88 → [هشتاد, هشت]."""
    if n < 10:
        return [UNITS[n]]
    if n in TEENS:
        return [TEENS[n]]
    tens, unit = divmod(n, 10)
    words = [TENS[tens * 10]]
    if unit:
        words.append(UNITS[unit])
    return words


def _build_table() -> dict[str, int]:
    table: dict[str, int] = {}
    ambiguous: set[str] = set()

    def put(tokens: list[str], year: int) -> None:
        key = _key(tokens)
        if not key:
            return
        if key in table and table[key] != year:
            ambiguous.add(key)
            return
        table[key] = year

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        thousand, rest = divmod(year, 1000)
        hundred, tail = divmod(rest, 100)
        head = HUNDREDS[hundred * 100]
        tail_words = _tens_units_words(tail) if tail else []
        digits = [UNITS[int(d)] for d in str(year)]

        put(["هزار", head, *tail_words], year)
        put([UNITS[thousand], "هزار", head, *tail_words], year)
        put([head, *tail_words], year)
        put(digits, year)
        put(digits[1:], year)
        if tail >= 10:
            put(tail_words, year)
            put([UNITS[tail // 10], UNITS[tail % 10]] if tail % 10 else [], year)
        put([str(year)], year)
        put([str(rest)], year)
        put([f"{tail:02d}"], year)

    for key in ambiguous:
        table.pop(key, None)
    return table


YEAR_TABLE: dict[str, int] = _build_table()


def _sum_words(tokens: list[str]) -> int | None:
    """Classic word-sum: هزار و سیصد و هشتاد و هشت → 1388."""
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok == "هزار":
            current = current or 1
            total += current * 1000
            current = 0
            seen = True
            continue
        value = _WORD_VALUES.get(tok)
        if value is None:
            return None
        current += value
        seen = True
    if not seen:
        return None
    return total + current


def _spoken_digits(tokens: list[str]) -> int | None:
    """یک سه نه نه → 1399 (each token one digit)."""
    digits = ""
    for tok in tokens:
        value = _WORD_VALUES.get(tok)
        if value is None or value > 9:
            return None
        digits += str(value)
    if not digits:
        return None
    return int(digits)


def _expand(n: int) -> int | None:
    """Turn 88 / 388 / 1388 into a plausible Shamsi year."""
    if MIN_YEAR <= n <= MAX_YEAR:
        return n
    if 370 <= n <= 410:
        return 1000 + n
    if 70 <= n <= 99:
        return 1300 + n
    if 0 <= n <= MAX_YEAR - 1400:
        return 1400 + n
    return None


def parse_shamsi_year(text: str) -> int | None:
    """Best Shamsi year in 1370–1410, or None when the phrase is not a year."""
    # A trailing «و» means the speaker was still listing a number
    # («یک هزار و سیصد و هشتاد و» is not 1380).
    raw = normalize_persian(text).replace("\u200c", " ")
    raw_parts = re.sub(r"[^\w\u0600-\u06FF]+", " ", raw).split()
    if raw_parts and raw_parts[-1] == "و":
        return None

    tokens = _tokens(text)
    if not tokens:
        return None

    hit = YEAR_TABLE.get(_key(tokens))
    if hit is not None:
        return hit

    for candidate in (_sum_words(tokens), _spoken_digits(tokens)):
        if candidate is None:
            continue
        year = _expand(candidate)
        if year is not None:
            return year

    # Digits mixed into the sentence: «مدل ۸۸ هست»
    for group in re.findall(r"\d+", " ".join(tokens)):
        year = _expand(int(group))
        if year is not None:
            return year

    # Word fragments inside a longer sentence: «ماشین هشتاد و هشت است»
    window: list[str] = []
    for tok in tokens:
        if tok == "هزار" or tok in _WORD_VALUES:
            window.append(tok)
            continue
        if window:
            hit = YEAR_TABLE.get(_key(window))
            if hit is not None:
                return hit
        window = []
    if window:
        hit = YEAR_TABLE.get(_key(window))
        if hit is not None:
            return hit
        for candidate in (_sum_words(window), _spoken_digits(window)):
            if candidate is None:
                continue
            year = _expand(candidate)
            if year is not None:
                return year
    return None


def year_words(year: int) -> str:
    """Canonical spoken form, used in prompts and confirmations."""
    if not MIN_YEAR <= year <= MAX_YEAR:
        return str(year)
    _thousand, rest = divmod(year, 1000)
    hundred, tail = divmod(rest, 100)
    parts = ["هزار", HUNDREDS[hundred * 100]]
    if tail:
        parts.extend(_tens_units_words(tail))
    return " و ".join(parts)
