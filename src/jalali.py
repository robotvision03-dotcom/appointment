"""Gregorian ↔ Jalali (Shamsi) conversion for the office calendar."""

from __future__ import annotations

from datetime import date, timedelta

_G_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_J_DAYS = (31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29)
WEEKDAYS_FA = ("دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه")
MONTHS_FA = (
    "فروردین",
    "اردیبهشت",
    "خرداد",
    "تیر",
    "مرداد",
    "شهریور",
    "مهر",
    "آبان",
    "آذر",
    "دی",
    "بهمن",
    "اسفند",
)


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    gy2 = gy - 1600
    gm2 = gm - 1
    gd2 = gd - 1
    g_day_no = 365 * gy2 + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
    for i in range(gm2):
        g_day_no += _G_DAYS[i]
    if gm > 2 and ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        g_day_no += 1
    g_day_no += gd2
    j_day_no = g_day_no - 79
    j_np = j_day_no // 12053
    j_day_no %= 12053
    jy = 979 + 33 * j_np + 4 * (j_day_no // 1461)
    j_day_no %= 1461
    if j_day_no >= 366:
        jy += (j_day_no - 1) // 365
        j_day_no = (j_day_no - 1) % 365
    jm = 0
    for i in range(11):
        if j_day_no < _J_DAYS[i]:
            jm = i + 1
            break
        j_day_no -= _J_DAYS[i]
    else:
        jm = 12
    jd = j_day_no + 1
    return jy, jm, jd


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    jy2 = jy - 979
    jm2 = jm - 1
    jd2 = jd - 1
    j_day_no = 365 * jy2 + (jy2 // 33) * 8 + ((jy2 % 33) + 3) // 4
    for i in range(jm2):
        j_day_no += _J_DAYS[i]
    j_day_no += jd2
    g_day_no = j_day_no + 79
    gy = 1600 + 400 * (g_day_no // 146097)
    g_day_no %= 146097
    leap = True
    if g_day_no >= 36525:
        g_day_no -= 1
        gy += 100 * (g_day_no // 36524)
        g_day_no %= 36524
        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = False
    gy += 4 * (g_day_no // 1461)
    g_day_no %= 1461
    if g_day_no >= 366:
        leap = False
        g_day_no -= 1
        gy += g_day_no // 365
        g_day_no %= 365
    gd = g_day_no + 1
    g_days = list(_G_DAYS)
    if leap:
        g_days[1] = 29
    gm = 1
    for dim in g_days:
        if gd <= dim:
            break
        gd -= dim
        gm += 1
    return gy, gm, gd


def to_jalali(d: date) -> tuple[int, int, int]:
    return gregorian_to_jalali(d.year, d.month, d.day)


def from_jalali(jy: int, jm: int, jd: int) -> date:
    gy, gm, gd = jalali_to_gregorian(jy, jm, jd)
    return date(gy, gm, gd)


def format_jalali(d: date) -> str:
    jy, jm, jd = to_jalali(d)
    return f"{WEEKDAYS_FA[d.weekday()]} {jd} {MONTHS_FA[jm - 1]} {jy}"


def jalali_month_matrix(jy: int, jm: int) -> list[list[date | None]]:
    """Weeks starting Saturday, like a typical Iranian calendar."""
    first = from_jalali(jy, jm, 1)
    # number of days in month
    if jm <= 6:
        dim = 31
    elif jm <= 11:
        dim = 30
    else:
        try:
            from_jalali(jy, 12, 30)
            dim = 30
        except ValueError:
            dim = 29
    # Saturday = 5 in Python weekday(). Offset from Saturday:
    offset = (first.weekday() - 5) % 7
    cells: list[date | None] = [None] * offset
    for day in range(1, dim + 1):
        cells.append(from_jalali(jy, jm, day))
    while len(cells) % 7:
        cells.append(None)
    return [cells[i : i + 7] for i in range(0, len(cells), 7)]


def iter_days(start: date, count: int):
    for i in range(count):
        yield start + timedelta(days=i)
