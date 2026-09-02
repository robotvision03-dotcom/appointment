"""SQLite: car catalog and 30-minute inspection appointments."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Iterator

from src.cars import CATALOG, match_car
from src.config import config
from src.jalali import format_jalali, jalali_month_matrix, to_jalali
from src.utils import log, normalize_persian

FRIDAY = 4  # datetime.weekday()


@contextmanager
def get_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db_path or config.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                keywords TEXT NOT NULL DEFAULT '',
                UNIQUE(make, model)
            );

            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                seller_name TEXT NOT NULL,
                phone TEXT,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                year TEXT,
                km INTEGER,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'booked'
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_insp_slot
                ON inspections(date, time) WHERE status = 'booked';
            """
        )
        count = conn.execute("SELECT COUNT(*) AS c FROM cars").fetchone()["c"]
        if count == 0:
            conn.executemany(
                "INSERT INTO cars (make, model, keywords) VALUES (?, ?, ?)",
                CATALOG,
            )
            log.info("Seeded %d car models", len(CATALOG))


def list_cars(db_path: Path | None = None) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, make, model, keywords FROM cars ORDER BY make, model"
        ).fetchall()
    return [dict(r) for r in rows]


def list_makes(db_path: Path | None = None) -> list[str]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT make FROM cars ORDER BY make"
        ).fetchall()
    return [r["make"] for r in rows]


def slot_times() -> list[str]:
    start = _parse_hhmm(config.office_hours_start)
    end = _parse_hhmm(config.office_hours_end)
    out: list[str] = []
    cur = datetime.combine(date.today(), start)
    limit = datetime.combine(date.today(), end)
    while cur < limit:
        out.append(cur.strftime("%H:%M"))
        cur += timedelta(minutes=30)
    return out


def _parse_hhmm(value: str) -> dtime:
    h, m = (value or "09:00").split(":")[:2]
    return dtime(int(h), int(m))


def is_office_open(day: date) -> bool:
    return day.weekday() != FRIDAY


def taken_times(day: str, db_path: Path | None = None) -> set[str]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT time FROM inspections
            WHERE date = ? AND status = 'booked'
            """,
            (day,),
        ).fetchall()
    return {r["time"] for r in rows}


def available_slots(day: str, db_path: Path | None = None) -> list[str]:
    d = date.fromisoformat(day)
    if not is_office_open(d):
        return []
    taken = taken_times(day, db_path)
    now = datetime.now()
    slots = []
    for hhmm in slot_times():
        if hhmm in taken:
            continue
        if d == now.date():
            h, m = map(int, hhmm.split(":"))
            if datetime.combine(d, dtime(h, m)) <= now:
                continue
        slots.append(hhmm)
    return slots


def is_slot_free(day: str, time: str, db_path: Path | None = None) -> bool:
    return time in available_slots(day, db_path)


def next_open_slots(limit: int = 6, db_path: Path | None = None) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    start = date.today()
    for i in range(60):
        day = start + timedelta(days=i)
        iso = day.isoformat()
        for hhmm in available_slots(iso, db_path):
            found.append(
                {
                    "date": iso,
                    "time": hhmm,
                    "label": f"{format_jalali(day)} ساعت {hhmm}",
                }
            )
            if len(found) >= limit:
                return found
    return found


def month_calendar(jy: int, jm: int, db_path: Path | None = None) -> dict[str, Any]:
    weeks = []
    for week in jalali_month_matrix(jy, jm):
        row = []
        for d in week:
            if d is None:
                row.append(None)
                continue
            iso = d.isoformat()
            free = available_slots(iso, db_path)
            row.append(
                {
                    "date": iso,
                    "jalali_day": to_jalali(d)[2],
                    "open": is_office_open(d) and d >= date.today(),
                    "free_count": len(free) if d >= date.today() else 0,
                    "weekday": d.weekday(),
                    "today": d == date.today(),
                }
            )
        weeks.append(row)
    return {
        "year": jy,
        "month": jm,
        "weeks": weeks,
        "address": config.office_address,
        "slot_minutes": 30,
    }


def book_inspection(
    seller_name: str,
    phone: str,
    make: str,
    model: str,
    year: str,
    km: int | None,
    day: str,
    time: str,
    db_path: Path | None = None,
) -> int:
    if not is_slot_free(day, time, db_path):
        raise ValueError("این ساعت قبلاً رزرو شده است. وقت خالی دیگری انتخاب کنید.")
    created = datetime.now().isoformat(timespec="seconds")
    try:
        with get_conn(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections
                    (seller_name, phone, make, model, year, km, date, time, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'booked')
                """,
                (seller_name, phone or "", make, model, year or "", km, day, time, created),
            )
            appt_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError("این ساعت قبلاً رزرو شده است. وقت خالی دیگری انتخاب کنید.") from exc
    log.info("Booked inspection id=%s %s %s %s %s", appt_id, seller_name, make, day, time)
    return appt_id


def get_inspection(appointment_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (appointment_id,)
        ).fetchone()
    return dict(row) if row else None


def list_inspections(db_path: Path | None = None, limit: int = 80) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM inspections
            ORDER BY date DESC, time DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["jalali"] = format_jalali(date.fromisoformat(item["date"]))
        except ValueError:
            item["jalali"] = item["date"]
        out.append(item)
    return out


def find_car(text: str) -> dict | None:
    return match_car(normalize_persian(text))


def snap_heard_text(text: str, db_path: Path | None = None) -> str:
    from src.lexicon import resolve_car

    hit = resolve_car(text)
    if not hit:
        return normalize_persian(text)
    if hit.get("model"):
        return f"{hit['make']} {hit['model']}"
    return hit["make"]
