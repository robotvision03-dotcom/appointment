"""SQLite models and CRUD for doctors and appointments."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.config import config
from src.utils import log

DEFAULT_SLOTS = [
    "09:00",
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
    "16:30",
    "17:00",
]

SEED_DOCTORS = [
    {
        "name": "دکتر احمدی",
        "specialty": "قلب و عروق",
        "available_days": ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه"],
    },
    {
        "name": "دکتر رضایی",
        "specialty": "پوست",
        "available_days": ["شنبه", "دوشنبه", "چهارشنبه"],
    },
    {
        "name": "دکتر کریمی",
        "specialty": "داخلی",
        "available_days": ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه"],
    },
    {
        "name": "دکتر نوری",
        "specialty": "اطفال",
        "available_days": ["یکشنبه", "سه‌شنبه", "پنجشنبه"],
    },
    {
        "name": "دکتر موسوی",
        "specialty": "زنان و زایمان",
        "available_days": ["شنبه", "دوشنبه", "چهارشنبه"],
    },
]


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
    """Create tables and seed sample doctors if the table is empty."""
    with get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                specialty TEXT NOT NULL,
                available_days TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                doctor_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                phone_number TEXT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'booked',
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            );

            CREATE INDEX IF NOT EXISTS idx_appt_doctor_date
                ON appointments(doctor_id, date, time);
            """
        )
        count = conn.execute("SELECT COUNT(*) AS c FROM doctors").fetchone()["c"]
        if count == 0:
            for doc in SEED_DOCTORS:
                conn.execute(
                    "INSERT INTO doctors (name, specialty, available_days) VALUES (?, ?, ?)",
                    (doc["name"], doc["specialty"], json.dumps(doc["available_days"], ensure_ascii=False)),
                )
            log.info("Seeded %d doctors", len(SEED_DOCTORS))


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    if "available_days" in data and isinstance(data["available_days"], str):
        try:
            data["available_days"] = json.loads(data["available_days"])
        except json.JSONDecodeError:
            data["available_days"] = []
    return data


def list_doctors(db_path: Path | None = None) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, specialty, available_days FROM doctors ORDER BY id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def get_doctor(doctor_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, specialty, available_days FROM doctors WHERE id = ?",
            (doctor_id,),
        ).fetchone()
    return _row_to_dict(row)


def find_doctor_by_name(name: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Match a spoken doctor name against the roster (substring, specialty, last name)."""
    if not name:
        return None
    needle = name.replace("دکتر", "").replace("دکتر ", "").strip()
    doctors = list_doctors(db_path)
    for doc in doctors:
        hay = f"{doc['name']} {doc['specialty']}"
        if needle and needle in hay:
            return doc
        last = doc["name"].replace("دکتر", "").strip()
        if last and last in name:
            return doc
    return None


def get_available_slots(doctor_id: int, date: str, db_path: Path | None = None) -> list[str]:
    """Return free times for a doctor on a date. Occupied booked slots are excluded."""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT time FROM appointments
            WHERE doctor_id = ? AND date = ? AND status = 'booked'
            """,
            (doctor_id, date),
        ).fetchall()
    taken = {r["time"] for r in rows}
    return [s for s in DEFAULT_SLOTS if s not in taken]


def is_slot_free(doctor_id: int, date: str, time: str, db_path: Path | None = None) -> bool:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT id FROM appointments
            WHERE doctor_id = ? AND date = ? AND time = ? AND status = 'booked'
            """,
            (doctor_id, date, time),
        ).fetchone()
    return row is None


def book_appointment(
    patient_name: str,
    phone: str,
    doctor_id: int,
    date: str,
    time: str,
    db_path: Path | None = None,
) -> int:
    """Insert a booked appointment and return its id. Raises ValueError if the slot is taken."""
    if not is_slot_free(doctor_id, date, time, db_path):
        raise ValueError("این ساعت قبلاً رزرو شده است.")
    created = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO appointments
                (patient_name, doctor_id, date, time, phone_number, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'booked')
            """,
            (patient_name, doctor_id, date, time, phone, created),
        )
        appt_id = int(cur.lastrowid)
    log.info(
        "Booked appointment id=%s patient=%s doctor_id=%s %s %s",
        appt_id,
        patient_name,
        doctor_id,
        date,
        time,
    )
    return appt_id


def cancel_appointment(appointment_id: int, db_path: Path | None = None) -> bool:
    with get_conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE appointments SET status = 'cancelled' WHERE id = ? AND status = 'booked'",
            (appointment_id,),
        )
        return cur.rowcount > 0


def list_appointments(db_path: Path | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.patient_name, a.doctor_id, d.name AS doctor_name,
                   d.specialty, a.date, a.time, a.phone_number, a.created_at, a.status
            FROM appointments a
            JOIN doctors d ON d.id = a.doctor_id
            ORDER BY a.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_appointment(appointment_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT a.id, a.patient_name, a.doctor_id, d.name AS doctor_name,
                   d.specialty, a.date, a.time, a.phone_number, a.created_at, a.status
            FROM appointments a
            JOIN doctors d ON d.id = a.doctor_id
            WHERE a.id = ?
            """,
            (appointment_id,),
        ).fetchone()
    return dict(row) if row else None
