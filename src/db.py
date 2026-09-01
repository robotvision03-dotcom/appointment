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

            CREATE TABLE IF NOT EXISTS services (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                keywords TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                area TEXT,
                FOREIGN KEY (service_id) REFERENCES services(id)
            );

            CREATE TABLE IF NOT EXISTS service_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT,
                customer_phone TEXT NOT NULL,
                provider_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                call_status TEXT,
                sms_status TEXT,
                FOREIGN KEY (provider_id) REFERENCES providers(id)
            );
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
        _ensure_service_catalog(conn)


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


def find_doctor_by_name(
    name: str,
    db_path: Path | None = None,
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Match a spoken doctor. strict=True avoids treating a patient name like علی رضایی as دکتر رضایی."""
    if not name:
        return None
    raw = name.replace("ي", "ی")
    titled = "دکتر" in raw or "doct" in raw.lower()
    tokens = [t for t in raw.replace("دکتر", " ").split() if len(t) >= 2]
    doctors = list_doctors(db_path)

    for doc in doctors:
        spec = (doc.get("specialty") or "").strip()
        if spec and len(spec) >= 3 and spec in raw:
            return doc

    for doc in doctors:
        last = doc["name"].replace("دکتر", "").strip()
        if not last:
            continue
        if last not in tokens and last not in raw:
            continue
        if titled or raw.strip() in {last, doc["name"], f"دکتر {last}"}:
            return doc
        if not strict and last in tokens and len(tokens) <= 4:
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


SEED_SERVICES = [
    {
        "name": "آرایشگر",
        "keywords": "آرایش آرایشگر آرایشگاه پیرایش سلمانی مو کوتاهی سالن زیبایی",
        "providers": [
            ("سالن گلبرگ", "09121001001", "ونک"),
            ("پیرایش نو", "09121001002", "انقلاب"),
            ("آرایشگاه نیلوفر", "09121001003", "سعادت‌آباد"),
            ("سالن مردانه کاسپین", "09121001004", "تجریش"),
            ("آکادمی زیبایی رز", "09121001005", "شهرک غرب"),
        ],
    },
    {
        "name": "مکانیک",
        "keywords": "مکانیک ماشین خودرو تعمیر باتری پنچری تعمیرگاه امداد",
        "providers": [
            ("تعمیرگاه آزادی", "09121111001", "آزادی"),
            ("امداد خودرو پارس", "09121111002", "تهرانپارس"),
            ("اتوسرویس شریف", "09121111003", "صادقیه"),
            ("تعمیرگاه بهمن", "09121111004", "جاده ساوه"),
            ("باتری و پنچری نصر", "09121111005", "نواب"),
        ],
    },
    {
        "name": "اورژانس",
        "keywords": "اورژانس آمبولانس تصادف اورژانسی ۱۱۵ 115",
        "providers": [
            ("اورژانس ۱۱۵", "115", "سراسر شهر"),
            ("درمانگاه شبانه‌روزی نور", "09121222001", "شریعتی"),
            ("اورژانس خصوصی سپهر", "09121222002", "ولیعصر"),
            ("درمانگاه شبانه‌روزی سپید", "09121222003", "پیروزی"),
            ("آمبولانس مهر", "09121222004", "رسالت"),
        ],
    },
    {
        "name": "پزشک",
        "keywords": "پزشک دکتر مطب ویزیت بیمارستان نوبت درمانگاه",
        "providers": [
            ("دکتر کریمی داخلی", "09121333001", "مطهری"),
            ("دکتر نوری اطفال", "09121333002", "نیاوران"),
            ("دکتر احمدی قلب", "09121333003", "جردن"),
            ("دکتر موسوی زنان", "09121333004", "زعفرانیه"),
            ("دکتر رضایی پوست", "09121333005", "سعادت‌آباد"),
        ],
    },
    {
        "name": "لوله‌کش",
        "keywords": "لوله لوله‌کش لوله‌کشی چکه فاضلاب سیفون تأسیسات",
        "providers": [
            ("تأسیسات رضایی", "09121444001", "پونک"),
            ("لوله‌کشی آریا", "09121444002", "تهرانپارس"),
            ("خدمات فاضلاب شهر", "09121444003", "افسریه"),
            ("تأسیسات شبانه پاسارگاد", "09121444004", "ستارخان"),
            ("رفع گرفتگی پایپ‌فیکس", "09121444005", "شهرری"),
        ],
    },
    {
        "name": "برقکار",
        "keywords": "برق برقکار سیم‌کشی فیوز روشنایی ساختمان",
        "providers": [
            ("برق ساختمان کاظمی", "09121555001", "سعادت‌آباد"),
            ("برق صنعتی نوران", "09121555002", "جاده مخصوص"),
            ("سیم‌کشی خانه سبز", "09121555003", "پونک"),
            ("رفع اتصالی فوری", "09121555004", "آزادی"),
            ("روشنایی و تابلو برق هما", "09121555005", "شهرک اکباتان"),
        ],
    },
]


def _ensure_service_catalog(conn: sqlite3.Connection) -> None:
    """Insert sample services/providers; add missing phones if the DB was seeded earlier."""
    added = 0
    for svc in SEED_SERVICES:
        row = conn.execute("SELECT id FROM services WHERE name = ?", (svc["name"],)).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO services (name, keywords) VALUES (?, ?)",
                (svc["name"], svc["keywords"]),
            )
            sid = int(cur.lastrowid)
        else:
            sid = int(row["id"])
            conn.execute(
                "UPDATE services SET keywords = ? WHERE id = ?",
                (svc["keywords"], sid),
            )
        have = {
            r["phone"]
            for r in conn.execute(
                "SELECT phone FROM providers WHERE service_id = ?", (sid,)
            ).fetchall()
        }
        have_names = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM providers WHERE service_id = ?", (sid,)
            ).fetchall()
        }
        for name, phone, area in svc["providers"]:
            if name in have_names:
                conn.execute(
                    "UPDATE providers SET phone = ?, area = ? WHERE service_id = ? AND name = ?",
                    (phone, area, sid, name),
                )
                have.add(phone)
                continue
            if phone in have:
                continue
            conn.execute(
                "INSERT INTO providers (service_id, name, phone, area) VALUES (?, ?, ?, ?)",
                (sid, name, phone, area),
            )
            have.add(phone)
            have_names.add(name)
            added += 1
    if added:
        log.info("Service catalog updated (%d new providers)", added)


def list_services(db_path: Path | None = None) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT id, name, keywords FROM services ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def find_service(text: str, db_path: Path | None = None) -> dict[str, Any] | None:
    t = (text or "").replace("ي", "ی")
    for svc in list_services(db_path):
        if svc["name"] in t:
            return svc
        for kw in (svc.get("keywords") or "").split():
            if kw and kw in t:
                return svc
    return None


def list_providers(service_id: int, db_path: Path | None = None) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.service_id, p.name, p.phone, p.area, s.name AS service_name
            FROM providers p JOIN services s ON s.id = p.service_id
            WHERE p.service_id = ? ORDER BY p.id
            """,
            (service_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_provider(provider_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT p.id, p.service_id, p.name, p.phone, p.area, s.name AS service_name
            FROM providers p JOIN services s ON s.id = p.service_id
            WHERE p.id = ?
            """,
            (provider_id,),
        ).fetchone()
    return dict(row) if row else None


_ORDINALS = {
    "اول": 1,
    "یکم": 1,
    "اولی": 1,
    "دوم": 2,
    "دومی": 2,
    "سوم": 3,
    "سومی": 3,
    "چهارم": 4,
    "پنجم": 5,
    "ششم": 6,
}


def find_provider(text: str, service_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    import re

    t = (text or "").replace("ي", "ی")
    providers = list_providers(service_id, db_path)
    numbered = None
    for word, idx in _ORDINALS.items():
        if word in t and 1 <= idx <= len(providers):
            numbered = providers[idx - 1]
            break
    m = re.search(r"(\d+)", t)
    if numbered is None and m:
        idx = int(m.group(1))
        if 1 <= idx <= len(providers):
            numbered = providers[idx - 1]
    for p in providers:
        if p["name"] in t:
            return p
        bits = [part for part in p["name"].split() if len(part) > 2 and part not in {"دکتر"}]
        if bits and all(part in t for part in bits):
            return p
        if any(part in t for part in bits if len(part) >= 4):
            return p
    return numbered


def save_service_request(
    customer_name: str,
    customer_phone: str,
    provider_id: int,
    call_status: str,
    sms_status: str,
    db_path: Path | None = None,
) -> int:
    created = datetime.now().isoformat(timespec="seconds")
    with get_conn(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO service_requests
                (customer_name, customer_phone, provider_id, created_at, call_status, sms_status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (customer_name, customer_phone, provider_id, created, call_status, sms_status),
        )
        return int(cur.lastrowid)


def list_service_requests(db_path: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.id, r.customer_name, r.customer_phone, r.created_at,
                   r.call_status, r.sms_status, p.name AS provider_name, p.phone AS provider_phone,
                   s.name AS service_name
            FROM service_requests r
            JOIN providers p ON p.id = r.provider_id
            JOIN services s ON s.id = p.service_id
            ORDER BY r.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
