"""Unit tests that do not require Vosk, Piper, Ollama, or Twilio."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("DB_PATH", str(Path(tempfile.gettempdir()) / "pva_test.db"))

from src import db
from src.call_manager import call_manager
from src.utils import parse_relative_date, parse_time, wants_transfer


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "appointments.db"
    monkeypatch.setattr(db.config, "db_path", path)
    from src.config import config as app_config

    monkeypatch.setattr(app_config, "db_path", path)
    db.init_db(path)
    return path


def test_seed_doctors(isolated_db):
    docs = db.list_doctors(isolated_db)
    assert len(docs) >= 5
    assert any("کریمی" in d["name"] for d in docs)


def test_booking_conflict(isolated_db):
    docs = db.list_doctors(isolated_db)
    kid = docs[0]["id"]
    aid = db.book_appointment("آزمایش", "+98", kid, "2026-09-01", "10:00", isolated_db)
    assert aid >= 1
    slots = db.get_available_slots(kid, "2026-09-01", isolated_db)
    assert "10:00" not in slots
    with pytest.raises(ValueError):
        db.book_appointment("دوم", "+98", kid, "2026-09-01", "10:00", isolated_db)


def test_parse_time_and_date():
    assert parse_time("ساعت ده صبح") == "10:00"
    assert parse_time("۱۴:۳۰") == "14:30"
    assert parse_time("ساعت ۵ عصر") == "17:00"
    d = parse_relative_date("فردا")
    assert d is not None and len(d) == 10


def test_transfer_phrase():
    assert wants_transfer("می‌خواهم با منشی صحبت کنم")


def test_full_booking_dialogue(isolated_db, monkeypatch):
    monkeypatch.setattr("src.call_manager.llm.is_available", lambda: False)
    sid = "unit-call"
    call_manager.end_call(sid)
    call_manager.start_call(sid, from_number="+989121234567")
    steps = [
        "مریم حسینی",
        "دکتر نوری",
        "فردا",
        "ساعت یازده صبح",
        "بله",
    ]
    result = None
    for line in steps:
        result = call_manager.handle_user_text(sid, line)
    assert result is not None
    assert result["intent"] == "book"
    assert result["appointment_id"]
    appt = db.get_appointment(result["appointment_id"], isolated_db)
    assert appt["patient_name"] == "مریم حسینی"
    assert "نوری" in appt["doctor_name"]
