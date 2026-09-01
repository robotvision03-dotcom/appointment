"""Unit tests that do not require Whisper, Piper, Ollama, or Twilio."""

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


def test_iran_mobile_normalize():
    from src.sms import extract_iran_mobile, normalize_iran_mobile, send_sms

    assert normalize_iran_mobile("09121234567") == "989121234567"
    assert normalize_iran_mobile("+98 912 123 4567") == "989121234567"
    assert extract_iran_mobile("شماره‌ام ۰۹۱۲۱۲۳۴۵۶۷ است") == "09121234567"
    result = send_sms("09121234567", "نوبت ثبت شد")
    assert result["ok"] is False
    assert result["reason"] == "kavenegar_not_configured"


def test_dispatch_dialogue(isolated_db, monkeypatch):
    monkeypatch.setattr("src.call_manager.llm.is_available", lambda: False)
    sid = "unit-call"
    call_manager.end_call(sid)
    call_manager.start_call(sid)
    r = call_manager.handle_user_text(sid, "مکانیک")
    assert r["patient_info"]["service_name"] == "مکانیک"
    assert r["phase"] == "ask_provider"
    assert "آزادی" in r["reply"]
    r = call_manager.handle_user_text(sid, "اول")
    assert r["phase"] == "ask_phone"
    r = call_manager.handle_user_text(sid, "09121234567")
    assert r["intent"] == "connect"
    assert r["connect"]["ok"] is True
    assert r["connect"]["provider"]["name"] == "تعمیرگاه آزادی"
    assert r["connect"]["sms"]["dry_run"] is True
    assert r["connect"]["sms_to_customer"]["dry_run"] is True
    assert "09121111001" in (r["connect"]["sms_to_customer"].get("message") or "")
    rows = db.list_service_requests(isolated_db)
    assert rows[0]["customer_phone"].endswith("9121234567") or "09121234567" in rows[0]["customer_phone"]


def test_salon_keyword_and_five_barbers(isolated_db):
    svc = db.find_service("آرایشگاه", isolated_db)
    assert svc is not None
    assert svc["name"] == "آرایشگر"
    barbers = db.list_providers(svc["id"], isolated_db)
    assert len(barbers) >= 5
    assert all(p["phone"] for p in barbers)
    call_manager.end_call("salon-kw")
    call_manager.start_call("salon-kw")
    r = call_manager.handle_user_text("salon-kw", "آرایشگاه")
    assert r["phase"] == "ask_provider"
    assert "نیلوفر" in r["reply"]
    assert "گلبرگ" in r["reply"]


def test_connect_api_emergency(isolated_db):
    from src.dispatch import connect_customer_to_provider

    emergency = next(p for p in db.list_providers(db.find_service("اورژانس")["id"]) if p["phone"] == "115")
    out = connect_customer_to_provider("09120000000", emergency["id"], "آزمایش")
    assert out["ok"] is True
    assert out["call"]["reason"] == "emergency_number"


def test_patient_name_not_confused_with_doctor(isolated_db, monkeypatch):
    monkeypatch.setattr("src.call_manager.llm.is_available", lambda: False)
    sid = "name-vs-doctor"
    call_manager.end_call(sid)
    call_manager.start_call(sid)
    r = call_manager.handle_user_text(sid, "پزشک")
    assert r["patient_info"]["service_name"] == "پزشک"
    r = call_manager.handle_user_text(sid, "دکتر کریمی")
    assert "کریمی" in r["patient_info"].get("provider_name", "")


def test_full_booking_dialogue(isolated_db, monkeypatch):
    monkeypatch.setattr("src.call_manager.llm.is_available", lambda: False)
    docs = db.list_doctors(isolated_db)
    kid = next(d["id"] for d in docs if "نوری" in d["name"])
    aid = db.book_appointment("مریم حسینی", "09120000000", kid, "2026-09-02", "11:00", isolated_db)
    appt = db.get_appointment(aid, isolated_db)
    assert appt["patient_name"] == "مریم حسینی"
    assert "نوری" in appt["doctor_name"]


def test_handoff_click_to_call():
    from src.handoff import start_warm_transfer

    sid = "handoff-call"
    call_manager.end_call(sid)
    call_manager.start_call(sid, from_number="09120000000")
    call_manager.handle_user_text(sid, "می‌خواهم با منشی صحبت کنم")
    out = start_warm_transfer(sid)
    assert out["ok"] is True
    assert out["method"] == "click_to_call"
    assert out["tel_url"].startswith("tel:")
