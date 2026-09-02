"""Unit tests for car-office dialogue and calendar slots."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
import tempfile

import pytest

os.environ.setdefault("DB_PATH", str(Path(tempfile.gettempdir()) / "pva_car_test.db"))

from src import db
from src.call_manager import call_manager
from src.cars import match_car, parse_km, parse_year
from src.jalali import from_jalali, gregorian_to_jalali, to_jalali
from src.lexicon import resolve_car
from src.stt import _pcm16_to_float32, _prepare_waveform, _sherpa_text


def test_sherpa_empty_result_is_not_json_dump():
    class Fake:
        text = ""

        def __str__(self):
            return '{"lang": "", "text": "", "ys_log_probs": []}'

    assert _sherpa_text(Fake()) == ""
    assert _sherpa_text({"text": "پژو پارس", "ys_log_probs": []}) == "پژو پارس"


def test_jalali_roundtrip():
    jy, jm, jd = gregorian_to_jalali(2026, 9, 2)
    g = from_jalali(jy, jm, jd)
    assert g == date(2026, 9, 2)
    assert to_jalali(date(2026, 9, 2)) == (jy, jm, jd)


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    path = tmp_path / "cars.db"
    monkeypatch.setattr(db.config, "db_path", path)
    from src.config import config as app_config

    monkeypatch.setattr(app_config, "db_path", path)
    db.init_db(path)
    return path


def test_seed_cars(isolated_db):
    cars = db.list_cars(isolated_db)
    assert len(cars) >= 40
    assert any(c["make"] == "پژو" and c["model"] == "پارس" for c in cars)


def test_match_peugeot_pars():
    hit = match_car("پژو پارس")
    assert hit is not None
    assert hit["make"] == "پژو"
    assert hit["model"] == "پارس"
    only_make = match_car("پژو")
    assert only_make["make"] == "پژو"
    assert not only_make["model"]


def test_lexicon_fixes_asr_fragments():
    samand = resolve_car("سمن")
    assert samand is not None
    assert "سمند" in (samand.get("model") or "")
    pars = resolve_car("پرس")
    assert pars is not None
    assert pars["model"] == "پارس"
    pride = resolve_car("پرا")
    assert pride is not None
    assert pride["model"] == "پراید"


def test_parse_year_and_km():
    assert parse_year("مدل ۱۳۹۹") == "1399"
    assert parse_year("۲۰۱۸") == "2018"
    assert parse_year("۹۹") == "1399"
    assert parse_year("۸۰ هزار") is None
    assert parse_year("مدل ۱۳۹۹ کارکرد ۸۰ هزار") == "1399"
    assert parse_km("۸۰ هزار") == 80000
    assert parse_km("120000") == 120000


def test_friday_closed(isolated_db):
    d = date.today()
    while d.weekday() != 4:
        d += timedelta(days=1)
    assert db.available_slots(d.isoformat(), isolated_db) == []


def test_slot_conflict(isolated_db):
    slots = db.next_open_slots(1, isolated_db)
    assert slots
    s = slots[0]
    aid = db.book_inspection("علی رضایی", "09120000000", "پژو", "پارس", "1399", 80000, s["date"], s["time"], isolated_db)
    assert aid >= 1
    with pytest.raises(ValueError):
        db.book_inspection("دوم", "09120000001", "سمند", "سمند", "1398", 10, s["date"], s["time"], isolated_db)


def test_pcm16_resamples_48k_to_16k():
    import numpy as np

    rate_in = 48000
    n = rate_in
    t = np.arange(n, dtype=np.float32) / rate_in
    tone = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()
    out = _pcm16_to_float32(tone, rate_in)
    assert 15000 < out.size < 17000


def test_prepare_waveform_boosts_quiet_speech():
    import numpy as np

    n = 16000
    t = np.arange(n, dtype=np.float32) / 16000
    quiet = (0.008 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = _prepare_waveform(quiet, 16000)
    rms = float(np.sqrt(np.mean(np.square(out[:n]))))
    assert rms > 0.04


def test_full_car_booking_dialogue(isolated_db):
    sid = "car-call"
    call_manager.end_call(sid)
    call_manager.start_call(sid)
    r = call_manager.handle_user_text(sid, "پژو پارس")
    assert r["phase"] == "ask_year"
    assert r["patient_info"]["make"] == "پژو"
    r = call_manager.handle_user_text(sid, "۱۳۹۹")
    assert r["phase"] == "ask_km"
    assert r["patient_info"]["year"] == "1399"
    r = call_manager.handle_user_text(sid, "هشتاد هزار")
    assert r["patient_info"]["km"] == 80000
    assert r["phase"] == "ask_name"
    r = call_manager.handle_user_text(sid, "علی رضایی")
    assert r["phase"] == "ask_slot"
    assert "ایثار" in r["reply"] or "هزینه" in r["reply"]
    r = call_manager.handle_user_text(sid, "بله")
    assert r["intent"] == "book"
    assert r["phase"] == "booked"
    assert "ایثار" in r["reply"]
    assert "هیچ هزینه‌ای" in r["reply"]
    rows = db.list_inspections(isolated_db)
    assert rows[0]["seller_name"] == "علی رضایی"
    assert rows[0]["make"] == "پژو"


def test_mileage_during_year_question_is_not_year(isolated_db):
    sid = "year-km-mix"
    call_manager.end_call(sid)
    call_manager.start_call(sid)
    call_manager.handle_user_text(sid, "پژو پارس")
    r = call_manager.handle_user_text(sid, "۸۰ هزار")
    assert r["phase"] == "ask_year"
    assert r["patient_info"].get("year") in (None, "")
    assert r["patient_info"]["km"] == 80000
    r = call_manager.handle_user_text(sid, "۱۳۹۹")
    assert r["patient_info"]["year"] == "1399"
    assert r["phase"] == "ask_name"
