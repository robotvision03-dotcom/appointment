"""Whisper writes Latin letters and punctuation into Persian car names."""

from __future__ import annotations

from src.lexicon import resolve_car
from src.whisper_fa import clean_transcript


def test_latin_letter_glued_into_a_persian_word_is_dropped():
    assert clean_transcript("Pژو پانس.") == "ژو پانس"
    assert clean_transcript("سمند.") == "سمند"


def test_pure_latin_model_codes_survive():
    assert clean_transcript("MVM X22") == "MVM X22"


def test_numeric_answers_survive():
    # A year, a mileage, or a phone number is often the whole utterance.
    assert clean_transcript("۱۳۸۸") == "۱۳۸۸"
    assert clean_transcript("۱۳۸۸.") == "۱۳۸۸"
    assert clean_transcript("1388") == "1388"
    assert clean_transcript("80000") == "80000"
    assert clean_transcript("09121234567") == "09121234567"


def test_punctuation_only_output_is_empty():
    assert clean_transcript(".") == ""
    assert clean_transcript("..؟") == ""
    assert clean_transcript("\u200c") == ""
    assert clean_transcript("  ") == ""
    assert clean_transcript("") == ""


def test_dirty_whisper_output_still_resolves_to_the_car():
    car = resolve_car(clean_transcript("Pژو پانس."))
    assert car is not None
    assert car["make"] == "پژو"
    assert car["model"] == "پارس"
