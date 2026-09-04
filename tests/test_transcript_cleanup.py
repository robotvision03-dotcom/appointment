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


def test_foreign_alphabets_are_suppressed_at_the_decoder():
    """«پژو پارس» once came back as «ежоپарс» — all Cyrillic but one letter.

    Cleaning cannot rebuild that word, so the decoder must never reach those
    tokens in the first place.
    """
    import re

    from src.whisper_fa import _persian_only_tokens

    class FakeTokenizer:
        vocab = ["پژو", " پارس", "ежо", "арс", "Peugeot", "۱۳۸۸", ".", "Ω"]

        def get_vocab_size(self):
            return len(self.vocab)

        def decode(self, ids):
            return self.vocab[ids[0]]

    suppressed = set(_persian_only_tokens(FakeTokenizer()))
    assert -1 in suppressed  # Whisper's own non-speech list is kept
    kept = [t for i, t in enumerate(FakeTokenizer.vocab) if i not in suppressed]
    assert kept == ["پژو", " پارس", "۱۳۸۸", "."]
    assert not any(re.search(r"[A-Za-z\u0400-\u04FF]", t) for t in kept)
