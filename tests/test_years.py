"""Spoken Shamsi model years, 1370–1410."""

from __future__ import annotations

import pytest

from src.cars import parse_km, parse_year
from src.years import MAX_YEAR, MIN_YEAR, YEAR_TABLE, parse_shamsi_year, year_words


@pytest.mark.parametrize(
    "said,expected",
    [
        ("هزار و سیصد و هشتاد و هشت", 1388),
        ("یک هزار و سیصد و هشتاد و هشت", 1388),
        ("سیصد و هشتاد و هشت", 1388),
        ("هشتاد و هشت", 1388),
        ("هشت و هشت", 1388),
        ("هشت هشت", 1388),
        ("یک سه هشت هشت", 1388),
        ("یک سه نه نه", 1399),
        ("یک سه نو نو", 1399),
        ("نود و نه", 1399),
        ("هزار و چهارصد", 1400),
        ("چهارصد و پنج", 1405),
        ("هفتاد", 1370),
        ("۱۳۸۸", 1388),
        ("۸۸", 1388),
        ("388", 1388),
    ],
)
def test_spoken_years(said, expected):
    assert parse_shamsi_year(said) == expected


def test_trailing_conjunction_is_not_a_finished_year():
    """«هشتاد و» is the speaker still going, not 1380."""
    assert parse_shamsi_year("یک هزار و سیصد و هشتاد و") is None
    assert parse_shamsi_year("هزار و سیصد و هشتاد و") is None
    assert parse_shamsi_year("یک هزار و سیصد و هشتاد و هشت") == 1388


def test_year_inside_a_sentence():
    assert parse_shamsi_year("ماشین مدل هشتاد و هشت است") == 1388
    assert parse_shamsi_year("مدل ۹۹ هست") == 1399


def test_non_years_are_rejected():
    assert parse_shamsi_year("") is None
    assert parse_shamsi_year("پژو پارس") is None
    assert parse_shamsi_year("هزار و نهصد") is None


def test_every_year_round_trips():
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        assert parse_shamsi_year(year_words(year)) == year
        assert parse_shamsi_year(str(year)) == year


def test_table_covers_the_whole_range():
    assert set(YEAR_TABLE.values()) == set(range(MIN_YEAR, MAX_YEAR + 1))


def test_parse_year_uses_spoken_forms():
    assert parse_year("هزار و سیصد و هشتاد و هشت") == "1388"
    assert parse_year("هشت و هشت") == "1388"
    assert parse_year("۲۰۱۸") == "2018"


def test_spoken_year_is_not_mileage():
    # «سیصد» must not read as «سی» → 30 هزار.
    assert parse_km("سیصد و هشتاد و هشت") != 30000
    assert parse_km("هشتاد هزار") == 80000
    assert parse_year("۸۰ هزار") is None
