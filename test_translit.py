"""Unit tests for translit_uz.  Run:  python -m pytest -q test_translit.py"""
import pytest
from translit_uz import lat2cyr, cyr2lat, latin_normalised, script_share

CASES = [
    ("Oʻzbekiston", "Ўзбекистон"),
    ("o'zbek", "ўзбек"),
    ("gʻarb", "ғарб"),
    ("yoʻq", "йўқ"),
    ("yo'l", "йўл"),
    ("yosh", "ёш"),
    ("yangi", "янги"),
    ("yuz", "юз"),
    ("yer", "ер"),
    ("ma'no", "маъно"),
    ("shahar", "шаҳар"),
    ("choy", "чой"),
    ("ongli", "онгли"),
    ("ekran", "экран"),
    ("teatr", "театр"),
    ("poezd", "поэзд"),
    ("Toshkent", "Тошкент"),
    ("qalam", "қалам"),
    ("xalq", "халқ"),
    ("hayot", "ҳаёт"),
    ("informatsiya", "информация"),
    ("ketsa", "кетса"),         # 'ts' must NOT become ц here
    ("Rossiya", "Россия"),
    ("SHAHAR", "ШАҲАР"),
    ("Shahar", "Шаҳар"),
    ("1-sentabr, 2024!", "1-сентабр, 2024!"),
]


@pytest.mark.parametrize("lat,cyr", CASES)
def test_lat2cyr(lat, cyr):
    assert lat2cyr(lat) == cyr


def test_apostrophe_variants_equal():
    variants = ["o'zbek", "oʻzbek", "o’zbek", "o‘zbek", "o`zbek", "oʼzbek"]
    assert len({lat2cyr(v) for v in variants}) == 1


def test_no_latin_left():
    s = lat2cyr("Turkiya uch tomondan dengizlar bilan oʻralgan: gʻarbda Egey dengizi.")
    assert script_share(s)["latin"] == 0.0


def test_roundtrip_mostly_identity():
    for lat in ["oʻzbek tili", "gʻoya", "yoʻl", "yangi kun", "shahar", "maʼno"]:
        back = cyr2lat(lat2cyr(lat))
        assert latin_normalised(back).replace("ʻ", "'").replace("ʼ", "'") == \
               latin_normalised(lat).replace("ʻ", "'").replace("ʼ", "'")
