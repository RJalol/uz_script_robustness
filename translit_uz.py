"""
translit_uz.py -- rule-based Uzbek Latin <-> Cyrillic transliteration.

Design goals
------------
* Deterministic and fully documented (so the paper can state the exact rules).
* Latin -> Cyrillic is the direction used in the experiments.
* Cyrillic -> Latin is provided for round-trip diagnostics only.

KNOWN, UNAVOIDABLE LIMITATIONS (must be reported in the paper)
--------------------------------------------------------------
1. Russian loanwords: Cyrillic Uzbek keeps etymological letters (ц, щ, ь, ы, ъ in
   some loans, ё, ю, я in loans) that the Latin script has already merged
   (e.g. s for both с and ц). A Latin->Cyrillic mapping therefore CANNOT recover them.
   A small stem list (TS_STEMS) restores ц in the commonest -tsiya/-tsion words;
   everything else is rendered with the regular mapping.
2. 'sh', 'ch', 'ng' digraphs are treated as single units; the rare sequences s+h,
   c+h, n+g (e.g. 'mashhur' is fine, 'ongli' is fine) are not disambiguated.
3. 'ye/yo/yu/ya' -> е/ё/ю/я, 'y' + other -> й.
4. 'e' -> э at word start or after a vowel, otherwise е.
5. Apostrophe-like characters are normalised first (see APOSTROPHES):
   after o/g -> modifier (ў/ғ); elsewhere -> tutuq belgisi (ъ).
   NOTE: this normalisation is itself a typographic intervention; the paper's
   "latin_norm" control condition isolates it from the script change.
"""
from __future__ import annotations

import re
import unicodedata

# characters that all stand for the Uzbek modifier / tutuq in real-world text
APOSTROPHES = "'`´ʻʼ‘’ʹ′ʻʼ‘’"
_APOS_RE = re.compile("[" + re.escape(APOSTROPHES) + "]")
CANON_APOS = "ʻ"  # we normalise to U+02BB internally

LAT_VOWELS = set("aeiouAEIOU")
CYR_VOWELS = set("аеёиоуэюяўАЕЁИОУЭЮЯЎ")

# Russian-origin stems where Latin 'ts' corresponds to Cyrillic 'ц'
TS_STEMS = [
    "tsiya", "tsiy", "tsion", "tsent", "tsikl", "tsirk", "tsement", "tsiv",
    "militsiy", "polytsi", "sentsion",
]

# ---------------------------------------------------------------- Latin -> Cyrillic
_L2C_SINGLE = {
    "a": "а", "b": "б", "d": "д", "f": "ф", "g": "г", "h": "ҳ", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "қ", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "x": "х",
    "y": "й", "z": "з",
    # letters occasionally present in loanwords / proper names
    "c": "к", "w": "в",
}


def normalize_apostrophes(text: str, to: str = CANON_APOS) -> str:
    """Map every apostrophe-like character to a single canonical character."""
    return _APOS_RE.sub(to, text)


def _case_like(src: str, out: str) -> str:
    return out.upper() if src[:1].isupper() else out


def _safe_lower(t: str) -> str:
    """Per-character lowercase that never changes the string length
    (str.lower() can expand e.g. 'İ' to two characters and break index alignment)."""
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in t)


def lat2cyr(text: str) -> str:
    """Latin Uzbek -> Cyrillic Uzbek (see module docstring for limitations)."""
    t = normalize_apostrophes(unicodedata.normalize("NFC", text))
    out: list[str] = []
    n = len(t)
    i = 0
    prev_is_vowel = False  # previous *Latin* char was a vowel letter
    at_word_start = True
    low = _safe_lower(t)

    while i < n:
        ch = t[i]
        lc = low[i]
        nxt = low[i + 1] if i + 1 < n else ""
        nxt2 = low[i + 2] if i + 2 < n else ""

        if not ch.isalpha() and ch != CANON_APOS:
            out.append(ch)
            at_word_start = True
            prev_is_vowel = False
            i += 1
            continue

        # ---- ts-stems (Russian loans)
        if lc == "t" and nxt == "s":
            rest = low[i:i + 8]
            if any(rest.startswith(stem) for stem in TS_STEMS):
                out.append(_case_like(ch, "ц"))
                i += 2
                at_word_start = False
                prev_is_vowel = False
                continue

        # ---- modifier letters oʻ, gʻ
        if lc in "og" and nxt == CANON_APOS:
            out.append(_case_like(ch, "ў" if lc == "o" else "ғ"))
            i += 2
            at_word_start = False
            prev_is_vowel = (lc == "o")
            continue

        # ---- digraphs sh, ch, ng
        if lc == "s" and nxt == "h":
            out.append(_case_like(ch, "ш")); i += 2
            at_word_start = prev_is_vowel = False; continue
        if lc == "c" and nxt == "h":
            out.append(_case_like(ch, "ч")); i += 2
            at_word_start = prev_is_vowel = False; continue
        if lc == "n" and nxt == "g":
            # 'ng' -> 'нг' (two Cyrillic letters)
            if ch.isupper() and t[i + 1].isupper():
                out.append("НГ")
            elif ch.isupper():
                out.append("Нг")
            else:
                out.append("нг")
            i += 2
            at_word_start = prev_is_vowel = False; continue

        # ---- y + vowel
        if lc == "y":
            if nxt == "e":
                out.append(_case_like(ch, "е")); i += 2
                at_word_start = False; prev_is_vowel = True; continue
            if nxt == "o" and nxt2 != CANON_APOS:
                out.append(_case_like(ch, "ё")); i += 2
                at_word_start = False; prev_is_vowel = True; continue
            if nxt == "u":
                out.append(_case_like(ch, "ю")); i += 2
                at_word_start = False; prev_is_vowel = True; continue
            if nxt == "a":
                out.append(_case_like(ch, "я")); i += 2
                at_word_start = False; prev_is_vowel = True; continue
            out.append(_case_like(ch, "й")); i += 1
            at_word_start = False; prev_is_vowel = False; continue

        # ---- e
        if lc == "e":
            out.append(_case_like(ch, "э" if (at_word_start or prev_is_vowel) else "е"))
            i += 1
            at_word_start = False; prev_is_vowel = True; continue

        # ---- standalone apostrophe -> tutuq belgisi
        if ch == CANON_APOS:
            out.append("ъ"); i += 1
            at_word_start = False; prev_is_vowel = False; continue

        # ---- plain letters
        if lc in _L2C_SINGLE:
            out.append(_case_like(ch, _L2C_SINGLE[lc]))
            prev_is_vowel = lc in "aiou"
        else:
            out.append(ch)  # already Cyrillic or other script: keep
            prev_is_vowel = ch in CYR_VOWELS
        at_word_start = False
        i += 1

    return "".join(out)


# ---------------------------------------------------------------- Cyrillic -> Latin
_C2L_SINGLE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ч": "ch", "ш": "sh", "э": "e", "ю": "yu", "я": "ya", "ё": "yo",
    "ў": "o" + CANON_APOS, "қ": "q", "ғ": "g" + CANON_APOS, "ҳ": "h",
    "ъ": CANON_APOS, "ь": "", "ы": "i", "ц": "s", "щ": "shch",
}


def cyr2lat(text: str) -> str:
    out: list[str] = []
    prev_cyr_vowel = False
    at_word_start = True
    for ch in text:
        lc = ch.lower()
        if lc == "е":  # Cyrillic Ie
            r = "ye" if (at_word_start or prev_cyr_vowel) else "e"
            out.append(r.capitalize() if ch.isupper() else r)
        elif lc in _C2L_SINGLE:
            r = _C2L_SINGLE[lc]
            out.append(r.capitalize() if (ch.isupper() and r) else r)
        else:
            out.append(ch)
        if ch.isalpha():
            at_word_start = False
            prev_cyr_vowel = lc in CYR_VOWELS
        else:
            at_word_start = True
            prev_cyr_vowel = False
    return "".join(out)


# ---------------------------------------------------------------- helpers for the experiments
def latin_normalised(text: str) -> str:
    """CONTROL condition: Latin text with apostrophes canonicalised to ASCII "'".
    Isolates the effect of typographic normalisation from the script change."""
    return normalize_apostrophes(text, to="'")


def script_share(text: str) -> dict:
    """Return fraction of alphabetic characters per script (Latin / Cyrillic / other)."""
    lat = cyr = oth = 0
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("LATIN"):
            lat += 1
        elif name.startswith("CYRILLIC"):
            cyr += 1
        else:
            oth += 1
    tot = max(lat + cyr + oth, 1)
    return {"latin": lat / tot, "cyrillic": cyr / tot, "other": oth / tot}


def char_error_rate(ref: str, hyp: str) -> float:
    """Levenshtein CER (no external deps)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


if __name__ == "__main__":
    demo = "Oʻzbekiston Respublikasi, gʻarbda Yevropa; yoʻq, ma'no, ekran, ongli, Toshkent, informatsiya."
    print(demo)
    print(lat2cyr(demo))
    print(cyr2lat(lat2cyr(demo)))
