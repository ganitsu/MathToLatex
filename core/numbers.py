r"""
core/numbers.py
───────────────
Spanish number-word normalizer for 0–999.

When config.EXTENDED_NUMBERS is True, normalize_numbers() scans a word list
for Spanish compound number phrases and replaces each phrase with the equivalent
sequence of single-digit words (e.g. "ciento doce" → ["uno","uno","dos"]).

The translator's tokenizer already concatenates consecutive digit words into a
single NUMBER token (e.g. ["uno","uno","dos"] → NUMBER("112")), so this plugs
in transparently with zero changes to the tokenizer.

Old-style digit words (cero–nueve) remain valid at all times, stacking with
extended words: "novecientos noventa y nueve tres cuatro cuarenta" → "9993440".

"y" as numeric connector
─────────────────────────
"y" is also the math variable y.  The normalizer only consumes "y" as a
connector when it appears *inside* a pure-tens lookahead (treinta … noventa),
i.e. the pattern  [tens] "y" [unit].  In all other positions "y" is left
unchanged, so "equis y cinco" → x*y*5 as expected.
"""

from __future__ import annotations
import config as _cfg

# ── lookup tables ─────────────────────────────────────────────────────────────

_HUNDREDS: dict[str, int] = {
    "cien": 100, "ciento": 100,
    "doscientos": 200, "doscientas": 200,
    "trescientos": 300, "trescientas": 300,
    "cuatrocientos": 400, "cuatrocientas": 400,
    "quinientos": 500, "quinientas": 500,
    "seiscientos": 600, "seiscientas": 600,
    "setecientos": 700, "setecientas": 700,
    "ochocientos": 800, "ochocientas": 800,
    "novecientos": 900, "novecientas": 900,
}

# 10–19 and the single-word compound twenties (21–29)
_COMPOUND_TENS: dict[str, int] = {
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14,
    "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19,
    "veintiuno": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
}

# Pure tens that support optional "y" + unit suffix
_PURE_TENS: dict[str, int] = {
    "veinte": 20,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
}

# Units 1–9 (cero excluded: not used in compound Spanish numbers)
_UNITS: dict[str, int] = {
    "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
}

# All extended number words combined (used for fast membership test)
_ALL_EXTENDED: frozenset[str] = frozenset(
    list(_HUNDREDS) + list(_COMPOUND_TENS) + list(_PURE_TENS)
)

_DIGIT_TO_WORD: dict[str, str] = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro",
    "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve",
}


def _to_digit_words(n: int) -> list[str]:
    """Convert an integer to single-digit Spanish words: 112 → ['uno','uno','dos']."""
    return [_DIGIT_TO_WORD[d] for d in str(n)]


# ── parser ────────────────────────────────────────────────────────────────────

def _try_parse(words: list[str], start: int) -> tuple[int, int] | None:
    """
    Attempt to parse one Spanish number (0–999) beginning at words[start].
    Returns (value, end_index) or None if no number word is found at start.

    Consumes at most:  hundreds + (compound_tens | pure_tens ["y" unit] | unit)
    """
    i     = start
    n     = len(words)
    total = 0
    any_consumed = False

    # 1. Optional hundreds prefix
    if i < n and words[i] in _HUNDREDS:
        total       += _HUNDREDS[words[i]]
        i           += 1
        any_consumed = True

    if i >= n:
        return (total, i) if any_consumed else None

    w = words[i]

    # 2. Compound forms: diez–diecinueve, veintiuno–veintinueve
    if w in _COMPOUND_TENS:
        return (total + _COMPOUND_TENS[w], i + 1)

    # 3. Pure tens (veinte, treinta … noventa) + optional "y" unit
    if w in _PURE_TENS:
        total += _PURE_TENS[w]
        i     += 1
        if (i + 1 < n
                and words[i] == "y"
                and words[i + 1] in _UNITS):
            total += _UNITS[words[i + 1]]
            i     += 2
        return (total, i)

    # 4. Standalone units (1–9)
    if w in _UNITS:
        return (total + _UNITS[w], i + 1)

    return (total, i) if any_consumed else None


# ── public API ────────────────────────────────────────────────────────────────

def normalize_numbers(words: list[str]) -> list[str]:
    """
    Replace Spanish compound number phrases with digit-word sequences.

    Only activated when config.EXTENDED_NUMBERS is True.
    Single existing digit words (cero–nueve) that appear alone are left as-is
    (they are already valid in the tokenizer and converting them would be a
    no-op anyway).
    """
    if not _cfg.EXTENDED_NUMBERS:
        return words

    result: list[str] = []
    i = 0
    n = len(words)

    while i < n:
        parsed = _try_parse(words, i)
        if parsed is not None:
            value, end = parsed
            consumed   = words[i:end]

            # Only substitute if at least one "extended" word was consumed.
            # This leaves bare digit words (uno–nueve, cero) unchanged so that
            # EXTENDED_NUMBERS=True doesn't break the old digit-by-digit syntax.
            if any(w in _ALL_EXTENDED for w in consumed):
                result.extend(_to_digit_words(value))
                i = end
                continue

        result.append(words[i])
        i += 1

    return result


# ── vocabulary helpers (consumed by config) ───────────────────────────────────

def extended_vocabulary() -> list[str]:
    """Return all extended number words for inclusion in Vosk vocabulary."""
    return sorted(_ALL_EXTENDED)