"""
config.py
─────────
Central configuration for text_to_latex.
"""

import json
import sys
from datetime import datetime

# ── Debug ─────────────────────────────────────────────────────────────────────
DEBUG = True

def debug_log(tag: str, message: str) -> None:
    if not DEBUG:
        return
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{tag}] {message}", file=sys.stderr)

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_DEVICE = None
SAMPLE_RATE  = 16_000

# ── Network ───────────────────────────────────────────────────────────────────
RPI_HOST     = "192.168.5.10"
NETWORK_PORT = 9876

# ── Feature flags ─────────────────────────────────────────────────────────────
EXTENDED_NUMBERS = True
CONCAT_DIGITS    = False   # False → "uno tres" raises an error instead of 13

# ── Special control words ─────────────────────────────────────────────────────
END_WORD  = "fin"
DEL_WORD  = "bo"
REDO_WORD = "mi"

# ── Operator / symbol words ───────────────────────────────────────────────────
WORD_PLUS     = "mas"
WORD_PLUS_ALT = "suma"
WORD_MINUS    = "menos"
WORD_TIMES    = "por"
WORD_OVER     = "sobre"
WORD_POW      = "elevado"
WORD_POW_ALT  = "ala"
WORD_LPAREN   = "ba"
WORD_RPAREN   = "ca"
WORD_EQUALS   = "igual"
WORD_UNTIL    = "hasta"

# ── Variable / constant words ─────────────────────────────────────────────────
WORD_PI      = "pi"
WORD_E       = "euler"   # Euler's number e ≈ 2.718
WORD_X       = "equis"
WORD_Y       = "y"
WORD_Z       = "zeta"
WORD_N       = "ene"

# ── Function words ────────────────────────────────────────────────────────────
WORD_SIN     = "seno"
WORD_COS     = "coseno"
WORD_TAN     = "tangente"
WORD_LN      = "logaritmo"
WORD_INVERSE = "inverso"
WORD_SQRT    = "raiz"

# ── Calculus words ────────────────────────────────────────────────────────────
WORD_SUM      = "sumatoria"
WORD_INTEGRAL = "integral"
WORD_FROM     = "desde"    # lower-bound marker for sum and integral
WORD_UNTIL    = "hasta"    # upper-bound marker

# ── Decimal separator ─────────────────────────────────────────────────────────
WORD_DECIMAL = "coma"

# ── SI Prefixes ───────────────────────────────────────────────────────────────
# Mapping: spoken word → (LaTeX symbol, power of 10)
SI_PREFIXES: dict[str, tuple[str, int]] = {
    # "yotta": ("Y",      24),
    # "zetta": ("Z",      21),
    # "exa":   ("E",      18),
    # "peta":  ("P",      15),
    "tera":  ("T",      12),
    "giga":  ("G",       9),
    "mega":  ("M",       6),
    "kilo":  ("k",       3),
    "hecto": ("h",       2),
    "deca":  ("da",      1),
    "deci":  ("d",      -1),
    "centi": ("c",      -2),
    "mili":  ("m",      -3),
    "micro": (r"\mu",   -6),
    "nano":  ("n",      -9),
    "pico":  ("p",     -12),
    # "femto": ("f",     -15),
    # "ato":   ("a",     -18),
    # "zepto": ("z",     -21),
    # "yocto": ("y",     -24),
}

# ── Full vocabulary (used by Vosk) ────────────────────────────────────────────
def _build_vocabulary() -> list[str]:
    base = [
        # single digits
        "cero", "uno", "dos", "tres", "cuatro", "cinco",
        "seis", "siete", "ocho", "nueve",
        # arithmetic
        WORD_PLUS, WORD_PLUS_ALT, WORD_MINUS, WORD_TIMES, WORD_OVER,
        WORD_POW, WORD_POW_ALT,
        # grouping
        WORD_LPAREN, WORD_RPAREN,
        # relational
        WORD_EQUALS,
        # decimal
        WORD_DECIMAL,
        # variables / constants
        WORD_PI, WORD_E, WORD_X, WORD_Y, WORD_Z, WORD_N,
        # functions
        WORD_SIN, WORD_COS, WORD_TAN, WORD_LN, WORD_INVERSE, WORD_SQRT,
        # calculus (no more 'sub')
        WORD_SUM, WORD_INTEGRAL, WORD_FROM,
        # upper-bound separator
        WORD_UNTIL,
        # SI prefixes
        *SI_PREFIXES.keys(),
        # control  ← always last
        DEL_WORD, REDO_WORD, END_WORD,
    ]
    if EXTENDED_NUMBERS:
        from core.numbers import extended_vocabulary
        extra = [w for w in extended_vocabulary() if w not in base]
        base = base[:-3] + extra + base[-3:]
    return base

VOCABULARY_LIST: list[str] = _build_vocabulary()
VOCABULARY_JSON: str       = json.dumps(VOCABULARY_LIST)

# ── Recogniser ────────────────────────────────────────────────────────────────
MODEL_PATH = "vosk-model-small-es-0.42"

# ── Display ───────────────────────────────────────────────────────────────────
MAX_WORDS = 40