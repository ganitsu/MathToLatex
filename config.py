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

# ── Audio / model configuration ─────────────────────────────────────────────────
AUDIO_DEVICE = "sysdefault:CARD=Loopback"   # None = system default (for Windows)

# ── Feature flags ─────────────────────────────────────────────────────────────
# When True, compound Spanish number words (catorce, doscientos, etc.) are
# recognized and converted before tokenisation.  The old digit-by-digit system
# (uno uno dos = 112) still works on top of this.
EXTENDED_NUMBERS = True


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

# Upper-bound separator for sumatoria / integral
# "hasta" is the preferred spoken form; "elevado ala" also works
WORD_UNTIL    = "hasta"

# ── Variable / constant words ─────────────────────────────────────────────────
WORD_PI      = "pi"
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
WORD_SQRT    = "raiz"         # √  —  raiz <arg>  →  \sqrt{arg}

# ── Calculus words ────────────────────────────────────────────────────────────
WORD_SUM      = "sumatoria"
WORD_INTEGRAL = "integral"
WORD_SUB      = "sub"         # lower-bound marker for sumatoria
WORD_FROM     = "desde"       # lower-bound marker for integral (definite)
                              # omitting "desde" makes the integral indefinite

# ── Decimal separator ─────────────────────────────────────────────────────────
WORD_DECIMAL = "coma"

# ── Full vocabulary (used by Vosk) ────────────────────────────────────────────
# Import extended number words when EXTENDED_NUMBERS is enabled
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
        WORD_PI, WORD_X, WORD_Y, WORD_Z, WORD_N,
        # functions
        WORD_SIN, WORD_COS, WORD_TAN, WORD_LN, WORD_INVERSE, WORD_SQRT,
        # calculus
        WORD_SUM, WORD_INTEGRAL, WORD_SUB, WORD_FROM,
        # upper-bound separator
        WORD_UNTIL,
        # control  ← always last
        DEL_WORD, REDO_WORD, END_WORD,
    ]
    if EXTENDED_NUMBERS:
        from core.numbers import extended_vocabulary
        extra = [w for w in extended_vocabulary() if w not in base]
        # Insert before control words
        base = base[:-3] + extra + base[-3:]
    return base

VOCABULARY_LIST: list[str] = _build_vocabulary()
VOCABULARY_JSON: str       = json.dumps(VOCABULARY_LIST)

# ── Recogniser ────────────────────────────────────────────────────────────────
MODEL_PATH  = "vosk-model-small-es-0.42"
SAMPLE_RATE = 16_000

# ── Display ───────────────────────────────────────────────────────────────────
MAX_WORDS = 40