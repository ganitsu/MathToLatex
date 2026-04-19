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
AUDIO_DEVICE = None   # None = system default (Windows mic)
SAMPLE_RATE  = 16_000

# ── Network ───────────────────────────────────────────────────────────────────
RPI_HOST     = "192.168.5.10"   # hostname or static IP of the RPi
NETWORK_PORT = 9876

# ── Feature flags ─────────────────────────────────────────────────────────────
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
WORD_SQRT    = "raiz"

# ── Calculus words ────────────────────────────────────────────────────────────
WORD_SUM      = "sumatoria"
WORD_INTEGRAL = "integral"
WORD_SUB      = "sub"
WORD_FROM     = "desde"

# ── Decimal separator ─────────────────────────────────────────────────────────
WORD_DECIMAL = "coma"

# ── Full vocabulary (used by Vosk) ────────────────────────────────────────────
def _build_vocabulary() -> list[str]:
    base = [
        "cero", "uno", "dos", "tres", "cuatro", "cinco",
        "seis", "siete", "ocho", "nueve",
        WORD_PLUS, WORD_PLUS_ALT, WORD_MINUS, WORD_TIMES, WORD_OVER,
        WORD_POW, WORD_POW_ALT,
        WORD_LPAREN, WORD_RPAREN,
        WORD_EQUALS,
        WORD_DECIMAL,
        WORD_PI, WORD_X, WORD_Y, WORD_Z, WORD_N,
        WORD_SIN, WORD_COS, WORD_TAN, WORD_LN, WORD_INVERSE, WORD_SQRT,
        WORD_SUM, WORD_INTEGRAL, WORD_SUB, WORD_FROM,
        WORD_UNTIL,
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