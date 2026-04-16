r"""
core/translator.py
──────────────────
Pipeline: word list -> TranslationResult

0. normalize_numbers(words) - Spanish compound numbers (when EXTENDED_NUMBERS=True)
1. apply_edits(words)       - bo/mi undo-redo
2. _balance_parens(words)   - auto-close unclosed ba
3. parse_tokens(words)      - words -> list[Token]
4. _Parser(tokens).parse()  - tokens -> ASTNode
5. render_latex(node)       - ASTNode -> LaTeX string

Public surface:
    translate(words) -> TranslationResult  (.ok, .latex, .error, .ast)

Summation:  sumatoria sub ene igual <lo> (hasta|elevado ala) <hi> <body>
Integral:   integral  sub           <lo> (hasta|elevado ala) <hi> <body>

LaTeX note: all backslash-command symbols (e.g. \pi) are wrapped in {} so
that juxtaposition is always safe: {\pi}x, {\pi}{\pi}, 2{\pi}.  Nobody
cares how the raw LaTeX looks as long as it renders correctly.
"""

from __future__ import annotations

import config as _cfg
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TranslationResult:
    latex:    str             = ""
    error:    str             = ""
    ast:      "ASTNode|None"  = None

    @property
    def ok(self) -> bool:
        return self.error == ""


# ─────────────────────────────────────────────────────────────────────────────
# Token types
# ─────────────────────────────────────────────────────────────────────────────

class TT(Enum):
    NUMBER   = auto()
    PI       = auto()
    X        = auto()
    Y        = auto()
    Z        = auto()
    N        = auto()
    PLUS     = auto()
    MINUS    = auto()
    TIMES    = auto()
    OVER     = auto()
    POW      = auto()
    LPAREN   = auto()
    RPAREN   = auto()
    EQUALS   = auto()
    SIN      = auto()
    COS      = auto()
    TAN      = auto()
    LN       = auto()
    INVERSE  = auto()
    SUM      = auto()
    INTEGRAL = auto()
    SUB      = auto()   # lower-bound marker for sumatoria
    FROM     = auto()   # 'desde' — lower-bound marker for integral
    UNTIL    = auto()   # 'hasta' — upper-bound marker for sum/integral
    SQRT     = auto()   # 'raiz'  — square root
    EOF      = auto()


@dataclass
class Token:
    kind:       TT
    value:      str   = ""
    word_index: int   = -1
    # Future: populate from Vosk word-level timing
    start_ms:   float | None = None
    end_ms:     float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# AST nodes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NumNode:
    value: str

@dataclass
class VarNode:
    name: str     # LaTeX: "x", "y", "z", "n", r"\pi"

@dataclass
class BinOpNode:
    op:    str
    left:  "ASTNode"
    right: "ASTNode"

@dataclass
class UnaryMinusNode:
    operand: "ASTNode"

@dataclass
class ParenNode:
    inner: "ASTNode"

@dataclass
class EqNode:
    left:  "ASTNode"
    right: "ASTNode"

@dataclass
class FuncNode:
    func:    str
    arg:     "ASTNode"
    inverse: bool = False

@dataclass
class SumNode:
    var:   str
    lower: "ASTNode"
    upper: "ASTNode"
    body:  "ASTNode"

@dataclass
class IntegralNode:
    lower: "ASTNode"
    upper: "ASTNode"
    body:  "ASTNode"

@dataclass
class IndefiniteIntegralNode:
    """Indefinite integral: ∫ body dx  (no bounds)"""
    body: "ASTNode"

ASTNode = Union[
    NumNode, VarNode, BinOpNode, UnaryMinusNode,
    ParenNode, EqNode, FuncNode, SumNode, IntegralNode, IndefiniteIntegralNode,
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 – bo / mi
# ─────────────────────────────────────────────────────────────────────────────

def _find_del_point(words: list[str]) -> int:
    barriers = {_cfg.WORD_PLUS, _cfg.WORD_PLUS_ALT, _cfg.WORD_MINUS, _cfg.WORD_EQUALS}
    for i in range(len(words) - 1, -1, -1):
        if words[i] in barriers:
            return i
    return 0

def apply_edits(words: list[str]) -> list[str]:
    committed: list[str]       = []
    undo_stack: list[list[str]] = []
    for w in words:
        if w == _cfg.DEL_WORD:
            cut = _find_del_point(committed)
            if committed[cut:] or cut < len(committed):
                undo_stack.append(committed[cut:])
                committed = committed[:cut]
        elif w == _cfg.REDO_WORD:
            if undo_stack:
                committed.extend(undo_stack.pop())
        else:
            committed.append(w)
            undo_stack.clear()
    return committed


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – tokeniser
# ─────────────────────────────────────────────────────────────────────────────

_DIGIT_WORDS = {
    "cero": "0", "uno": "1", "dos": "2", "tres": "3", "cuatro": "4",
    "cinco": "5", "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
}


class TranslatorError(Exception):
    pass


def parse_tokens(words: list[str]) -> list[Token]:
    tokens: list[Token] = []
    i = 0

    def _add(tok: Token) -> None:
        # Collapse consecutive POW (elevado + ala → single ^)
        if tok.kind == TT.POW and tokens and tokens[-1].kind == TT.POW:
            return
        tokens.append(tok)

    while i < len(words):
        w = words[i]

        # ── digit sequence ────────────────────────────────────────────────────
        if w in _DIGIT_WORDS:
            digits       = [_DIGIT_WORDS[w]]
            start        = i
            decimal_seen = False
            i += 1
            while i < len(words):
                nw = words[i]
                if nw in _DIGIT_WORDS:
                    digits.append(_DIGIT_WORDS[nw]); i += 1
                elif nw == _cfg.WORD_DECIMAL and not decimal_seen:
                    digits.append("."); decimal_seen = True; i += 1
                else:
                    break
            tokens.append(Token(TT.NUMBER, "".join(digits), start))
            continue

        # ── keyword map ───────────────────────────────────────────────────────
        kw: dict[str, Token | None] = {
            _cfg.WORD_PLUS:      Token(TT.PLUS,     "+",   i),
            _cfg.WORD_PLUS_ALT:  Token(TT.PLUS,     "+",   i),
            _cfg.WORD_MINUS:     Token(TT.MINUS,    "-",   i),
            _cfg.WORD_TIMES:     Token(TT.TIMES,    "*",   i),
            _cfg.WORD_OVER:      Token(TT.OVER,     "/",   i),
            _cfg.WORD_POW:       Token(TT.POW,      "^",   i),
            _cfg.WORD_POW_ALT:   Token(TT.POW,      "^",   i),
            _cfg.WORD_UNTIL:     Token(TT.UNTIL,    "hasta", i),
            _cfg.WORD_LPAREN:    Token(TT.LPAREN,   "(",   i),
            _cfg.WORD_RPAREN:    Token(TT.RPAREN,   ")",   i),
            _cfg.WORD_EQUALS:    Token(TT.EQUALS,   "=",   i),
            _cfg.WORD_PI:        Token(TT.PI,       "pi",  i),
            _cfg.WORD_X:         Token(TT.X,        "x",   i),
            _cfg.WORD_Y:         Token(TT.Y,        "y",   i),
            _cfg.WORD_Z:         Token(TT.Z,        "z",   i),
            _cfg.WORD_N:         Token(TT.N,        "n",   i),
            _cfg.WORD_SIN:       Token(TT.SIN,      "sin", i),
            _cfg.WORD_COS:       Token(TT.COS,      "cos", i),
            _cfg.WORD_TAN:       Token(TT.TAN,      "tan", i),
            _cfg.WORD_LN:        Token(TT.LN,       "ln",  i),
            _cfg.WORD_INVERSE:   Token(TT.INVERSE,  "inv", i),
            _cfg.WORD_SUM:       Token(TT.SUM,      "sum", i),
            _cfg.WORD_INTEGRAL:  Token(TT.INTEGRAL, "int", i),
            _cfg.WORD_SUB:       Token(TT.SUB,      "_",   i),
            _cfg.WORD_FROM:      Token(TT.FROM,     "desde", i),
            _cfg.WORD_SQRT:      Token(TT.SQRT,     "raiz",  i),
            # silently consumed
            _cfg.END_WORD:       None,
            _cfg.DEL_WORD:       None,
            _cfg.REDO_WORD:      None,
            _cfg.WORD_DECIMAL:   None,
        }

        if w in kw:
            tok = kw[w]
            if tok is not None:
                _add(tok)
            i += 1
            continue

        raise TranslatorError(f"Palabra desconocida: '{w}'")

    tokens.append(Token(TT.EOF, "", len(words)))
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – parser
# ─────────────────────────────────────────────────────────────────────────────

_PRIMARY_STARTERS = {
    TT.NUMBER, TT.PI, TT.X, TT.Y, TT.Z, TT.N,
    TT.LPAREN, TT.SIN, TT.COS, TT.TAN, TT.LN,
    TT.SUM, TT.INTEGRAL, TT.SQRT,
}


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._t   = tokens
        self._pos = 0

    def _peek(self)      -> Token: return self._t[self._pos]
    def _advance(self)   -> Token:
        t = self._t[self._pos]; self._pos += 1; return t
    def _match(self, *k) -> bool:  return self._peek().kind in k

    def _expect(self, k: TT, msg: str) -> Token:
        if self._peek().kind != k:
            raise TranslatorError(msg)
        return self._advance()

    def _expect_upper_sep(self, ctx: str) -> None:
        """Accept either 'elevado ala' (POW) or 'hasta' (UNTIL) as upper-bound marker."""
        if self._match(TT.POW, TT.UNTIL):
            self._advance()
        else:
            raise TranslatorError(
                f"Falta 'hasta' o 'elevado ala' para el límite superior de {ctx}"
            )

    def parse(self) -> ASTNode:
        node = self._equality()
        if self._peek().kind != TT.EOF:
            raise TranslatorError(
                f"Token inesperado '{self._peek().value}' — "
                "revisá que los operadores y operandos se alternen"
            )
        return node

    def _equality(self) -> ASTNode:
        node = self._additive()
        while self._match(TT.EQUALS):
            self._advance()
            node = EqNode(node, self._additive())
        return node

    def _additive(self) -> ASTNode:
        node = self._term()
        while self._match(TT.PLUS, TT.MINUS):
            tok = self._advance()
            node = BinOpNode("+" if tok.kind == TT.PLUS else "-", node, self._term())
        return node

    def _term(self) -> ASTNode:
        node = self._impl_mul()
        while self._match(TT.TIMES, TT.OVER):
            tok = self._advance()
            node = BinOpNode("*" if tok.kind == TT.TIMES else "/", node, self._impl_mul())
        return node

    def _impl_mul(self) -> ASTNode:
        node = self._power()
        while self._peek().kind in _PRIMARY_STARTERS:
            node = BinOpNode("implicit*", node, self._power())
        return node

    def _power(self) -> ASTNode:
        base = self._unary()
        if self._match(TT.POW):
            self._advance()
            return BinOpNode("^", base, self._power())   # right-associative
        return base

    def _unary(self) -> ASTNode:
        if self._match(TT.MINUS):
            self._advance()
            return UnaryMinusNode(self._unary())
        if self._match(TT.INVERSE):
            self._advance()
            if not self._match(TT.SIN, TT.COS, TT.TAN, TT.LN, TT.SQRT):
                raise TranslatorError(
                    "'inverso' al inicio debe ir seguido de una función"
                )
            return self._primary_func(inverse=True)
        node = self._primary()
        if self._match(TT.INVERSE):
            self._advance()
            if isinstance(node, FuncNode):
                node.inverse = True
            else:
                node = BinOpNode("^", node, UnaryMinusNode(NumNode("1")))
        return node

    def _primary_func(self, inverse: bool = False) -> ASTNode:
        if self._peek().kind == TT.SQRT:
            self._advance()
            return FuncNode(r"\sqrt", self._primary())
        func_map = {TT.SIN: r"\sin", TT.COS: r"\cos", TT.TAN: r"\tan", TT.LN: r"\ln"}
        tok = self._advance()
        func = func_map[tok.kind]
        if self._match(TT.INVERSE):
            self._advance(); inverse = True
        return FuncNode(func, self._primary(), inverse=inverse)

    def _primary(self) -> ASTNode:
        tok = self._peek()

        if tok.kind == TT.NUMBER:   self._advance(); return NumNode(tok.value)
        if tok.kind == TT.PI:       self._advance(); return VarNode(r"\pi")
        if tok.kind == TT.X:        self._advance(); return VarNode("x")
        if tok.kind == TT.Y:        self._advance(); return VarNode("y")
        if tok.kind == TT.Z:        self._advance(); return VarNode("z")
        if tok.kind == TT.N:        self._advance(); return VarNode("n")

        if tok.kind in (TT.SIN, TT.COS, TT.TAN, TT.LN, TT.SQRT):
            return self._primary_func()

        # ── sumatoria sub ene igual <lo> (hasta|POW) <hi> <body> ─────────────
        if tok.kind == TT.SUM:
            self._advance()
            self._expect(TT.SUB,    "Falta 'sub' después de 'sumatoria'")
            self._expect(TT.N,      "Falta 'ene' después de 'sub' en sumatoria")
            self._expect(TT.EQUALS, "Falta 'igual' para el límite inferior")
            lower = self._primary()
            self._expect_upper_sep("sumatoria")
            upper = self._primary()
            body  = self._primary()
            return SumNode("n", lower, upper, body)

        # ── integral [desde <lo> hasta <hi>] <body> ────────────────────────────
        # 'desde' present → definite integral; absent → indefinite
        if tok.kind == TT.INTEGRAL:
            self._advance()
            if self._match(TT.FROM):
                self._advance()
                lower = self._primary()
                self._expect_upper_sep("integral")
                upper = self._primary()
                body  = self._primary()
                return IntegralNode(lower, upper, body)
            else:
                body = self._primary()
                return IndefiniteIntegralNode(body)

        if tok.kind == TT.LPAREN:
            self._advance()
            inner = self._equality()
            self._expect(TT.RPAREN, "Falta cierre de paréntesis 'ca'")
            return ParenNode(inner)

        if tok.kind == TT.RPAREN:
            raise TranslatorError("'ca' sin 'ba' correspondiente")
        if tok.kind == TT.SUB:
            raise TranslatorError("'sub' solo puede usarse después de 'sumatoria'")
        if tok.kind == TT.FROM:
            raise TranslatorError("'desde' solo puede usarse después de 'integral'")
        if tok.kind == TT.UNTIL:
            raise TranslatorError("'hasta' solo puede usarse dentro de sumatoria o integral definida")
        if tok.kind == TT.EOF:
            raise TranslatorError("Expresión incompleta — falta un operando")
        if tok.kind == TT.INVERSE:
            raise TranslatorError("'inverso' debe seguir a una expresión o preceder a una función")
        raise TranslatorError(f"Token inesperado: '{tok.value}'")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – LaTeX renderer
# ─────────────────────────────────────────────────────────────────────────────

def _mul_style(left: ASTNode, right: ASTNode) -> str:
    """
    Decide multiplication rendering style.
    All backslash symbols are wrapped in braces so juxtaposition is always safe.
    Rules:
      number * number   -> cdot  (avoids "35" being read as thirty-five)
      func   * anything -> cdot  (sin(x)*2 looks better with explicit dot)
      everything else   -> juxt  (2x, xy, {pi}x, 2(x+1), etc.)
    """
    if isinstance(left, NumNode) and isinstance(right, NumNode):
        return "cdot"
    # \sqrt is written without cdot: 2\sqrt{x}, x\sqrt{y}
    if isinstance(right, FuncNode) and right.func == r"\sqrt":
        return "juxt"
    if isinstance(left, FuncNode) or isinstance(right, FuncNode):
        return "cdot"
    return "juxt"


def render_latex(node: ASTNode) -> str:
    if isinstance(node, NumNode):
        return node.value

    if isinstance(node, VarNode):
        # Wrap backslash commands in {} so juxtaposition is always unambiguous.
        # {\pi}x renders correctly; {\pi}{\pi} too.  Plain variables (x,y,z,n)
        # are single chars and need no wrapping.
        n = node.name
        return f"{{{n}}}" if n.startswith("\\") else n

    if isinstance(node, UnaryMinusNode):
        return f"-{render_latex(node.operand)}"

    if isinstance(node, ParenNode):
        return rf"\left({render_latex(node.inner)}\right)"

    if isinstance(node, EqNode):
        return f"{render_latex(node.left)} = {render_latex(node.right)}"

    if isinstance(node, FuncNode):
        # sqrt renders as \sqrt{arg} — no \left\right wrapping
        if node.func == r"\sqrt":
            return rf"\sqrt{{{render_latex(node.arg)}}}"
        inv = "^{-1}" if node.inverse else ""
        arg = render_latex(node.arg)
        if isinstance(node.arg, ParenNode):
            return f"{node.func}{inv}{arg}"
        return rf"{node.func}{inv}\left({arg}\right)"

    if isinstance(node, SumNode):
        lo   = render_latex(node.lower)
        hi   = render_latex(node.upper)
        body = render_latex(node.body)
        return rf"\sum_{{n={lo}}}^{{{hi}}} {body}"

    if isinstance(node, IntegralNode):
        lo   = render_latex(node.lower)
        hi   = render_latex(node.upper)
        body = render_latex(node.body)
        return rf"\int_{{{lo}}}^{{{hi}}} {body} \, dx"

    if isinstance(node, IndefiniteIntegralNode):
        body = render_latex(node.body)
        return rf"\int {body} \, dx"

    if isinstance(node, BinOpNode):
        L = render_latex(node.left)
        R = render_latex(node.right)

        if node.op == "+": return f"{L} + {R}"
        if node.op == "-": return f"{L} - {R}"

        if node.op in ("*", "implicit*"):
            style = _mul_style(node.left, node.right)
            if style == "cdot": return rf"{L} \cdot {R}"
            return f"{L}{R}"   # juxt — {} wrapping already ensures correctness

        if node.op == "/":
            return rf"\frac{{{L}}}{{{R}}}"

        if node.op == "^":
            base_s = L if isinstance(node.left, ParenNode) or len(L) == 1 else f"{{{L}}}"
            return f"{base_s}^{{{R}}}"

    raise TranslatorError(f"Nodo AST desconocido: {type(node)}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _balance_parens(words: list[str]) -> list[str]:
    depth = sum(
        1 if w == _cfg.WORD_LPAREN else -1 if w == _cfg.WORD_RPAREN else 0
        for w in words
    )
    return words + [_cfg.WORD_RPAREN] * max(0, depth)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def translate(words: list[str]) -> TranslationResult:
    from core.numbers import normalize_numbers

    clean = [w for w in words if w != _cfg.END_WORD]
    clean = normalize_numbers(clean)
    clean = apply_edits(clean)
    if not clean:
        return TranslationResult(error="No hay palabras para traducir.")
    clean = _balance_parens(clean)
    try:
        tokens = parse_tokens(clean)
        ast    = _Parser(tokens).parse()
        latex  = render_latex(ast)
        return TranslationResult(latex=latex, ast=ast)
    except TranslatorError as exc:
        return TranslationResult(error=str(exc))
    except Exception as exc:
        return TranslationResult(error=f"Error interno: {exc}")