r"""
core/sympy_engine.py
─────────────────────
SymPy-powered analysis: solving and graphing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import sympy as sp
from sympy import lambdify, latex as sp_latex

from core.translator import (
    ASTNode, NumNode, PrefixNode, VarNode, BinOpNode, UnaryMinusNode,  # ← PrefixNode added
    ParenNode, EqNode, FuncNode, SumNode, IntegralNode, IndefiniteIntegralNode,
    TranslatorError,
)
from config import debug_log

_x  = sp.Symbol("x", real=True)
_y  = sp.Symbol("y", real=True)
_z  = sp.Symbol("z", real=True)
_n  = sp.Symbol("n", integer=True)
_GRAPH_VARS = {_x, _y, _z}

_BG   = "#141720"
_LINE = "#5b7fff"
_GRID = "#1e2130"
_TICK = "#4a5270"


@dataclass
class AnalysisResult:
    solution_latex: str | None    = None
    graph_fig:      Figure | None = None
    info:           str | None    = None


def ast_to_sympy(node: ASTNode) -> sp.Basic:
    if isinstance(node, NumNode):
        return sp.Float(node.value) if "." in node.value else sp.Integer(int(node.value))

    if isinstance(node, PrefixNode):
        return ast_to_sympy(node.number) * sp.Integer(10) ** node.exponent

    if isinstance(node, VarNode):
        return {
            "x": _x, "y": _y, "z": _z, "n": _n,
            r"\pi": sp.pi,
            "e":    sp.E,
        }.get(node.name, sp.Symbol(node.name))
    if isinstance(node, UnaryMinusNode):
        return -ast_to_sympy(node.operand)
    if isinstance(node, ParenNode):
        return ast_to_sympy(node.inner)
    if isinstance(node, EqNode):
        return sp.Eq(ast_to_sympy(node.left), ast_to_sympy(node.right))
    if isinstance(node, BinOpNode):
        L = ast_to_sympy(node.left)
        R = ast_to_sympy(node.right)
        if node.op == "+":               return L + R
        if node.op == "-":               return L - R
        if node.op in ("*","implicit*"): return L * R
        if node.op == "/":               return L / R
        if node.op == "^":               return L ** R
        raise TranslatorError(f"Operador desconocido: '{node.op}'")
    if isinstance(node, FuncNode):
        arg = ast_to_sympy(node.arg)
        if node.func == r"\sqrt":
            return sp.sqrt(arg)
        _FUNCS = {
            r"\sin": (sp.sin,  sp.asin),
            r"\cos": (sp.cos,  sp.acos),
            r"\tan": (sp.tan,  sp.atan),
            r"\ln":  (sp.log,  sp.exp),
        }
        if node.func not in _FUNCS:
            raise TranslatorError(f"Función desconocida: '{node.func}'")
        normal, inverse = _FUNCS[node.func]
        return inverse(arg) if node.inverse else normal(arg)
    if isinstance(node, SumNode):
        return sp.Sum(ast_to_sympy(node.body), (_n, ast_to_sympy(node.lower), ast_to_sympy(node.upper)))
    if isinstance(node, IntegralNode):
        return sp.Integral(ast_to_sympy(node.body), (_x, ast_to_sympy(node.lower), ast_to_sympy(node.upper)))
    if isinstance(node, IndefiniteIntegralNode):
        return sp.Integral(ast_to_sympy(node.body), _x)
    raise TranslatorError(f"Nodo AST desconocido en sympy_engine: {type(node)}")


def _add_approx(latex_str: str, val: sp.Basic) -> str:
    if getattr(val, "is_integer", False):
        return latex_str
    try:
        f = float(sp.N(val, 8))
        return latex_str + rf" \approx {round(f, 4)}"
    except Exception:
        return latex_str


def _solve(sym_eq: sp.Eq) -> str | None:
    free = sym_eq.free_symbols & _GRAPH_VARS
    if len(free) != 1:
        return None
    var = next(iter(free))
    try:
        solutions = sp.solve(sym_eq, var)
    except Exception as exc:
        debug_log("SYMPY", f"solve failed: {exc}")
        return None
    if not solutions:
        return None
    var_str   = sp_latex(var)
    sol_parts = []
    for s in solutions:
        s = sp.simplify(s)
        part = f"{var_str} = {sp_latex(s)}"
        part = _add_approx(part, s)
        sol_parts.append(part)
    return ",\\ ".join(sol_parts)


# ── Pure number analysis ───────────────────────────────────────────────────────

def _analyse_pure_number(sym_expr: sp.Basic) -> str:
    """
    For expressions with no free variables, return decimal + fraction if applicable.
    e.g.  3/4  →  \frac{3}{4} \approx 0.75
          22/7  →  \frac{22}{7} \approx 3.1429
          4     →  4
    """
    simplified = sp.nsimplify(sym_expr, rational=True, tolerance=1e-9)
    latex_exact = sp_latex(simplified)

    # Add decimal approx if not already an integer
    if not simplified.is_Integer:
        try:
            f = float(sp.N(simplified, 10))
            latex_exact += rf" \approx {round(f, 6)}"
        except Exception:
            pass

    return latex_exact


# ── Graphing ───────────────────────────────────────────────────────────────────

def _style_ax(ax) -> None:
    ax.set_facecolor(_BG)
    for spine in getattr(ax, "spines", {}).values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_TICK, labelsize=8)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)
    ax.axhline(0, color=_TICK, linewidth=0.8)
    ax.axvline(0, color=_TICK, linewidth=0.8)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(_TICK)


def _plot_2d(expr: sp.Basic) -> Figure:
    f  = lambdify(_x, expr, modules=["numpy"])
    xs = np.linspace(-10, 10, 1000)
    ys = np.asarray(f(xs), dtype=float)
    y_fin = ys[np.isfinite(ys)]
    if y_fin.size == 0:
        raise ValueError("La expresión no produce valores finitos en [-10, 10]")
    clip = max(abs(y_fin).max() * 1.2, 1.0)
    ys   = np.clip(ys, -clip, clip)
    ys[~np.isfinite(ys)] = np.nan

    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=_BG)
    _style_ax(ax)
    ax.plot(xs, ys, color=_LINE, linewidth=2.2, solid_capstyle="round")
    ax.set_aspect("equal", adjustable="datalim")   # ← equal axis scale
    ax.set_xlabel("x", color=_TICK, fontsize=9)
    ax.set_ylabel("y", color=_TICK, fontsize=9, rotation=0, labelpad=10)
    fig.tight_layout(pad=0.6)
    return fig


def _plot_3d_surfaces(z_expressions: list, var_pair=(_x, _y)) -> Figure:
    cmaps  = ["cool", "autumn"]
    v1, v2 = var_pair
    t      = np.linspace(-5, 5, 55)
    A, B   = np.meshgrid(t, t)

    fig = plt.figure(figsize=(7.2, 4.5), facecolor=_BG)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(_BG)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(_GRID)

    plotted = False
    for i, z_expr in enumerate(z_expressions[:2]):
        try:
            f = lambdify((v1, v2), z_expr, modules=["numpy"])
            Z = np.asarray(f(A, B), dtype=float)
            Z[~np.isfinite(Z)] = np.nan
            ax.plot_surface(A, B, Z, cmap=cmaps[i], alpha=0.82, linewidth=0, antialiased=True)
            plotted = True
        except Exception as exc:
            debug_log("SYMPY", f"surface {i} failed: {exc}")

    if not plotted:
        plt.close(fig)
        raise ValueError("No se pudo graficar ninguna de las superficies")

    # equal scale on all three axes
    ax.set_box_aspect([1, 1, 1])   # ← equal axis scale for 3D
    ax.tick_params(colors=_TICK, labelsize=7)
    ax.set_xlabel(v1.name, color=_TICK, fontsize=9)
    ax.set_ylabel(v2.name, color=_TICK, fontsize=9)
    ax.set_zlabel("z",     color=_TICK, fontsize=9)
    fig.tight_layout(pad=0.4)
    return fig

def _plot_2d_multi(exprs: list) -> Figure:
    """Plot multiple y=f(x) branches on the same axes (e.g. both halves of a circle)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=_BG)
    _style_ax(ax)
    plotted = False
    for expr in exprs:
        try:
            f  = lambdify(_x, expr, modules=["numpy"])
            xs = np.linspace(-10, 10, 1000)
            ys = np.asarray(f(xs), dtype=float)
            y_fin = ys[np.isfinite(ys)]
            if y_fin.size == 0:
                continue
            clip = max(abs(y_fin).max() * 1.2, 1.0)
            ys   = np.clip(ys, -clip, clip)
            ys[~np.isfinite(ys)] = np.nan
            ax.plot(xs, ys, color=_LINE, linewidth=2.2, solid_capstyle="round")
            plotted = True
        except Exception as exc:
            debug_log("SYMPY", f"multi-branch plot failed: {exc}")
    if not plotted:
        plt.close(fig)
        raise ValueError("No se pudo graficar ninguna rama")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x", color=_TICK, fontsize=9)
    ax.set_ylabel("y", color=_TICK, fontsize=9, rotation=0, labelpad=10)
    fig.tight_layout(pad=0.6)
    return fig


def _try_graph(sym_expr: sp.Basic) -> Figure | None:
    if isinstance(sym_expr, sp.Eq):
        lhs, rhs = sym_expr.lhs, sym_expr.rhs
        free_3 = sym_expr.free_symbols & _GRAPH_VARS
        if free_3 == {_x, _y, _z}:
            try:
                z_sols = sp.solve(sym_expr, _z)
                if z_sols:
                    return _plot_3d_surfaces(z_sols, (_x, _y))
            except Exception as exc:
                debug_log("SYMPY", f"3D implicit solve failed: {exc}")
            return None
        if lhs == _z:
            expr = rhs
        elif lhs == _y:
            expr = rhs
        elif lhs == _x:
            expr = rhs
        else:
            free_2 = sym_expr.free_symbols & {_y, _z}
            if free_2:
                solve_for = next(iter(free_2))
                try:
                    sols = sp.solve(sym_expr, solve_for)
                    if sols:
                        # Multiple solutions (e.g. x²+y²=1 → y=±√(1-x²)) → plot all branches
                        if len(sols) > 1 and solve_for == _y:
                            return _plot_2d_multi(sols)
                        expr = sols[0]
                    else:
                        expr = lhs - rhs
                except Exception:
                    expr = lhs - rhs
    else:
        expr = sym_expr

    if expr.has(sp.Sum) or expr.has(sp.Integral):
        return None
    free = expr.free_symbols & _GRAPH_VARS
    if not free:
        return None
    if free == {_x}:
        return _plot_2d(expr)
    if len(free) == 2:
        vars_sorted = sorted(free, key=lambda s: s.name)
        return _plot_3d_surfaces([expr], tuple(vars_sorted))
    return None


def analyse(ast: ASTNode) -> AnalysisResult:
    result = AnalysisResult()

    try:
        sym_expr = ast_to_sympy(ast)
        debug_log("SYMPY", f"converted: {sym_expr}")
    except Exception as exc:
        debug_log("SYMPY", f"ast_to_sympy failed: {exc}")
        result.info = f"SymPy: {exc}"
        return result

    # ── Pure number (no free variables) ───────────────────────────────────────
    if not sym_expr.free_symbols and not isinstance(sym_expr, sp.Eq):
        try:
            result.solution_latex = _analyse_pure_number(sym_expr)
            debug_log("SYMPY", f"pure number: {result.solution_latex}")
        except Exception as exc:
            debug_log("SYMPY", f"pure number failed: {exc}")
        return result

    # ── Sums and integrals ────────────────────────────────────────────────────
    if isinstance(sym_expr, (sp.Integral, sp.Sum)):
        is_sum      = isinstance(sym_expr, sp.Sum)
        is_definite = is_sum or len(sym_expr.limits[0]) == 3
        label       = "sumatoria" if is_sum else "integral"
        try:
            evaluated  = sym_expr.doit()
            simplified = sp.simplify(evaluated)
            still_uneval = simplified.has(sp.Integral) or simplified.has(sp.Sum)
            if not still_uneval:
                sol_str = sp_latex(simplified)
                if not is_definite:
                    sol_str += r" + C"
                sol_str = _add_approx(sol_str, simplified)
                result.solution_latex = sol_str
                debug_log("SYMPY", f"{label} result: {sol_str}")
            else:
                result.info = f"La {label} no tiene solución analítica conocida"
        except Exception as exc:
            debug_log("SYMPY", f"{label} doit failed: {exc}")
            result.info = f"No se pudo evaluar la {label}: {exc}"
        return result

    # ── Equations ─────────────────────────────────────────────────────────────
    if isinstance(sym_expr, sp.Eq):
        free_cont = sym_expr.free_symbols & _GRAPH_VARS
        if len(free_cont) == 1:
            try:
                sol = _solve(sym_expr)
                if sol:
                    result.solution_latex = sol
                else:
                    result.info = "No se encontró solución simbólica"
            except Exception as exc:
                result.info = f"Error al resolver: {exc}"

    # ── Graph ─────────────────────────────────────────────────────────────────
    try:
        fig = _try_graph(sym_expr)
        if fig is not None:
            result.graph_fig = fig
    except Exception as exc:
        note = f"No se puede graficar: {exc}"
        result.info = f"{result.info}  •  {note}" if result.info else note

    return result