"""
ui/app.py
─────────
Main application window.
"""

from __future__ import annotations

import io
import queue
import tkinter as tk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageTk

import config
from config import debug_log
from core.translator import translate, TranslationResult
from core.sympy_engine import analyse, AnalysisResult

# ── palette ───────────────────────────────────────────────────────────────────
BG          = "#0d0f14"
SURFACE     = "#141720"
SURFACE_ALT = "#0f1118"
BORDER      = "#1e2130"
FG_MAIN     = "#e8eaf2"
FG_DIM      = "#4a5270"
FG_END      = "#3ddc84"
FG_LATEX    = "#a8c4ff"
FG_SOL      = "#3ddc84"
FG_WARN     = "#f0a030"
FG_ERROR    = "#ff4f5e"

FONT_HEADER = ("JetBrains Mono", 13, "bold")
FONT_SMALL  = ("JetBrains Mono", 11)
FONT_TINY   = ("JetBrains Mono", 9)

_LATEX_H = 150
_SOL_H   = 80    # taller now — renders LaTeX image
_GRAPH_H = 280


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_photo(fig: Figure) -> ImageTk.PhotoImage:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return ImageTk.PhotoImage(Image.open(buf))


def _render_latex_image(
    latex: str,
    width_px: int,
    height_px: int,
    color: str   = FG_LATEX,
    bg:    str   = SURFACE,
    fontsize: int = 36,
) -> ImageTk.PhotoImage:
    dpi = 110
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), facecolor=bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.5, 0.5, f"${latex}$",
            ha="center", va="center",
            fontsize=fontsize, color=color,
            transform=ax.transAxes, clip_on=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=dpi, facecolor=bg, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return ImageTk.PhotoImage(Image.open(buf))


def _render_text_image(
    message: str,
    width_px: int,
    height_px: int,
    color: str = FG_WARN,
    bg:    str = SURFACE_ALT,
) -> ImageTk.PhotoImage:
    """Render a plain-text string as an image (for info/error messages)."""
    dpi = 110
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), facecolor=bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.5, 0.5, message,
            ha="center", va="center",
            fontsize=11, color=color,
            transform=ax.transAxes, wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=dpi, facecolor=bg, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return ImageTk.PhotoImage(Image.open(buf))


def _render_error_image(
    message: str,
    width_px: int,
    height_px: int,
    bg: str = SURFACE,
) -> ImageTk.PhotoImage:
    return _render_text_image(f"⚠  {message}", width_px, height_px,
                              color=FG_ERROR, bg=bg)


def _graph_fig_to_photo(fig: Figure, width_px: int, height_px: int) -> ImageTk.PhotoImage:
    dpi = 110
    fig.set_size_inches(width_px / dpi, height_px / dpi)
    return _fig_to_photo(fig)


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, event_queue: queue.Queue) -> None:
        super().__init__()
        self._q                = event_queue
        self._words: list[str] = []

        self._latex_photo: ImageTk.PhotoImage | None = None
        self._sol_photo:   ImageTk.PhotoImage | None = None
        self._graph_photo: ImageTk.PhotoImage | None = None

        self._last_result: TranslationResult | None = None

        self._build_ui()
        self._poll_queue()

    # ── construction ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.title("Text → LaTeX")
        self.configure(bg=BG)
        self.geometry("900x580")
        self.minsize(660, 400)
        self.resizable(True, True)

        self._build_header()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._build_latex_panel()
        self._build_solution_panel()
        self._build_graph_panel()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._build_input_panel()

        self.bind("<Configure>", self._on_resize)
        self._last_wh = (0, 0)

    def _build_header(self) -> None:
        bar = tk.Frame(self, bg=BG, pady=10, padx=20)
        bar.pack(fill="x")
        tk.Label(bar, text="TEXT → LaTeX", bg=BG, fg=FG_MAIN,
                 font=FONT_HEADER).pack(side="left")
        self._status_dot = tk.Label(bar, text="●", bg=BG, fg=FG_DIM,
                                    font=FONT_HEADER)
        self._status_dot.pack(side="right")
        self._status_lbl = tk.Label(bar, text="waiting…", bg=BG, fg=FG_DIM,
                                    font=FONT_SMALL)
        self._status_lbl.pack(side="right", padx=(0, 6))

    def _build_latex_panel(self) -> None:
        frame = tk.Frame(self, bg=SURFACE, height=_LATEX_H)
        frame.pack(fill="x")
        frame.pack_propagate(False)
        self._latex_lbl = tk.Label(frame, bg=SURFACE)
        self._latex_lbl.pack(expand=True, fill="both")
        self._latex_panel = frame

    def _build_solution_panel(self) -> None:
        """Shown only when SymPy finds a solution — renders LaTeX image."""
        frame = tk.Frame(self, bg=SURFACE_ALT, height=_SOL_H)
        frame.pack_propagate(False)
        self._sol_lbl   = tk.Label(frame, bg=SURFACE_ALT)
        self._sol_lbl.pack(expand=True, fill="both")
        self._sol_panel  = frame
        self._sol_shown  = False

    def _build_graph_panel(self) -> None:
        frame = tk.Frame(self, bg=SURFACE, height=_GRAPH_H)
        frame.pack_propagate(False)
        self._graph_lbl = tk.Label(frame, bg=SURFACE)
        self._graph_lbl.pack(expand=True, fill="both")
        self._graph_panel = frame
        self._graph_shown = False

    def _build_input_panel(self) -> None:
        frame = tk.Frame(self, bg=SURFACE_ALT, pady=8, padx=20)
        frame.pack(fill="x", side="bottom")

        top_row = tk.Frame(frame, bg=SURFACE_ALT)
        top_row.pack(fill="x")
        self._words_var = tk.StringVar(value="")
        tk.Label(top_row, textvariable=self._words_var, bg=SURFACE_ALT,
                 fg=FG_DIM, font=FONT_SMALL, anchor="w").pack(side="left", fill="x", expand=True)
        self._count_var = tk.StringVar(value="0 / 40")
        tk.Label(top_row, textvariable=self._count_var, bg=SURFACE_ALT,
                 fg=FG_DIM, font=FONT_TINY).pack(side="right")

        self._partial_var = tk.StringVar(value="")
        tk.Label(frame, textvariable=self._partial_var, bg=SURFACE_ALT,
                 fg=FG_DIM, font=FONT_TINY, anchor="w").pack(fill="x", pady=(2, 0))

    # ── panel show / hide ─────────────────────────────────────────────────────

    def _show_solution_panel(self) -> bool:
        if not self._sol_shown:
            self._latex_panel.pack_forget()
            self._sol_panel.pack_forget()
            self._graph_panel.pack_forget()
            self._latex_panel.pack(fill="x")
            self._sol_panel.pack(fill="x")
            if self._graph_shown:
                self._graph_panel.pack(fill="x", expand=True)
        return True

    def _hide_solution_panel(self) -> bool:
        if self._sol_shown:
            self._sol_panel.pack_forget()
        return False

    def _set_solution_panel(
        self,
        solution_latex: str | None,
        info: str | None,
        width: int,
    ) -> None:
        if solution_latex:
            # render as LaTeX math image
            photo = _render_latex_image(
                solution_latex, width, _SOL_H,
                color=FG_SOL, bg=SURFACE_ALT, fontsize=22,
            )
            self._sol_photo = photo
            self._sol_lbl.config(image=photo)
            self._sol_shown = self._show_solution_panel()
        elif info:
            # render as plain text image
            photo = _render_text_image(
                f"ℹ  {info}", width, _SOL_H,
                color=FG_WARN, bg=SURFACE_ALT,
            )
            self._sol_photo = photo
            self._sol_lbl.config(image=photo)
            self._sol_shown = self._show_solution_panel()
        else:
            self._sol_shown = self._hide_solution_panel()

    def _set_graph_panel(self, fig: Figure | None, width: int) -> None:
        if fig is not None:
            photo = _graph_fig_to_photo(fig, width, _GRAPH_H)
            self._graph_photo = photo
            self._graph_lbl.config(image=photo)
            if not self._graph_shown:
                self._graph_panel.pack(fill="x", expand=True)
                self._graph_shown = True
        else:
            if self._graph_shown:
                self._graph_panel.pack_forget()
                self._graph_shown = False

    # ── resize ────────────────────────────────────────────────────────────────

    def _on_resize(self, event: tk.Event) -> None:
        wh = (self.winfo_width(), self.winfo_height())
        if wh != self._last_wh and self._last_result is not None:
            self._last_wh = wh
            self._render_result(self._last_result)

    # ── queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self) -> None:
        try:
            while True:
                self._handle_event(self._q.get_nowait())
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    # ── events ────────────────────────────────────────────────────────────────

    def _handle_event(self, event: dict) -> None:
        kind = event.get("type")

        if kind == "partial":
            self._partial_var.set(event.get("text", ""))

        elif kind == "final":
            self._partial_var.set("")
            for word in event.get("words", []):
                if word == config.END_WORD:
                    self._finish_sentence()
                    break
                if len(self._words) < config.MAX_WORDS:
                    self._words.append(word)
            self._refresh_words()

        elif kind == "ready":
            # only set listening when receiver actually connects
            self._status_lbl.config(text="listening", fg=FG_END)
            self._status_dot.config(fg=FG_END)

        elif kind == "disconnected":
            self._status_lbl.config(text="waiting…", fg=FG_DIM)
            self._status_dot.config(fg=FG_DIM)

        elif kind == "error":
            msg = event.get("message", "")
            debug_log("UI", f"error event: {msg}")
            self._status_lbl.config(text="error", fg=FG_ERROR)
            self._status_dot.config(fg=FG_ERROR)
            self._partial_var.set(f"⚠  {msg}")

    # ── sentence completion ───────────────────────────────────────────────────

    def _finish_sentence(self) -> None:
        self._latex_lbl.config(bg=FG_END)
        self.after(350, lambda: self._latex_lbl.config(bg=SURFACE))

        if not self._words:
            debug_log("TRANSLATOR", "fin on empty buffer")
            self._words.clear()
            self._refresh_words()
            return

        debug_log("TRANSLATOR", f"translating: {self._words}")
        result = translate(self._words)
        debug_log("TRANSLATOR", f"→ ok={result.ok}  latex={result.latex or result.error}")

        self._last_result = result
        self.after(360, lambda: self._render_result(result))

        self._words.clear()
        self._refresh_words()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _render_result(self, result: TranslationResult) -> None:
        self.update_idletasks()
        w = max(self.winfo_width(), 600)

        # LaTeX panel
        if result.ok:
            photo = _render_latex_image(result.latex, w, _LATEX_H)
        else:
            photo = _render_error_image(result.error, w, _LATEX_H)
        self._latex_photo = photo
        self._latex_lbl.config(image=photo)

        # SymPy analysis
        analysis = None
        if result.ok and result.ast is not None:
            try:
                analysis = analyse(result.ast)
                debug_log("SYMPY", f"solution={analysis.solution_latex}  "
                          f"graph={'yes' if analysis.graph_fig else 'no'}  "
                          f"info={analysis.info}")
            except Exception as exc:
                debug_log("SYMPY", f"analyse failed: {exc}")

        # Solution panel — pass latex and info separately
        self._set_solution_panel(
            analysis.solution_latex if analysis else None,
            analysis.info           if analysis else None,
            w,
        )

        # Graph panel
        self._set_graph_panel(
            analysis.graph_fig if analysis else None,
            w,
        )

    # ── word display ──────────────────────────────────────────────────────────

    def _refresh_words(self) -> None:
        self._words_var.set("  ".join(self._words))
        self._count_var.set(f"{len(self._words)} / {config.MAX_WORDS}")

    def _show_placeholder(self) -> None:
        self.update_idletasks()
        w = max(self.winfo_width(), 600)
        photo = _render_latex_image(
            r"\text{waiting for speech...}",
            w, _LATEX_H, color=FG_DIM, fontsize=20,
        )
        self._latex_photo = photo
        self._latex_lbl.config(image=photo)