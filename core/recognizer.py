"""
core/recognizer.py
──────────────────
Reads raw audio from a FIFO instead of a sounddevice stream.
All audio work happens in a daemon thread; results are pushed
onto a queue.Queue so the UI can poll them safely.

Queue message format
────────────────────
Every item is a dict with a "type" key:

    {"type": "partial", "text": "uno dos"}
        Live, not-yet-confirmed hypothesis.

    {"type": "final", "words": ["uno", "dos", "tres"]}
        Confirmed words from a completed utterance.
        If END_WORD appears it is included so the UI can act on it.

    {"type": "error", "message": "..."}
        Something went wrong in the audio thread.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Callable

from vosk import KaldiRecognizer, Model

import config


class Recognizer:
    """
    Manages the Vosk model and reads audio from a FIFO.

    Parameters
    ----------
    event_queue:
        Queue where recognition events are posted.
    on_ready:
        Optional callback invoked (from the audio thread) once the model
        has loaded and the FIFO is about to be opened.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self._q        = event_queue
        self._on_ready = on_ready
        self._thread:     threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start recognition in a background daemon thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to shut down."""
        self._stop_event.set()

    # ── internals ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        config.debug_log("RECOGNIZER", f"loading model from '{config.MODEL_PATH}'")
        try:
            model      = Model(config.MODEL_PATH)
            recognizer = KaldiRecognizer(model, config.SAMPLE_RATE, config.VOCABULARY_JSON)
            config.debug_log("RECOGNIZER", "model loaded OK")
            config.debug_log("RECOGNIZER", f"vocabulary → {config.VOCABULARY_JSON}")
        except Exception as exc:
            config.debug_log("RECOGNIZER", f"model load FAILED: {exc}")
            self._q.put({"type": "error", "message": str(exc)})
            return

        if self._on_ready:
            self._on_ready()

        # 100ms worth of int16 mono samples
        chunk_bytes = int(config.SAMPLE_RATE * 0.1) * 2

        config.debug_log("RECOGNIZER", f"opening FIFO at '{config.FIFO_PATH}'")
        try:
            # ensure the FIFO exists
            if not os.path.exists(config.FIFO_PATH):
                os.mkfifo(config.FIFO_PATH)

            # open() on a FIFO blocks until the writer (audio_bridge) connects
            with open(config.FIFO_PATH, "rb") as fifo:
                config.debug_log("RECOGNIZER", "FIFO open — listening")
                while not self._stop_event.is_set():
                    data = fifo.read(chunk_bytes)
                    if not data:
                        # writer disconnected — wait for reconnect
                        config.debug_log("RECOGNIZER", "FIFO closed, waiting for writer...")
                        break

                    if recognizer.AcceptWaveform(data):
                        result    = json.loads(recognizer.Result())
                        text: str = result.get("text", "").strip()
                        if text:
                            words = text.split()
                            config.debug_log("FINAL", f"{words}")
                            self._q.put({"type": "final", "words": words})
                    else:
                        partial      = json.loads(recognizer.PartialResult())
                        partial_text = partial.get("partial", "").strip()
                        if partial_text:
                            config.debug_log("PARTIAL", partial_text)
                        self._q.put({"type": "partial", "text": partial_text})

                config.debug_log("RECOGNIZER", "stop requested — closing FIFO")

        except Exception as exc:
            config.debug_log("RECOGNIZER", f"FIFO error: {exc}")
            self._q.put({"type": "error", "message": str(exc)})