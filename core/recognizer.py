"""
core/recognizer.py
──────────────────
Wraps Vosk + sounddevice.  All audio work happens in a daemon thread;
results are pushed onto a queue.Queue so the UI can poll them safely.

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
import queue
import sys
import threading
from typing import Callable

import sounddevice as sd
from vosk import KaldiRecognizer, Model

import config


class Recognizer:
    """
    Manages the Vosk model and the sounddevice input stream.

    Parameters
    ----------
    event_queue:
        Queue where recognition events are posted.
    on_ready:
        Optional callback invoked (from the audio thread) once the model
        has loaded and the stream is about to start.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self._q = event_queue
        self._on_ready = on_ready
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
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
            model = Model(config.MODEL_PATH)
            recognizer = KaldiRecognizer(model, config.SAMPLE_RATE, config.VOCABULARY_JSON)
            config.debug_log("RECOGNIZER", "model loaded OK")
            config.debug_log("RECOGNIZER", f"vocabulary → {config.VOCABULARY_JSON}")
        except Exception as exc:
            config.debug_log("RECOGNIZER", f"model load FAILED: {exc}")
            self._q.put({"type": "error", "message": str(exc)})
            return

        if self._on_ready:
            self._on_ready()

        def _audio_cb(indata, frames, time, status):  # noqa: ANN001
            if status:
                config.debug_log("AUDIO", f"stream status: {status}")
                print(status, file=sys.stderr)

            if recognizer.AcceptWaveform(bytes(indata)):
                result = json.loads(recognizer.Result())
                text: str = result.get("text", "").strip()
                if text:
                    words = text.split()
                    config.debug_log("FINAL", f"{words}")
                    self._q.put({"type": "final", "words": words})
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text: str = partial.get("partial", "").strip()
                if partial_text:
                    config.debug_log("PARTIAL", partial_text)
                self._q.put({"type": "partial", "text": partial_text})

        config.debug_log("RECOGNIZER", f"opening audio stream at {config.SAMPLE_RATE} Hz")
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                callback=_audio_cb,
            ):
                config.debug_log("RECOGNIZER", "stream open — listening")
                while not self._stop_event.is_set():
                    sd.sleep(100)
                config.debug_log("RECOGNIZER", "stop requested — closing stream")
        except Exception as exc:
            config.debug_log("RECOGNIZER", f"stream error: {exc}")
            self._q.put({"type": "error", "message": str(exc)})