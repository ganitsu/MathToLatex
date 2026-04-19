"""
core/recognizer.py
──────────────────
Wraps Vosk + sounddevice. Used in recognizer and standalone modes.
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
    def __init__(
        self,
        event_queue: queue.Queue,
        on_ready: Callable[[], None] | None = None,
    ) -> None:
        self._q          = event_queue
        self._on_ready   = on_ready
        self._thread:     threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        config.debug_log("RECOGNIZER", f"loading model from '{config.MODEL_PATH}'")
        try:
            model      = Model(config.MODEL_PATH)
            recognizer = KaldiRecognizer(model, config.SAMPLE_RATE, config.VOCABULARY_JSON)
            config.debug_log("RECOGNIZER", "model loaded OK")
        except Exception as exc:
            config.debug_log("RECOGNIZER", f"model load FAILED: {exc}")
            self._q.put({"type": "error", "message": str(exc)})
            return

        if self._on_ready:
            self._on_ready()

        def _audio_cb(indata, frames, time, status):
            if status:
                config.debug_log("AUDIO", f"stream status: {status}")
                print(status, file=sys.stderr)
            if recognizer.AcceptWaveform(bytes(indata)):
                result = json.loads(recognizer.Result())
                text   = result.get("text", "").strip()
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

        config.debug_log("RECOGNIZER", f"opening audio stream at {config.SAMPLE_RATE} Hz")
        try:
            with sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                device=config.AUDIO_DEVICE,
                channels=1,
                dtype="int16",
                callback=_audio_cb,
            ):
                config.debug_log("RECOGNIZER", "stream open — listening")
                while not self._stop_event.is_set():
                    sd.sleep(100)
        except Exception as exc:
            config.debug_log("RECOGNIZER", f"stream error: {exc}")
            self._q.put({"type": "error", "message": str(exc)})