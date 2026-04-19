"""
core/word_sender.py
───────────────────
Reads events from a queue and sends them over TCP to the RPi.
Runs in recognizer mode on Windows.
Reconnects automatically if the RPi drops.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time

import config


class WordSender:
    def __init__(self, event_queue: queue.Queue) -> None:
        self._q          = event_queue
        self._stop_event = threading.Event()
        self._thread:     threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                config.debug_log("SENDER", f"connecting to {config.RPI_HOST}:{config.NETWORK_PORT}")
                with socket.create_connection(
                    (config.RPI_HOST, config.NETWORK_PORT), timeout=5
                ) as sock:
                    config.debug_log("SENDER", "connected")
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    while not self._stop_event.is_set():
                        try:
                            event = self._q.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        line = json.dumps(event) + "\n"
                        sock.sendall(line.encode())
                        config.debug_log("SENDER", f"sent: {event}")

            except Exception as exc:
                config.debug_log("SENDER", f"connection error: {exc} — retrying in 3s")
                time.sleep(3)