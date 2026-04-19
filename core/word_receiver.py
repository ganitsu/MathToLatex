"""
core/word_receiver.py
─────────────────────
Listens on a TCP port for word events sent by the Windows recognizer.
Puts them directly into the UI event queue.
Runs in display mode on the RPi.
"""

from __future__ import annotations

import json
import queue
import socket
import threading

import config


class WordReceiver:
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
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", config.NETWORK_PORT))
        server.listen(1)
        server.settimeout(1.0)
        config.debug_log("RECEIVER", f"listening on port {config.NETWORK_PORT}")

        while not self._stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            config.debug_log("RECEIVER", f"connected: {addr}")
            self._q.put({"type": "ready"})  # signal UI that connection is live

            try:
                buf = ""
                while not self._stop_event.is_set():
                    data = conn.recv(4096).decode()
                    if not data:
                        break
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                            config.debug_log("RECEIVER", f"got: {event}")
                            self._q.put(event)
                        except json.JSONDecodeError:
                            config.debug_log("RECEIVER", f"bad JSON: {line}")
            except Exception as exc:
                config.debug_log("RECEIVER", f"connection lost: {exc}")
            finally:
                conn.close()
                self._q.put({"type": "disconnected"})   # ← add this
                config.debug_log("RECEIVER", "client disconnected — waiting for new connection")

        server.close()