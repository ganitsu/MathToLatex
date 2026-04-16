"""
main.py
───────
Entry point.  Wires the recognizer to the UI via a shared queue.

Run:
    python main.py
"""

import queue

from core.recognizer import Recognizer
from ui.app import App


def main() -> None:
    event_queue: queue.Queue = queue.Queue()

    # The recognizer posts a {"type": "ready"} event so the UI can update
    # its status indicator once the model has finished loading.
    def on_ready() -> None:
        event_queue.put({"type": "ready"})

    recognizer = Recognizer(event_queue, on_ready=on_ready)
    recognizer.start()

    app = App(event_queue)

    try:
        app.mainloop()
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main()