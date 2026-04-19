"""
main.py
───────
Entry point.

Modes
─────
    python main.py                    # standalone — mic + UI on same machine
    python main.py --mode recognizer  # Windows — mic → sends words to RPi
    python main.py --mode display     # RPi    — receives words → UI
"""

import argparse
import queue

from ui.app import App


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["standalone", "recognizer", "display"],
        default="standalone",
    )
    args = parser.parse_args()

    event_queue: queue.Queue = queue.Queue()

    if args.mode == "recognizer":
        # Windows: mic → Vosk → send words over TCP to RPi
        from core.recognizer import Recognizer
        from core.word_sender import WordSender

        sender     = WordSender(event_queue)
        recognizer = Recognizer(
            event_queue,
            on_ready=lambda: print("[main] model ready — listening"),
        )
        sender.start()
        recognizer.start()

        print("[main] recognizer mode — press Ctrl+C to stop")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            recognizer.stop()
            sender.stop()

    elif args.mode == "display":
        # RPi: receive words over TCP → UI
        from core.word_receiver import WordReceiver

        receiver = WordReceiver(event_queue)
        receiver.start()

        app = App(event_queue)
        try:
            app.mainloop()
        finally:
            receiver.stop()

    else:
        # Standalone: original behavior
        from core.recognizer import Recognizer

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