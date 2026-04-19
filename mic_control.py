"""
mic_control.py
──────────────
Usage:
    python mic_control.py start
    python mic_control.py stop
    python mic_control.py status
"""

import sys
import urllib.request

PC_HOSTNAME  = "NITSUGAPC.local"
CONTROL_PORT = 8766


def send(command: str) -> None:
    try:
        url = f"http://{PC_HOSTNAME}:{CONTROL_PORT}/{command}"
        res = urllib.request.urlopen(url, timeout=3)
        print(f"[control] {command} → {res.read().decode()}")
    except Exception as e:
        print(f"[control] failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("start", "stop", "status"):
        print("Usage: python mic_control.py start|stop|status")
        sys.exit(1)

    send(sys.argv[1])