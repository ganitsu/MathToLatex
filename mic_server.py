"""
mic_server.py
─────────────
Lightweight HTTP control server. Waits for start/stop
commands from the RPi, spawns/kills the recognizer.

Run once at startup:
    python mic_server.py
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import threading
import sys
import os

CONTROL_PORT = 8766

proc        = None
lock        = threading.Lock()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use the venv python if it exists, fall back to current interpreter
venv_python = os.path.join(script_dir, "venv", "Scripts", "python.exe")
python      = venv_python if os.path.exists(venv_python) else sys.executable

print(f"[server] using python: {python}")

def start_recognizer():
    global proc
    with lock:
        if proc and proc.poll() is None:
            print("[server] already running")
            return
        print("[server] starting recognizer...")
        proc = subprocess.Popen(
            f'"{os.path.join(script_dir, "venv", "Scripts", "activate")}" && python main.py --mode recognizer',
            cwd=script_dir,
            shell=True,
        )
        print(f"[server] started (pid {proc.pid})")


def stop_recognizer():
    global proc
    with lock:
        if proc and proc.poll() is None:
            subprocess.run(
                ["taskkill", "/f", "/t", "/pid", str(proc.pid)],
                capture_output=True
            )
            proc = None
            print("[server] stopped")
        else:
            print("[server] nothing to stop")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/start":
            start_recognizer()
            self._respond(200, "started")
        elif self.path == "/stop":
            stop_recognizer()
            self._respond(200, "stopped")
        elif self.path == "/status":
            running = proc is not None and proc.poll() is None
            self._respond(200, "running" if running else "stopped")
        else:
            self._respond(404, "not found")

    def _respond(self, code, msg):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(msg.encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"[server] control server listening on port {CONTROL_PORT}")
    print("[server] waiting for commands from RPi...")
    try:
        HTTPServer(("0.0.0.0", CONTROL_PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        stop_recognizer()
        print("\n[server] shut down")