"""HTTP viewer and commands for the VM's authoritative physics simulation."""

from __future__ import annotations

import json
import os
import re
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .simulation import Simulation
from .yard import description

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/three/three.module.js": ("three/three.module.js", "text/javascript; charset=utf-8"),
    "/three/three.core.js": ("three/three.core.js", "text/javascript; charset=utf-8"),
    "/three/addons/controls/OrbitControls.js": ("three/addons/controls/OrbitControls.js", "text/javascript; charset=utf-8"),
}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, simulation):
        self.simulation = simulation
        self.stopping = threading.Event()
        super().__init__(address, DemoHandler)


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer
    protocol_version = "HTTP/1.1"

    def _body(self, data, content_type="application/json", status=200):
        if isinstance(data, dict):
            data = json.dumps(data, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        sim = self.server.simulation
        if path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            self._body((STATIC_ROOT / name).read_bytes(), content_type)
        elif path == "/api/scene":
            self._body(description())
        elif path in ("/api/health", "/api/status"):
            status = sim.status()
            self._body(status, status=200 if status["ready"] else 503)
        elif path == "/api/state":
            with sim.lock:
                state = sim.snapshot()
            self._body(state)
        elif path == "/api/camera.jpg":
            with sim.lock:
                image = sim._camera_jpeg
            if image:
                self._body(image, "image/jpeg")
            else:
                self._body({"error": sim.sensor_error or "Camera is loading"}, status=503)
        elif path == "/api/report":
            with sim.lock:
                report = sim.missions.snapshot()
            self._body(report)
        elif path == "/api/stream":
            self._stream()
        elif re.fullmatch(r"/api/evidence/[a-f0-9]{12}-[A-Z0-9]{1,8}\.png", path):
            target = sim.missions.evidence / path.rsplit("/", 1)[-1]
            if target.is_file():
                self._body(target.read_bytes(), "image/png")
            else:
                self._body({"error": "Evidence image not found"}, status=404)
        else:
            self._body({"error": "Endpoint not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if path == "/api/voice":
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("Send microphone audio smaller than 8 MiB")
                transcript = self.server.simulation.voice.transcribe(self.rfile.read(size), self.headers.get("Content-Type", "audio/webm"))
                if not transcript:
                    self._body({"ok": False, "message": "No speech was detected."}, status=400)
                    return
                result = self.server.simulation.command(transcript)
                result.update(ok=result.get("accepted", False), transcript=transcript)
                self._body(result, status=200 if result["ok"] else 400)
                return
            if not 0 < size <= 4096:
                raise ValueError("Send a JSON request smaller than 4 KiB")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Send a JSON object")
            if path == "/api/command":
                if not isinstance(body.get("text"), str) or len(body["text"]) > 500:
                    raise ValueError("A command text of at most 500 characters is required")
                result = self.server.simulation.command(body["text"])
                result["ok"] = result.get("accepted", False)
                self._body(result, status=200 if result["ok"] else 400)
            elif path == "/api/action":
                result = self.server.simulation.action(body)
                result["ok"] = result.get("accepted", False)
                self._body(result, status=200 if result["ok"] else 400)
            elif path == "/api/incident":
                result = self.server.simulation.action({"action": "inject", "hazard": body.get("type"), "position": body.get("position")})
                result["ok"] = result.get("accepted", False)
                self._body(result, status=200 if result["ok"] else 400)
            elif path == "/api/worker":
                result = self.server.simulation.move_worker(float(body.get("dx", 0)), float(body.get("dy", 0)), stop=body.get("stop") is True)
                result["ok"] = result.get("accepted", False)
                self._body(result, status=200 if result["ok"] else 400)
            else:
                self._body({"ok": False, "message": "Endpoint not found"}, status=404)
        except RuntimeError as error:
            self._body({"ok": False, "message": str(error)}, status=503)
        except (ValueError, TypeError, KeyError) as error:
            self._body({"ok": False, "message": str(error)}, status=400)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        sim = self.server.simulation
        try:
            while not self.server.stopping.is_set():
                started = time.monotonic()
                with sim.lock:
                    state = sim.snapshot()
                self.wfile.write(b"event: state\ndata: " + json.dumps(state, separators=(",", ":"), allow_nan=False).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(max(0.001, 1 / 10 - (time.monotonic() - started)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        if os.environ.get("YARD_DRONE_HTTP_LOG") == "1":
            super().log_message(*args)


def main():
    sim = Simulation(realtime_speed=float(os.environ.get("YARD_DRONE_SPEED", "1")))
    sim.start()
    port = int(os.environ.get("YARD_DRONE_PORT", "8895"))
    server = DemoServer((os.environ.get("YARD_DRONE_HOST", "0.0.0.0"), port), sim)

    def shutdown(_signum, _frame):
        server.stopping.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print(f"Construction yard drone: MuJoCo {sim.status()['mujocoVersion']}, 500 Hz physics, viewer port {port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        sim.close()


if __name__ == "__main__":
    main()
