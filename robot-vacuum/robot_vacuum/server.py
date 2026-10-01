"""HTTP viewer and a single fixed-step physics worker."""
import json
import logging
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .paths import asset_dir
from .simulation import Simulator

LOG = logging.getLogger("vacuum")


class Mission:
    def __init__(self, home: Path | None, seed: int):
        self.sim = Simulator(home_path=home, seed=seed)
        self.lock = threading.RLock()
        self.speed = 4.0
        self.stop = threading.Event()
        self.error = None

    def tick(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    for _ in range(max(1, round(self.speed))):
                        self.sim.step(0.05)
                self.stop.wait(max(0.0, 0.05 - (time.monotonic() - started)))
        except Exception:
            LOG.exception("Physics worker stopped")
            with self.lock:
                self.error = "The physics worker stopped. Restart the server and check its log."

    def snapshot(self):
        with self.lock:
            state = self.sim.snapshot()
            state.update(speed=self.speed, engine="MuJoCo", error=self.error)
            return state

    def control(self, payload):
        action = payload.get("action")
        with self.lock:
            if action == "start":
                self.sim.start()
            elif action == "pause":
                self.sim.pause()
            elif action == "reset":
                self.sim.reset()
            elif action == "dock":
                self.sim.return_to_dock()
            elif action == "speed":
                speed = payload.get("speed")
                if not isinstance(speed, (int, float)) or speed not in (1, 4, 12, 32):
                    raise ValueError("Choose a speed of 1, 4, 12, or 32.")
                self.speed = float(speed)
            else:
                raise ValueError("Unknown mission action.")
        return self.snapshot()


def run_server(host="127.0.0.1", port=18881, home=None, seed=7):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    mission = Mission(home, seed)
    static = asset_dir("static").resolve()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if "/api/state" not in (args[0] if args else ""):
                LOG.info(fmt, *args)

        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
            self.end_headers()
            self.close_connection = True
            # Send static assets in small chunks across the VM's forwarded connection.
            for start in range(0, len(body), 8192):
                self.wfile.write(body[start:start + 8192])


        def json(self, status, value):
            self.send(status, json.dumps(value, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/api/state":
                    self.json(200, mission.snapshot())
                elif path == "/api/home":
                    self.json(200, mission.sim.home)
                elif path in ("/health", "/api/health"):
                    self.json(200 if mission.error is None else 503,
                              {"app": "robot-vacuum", "engine": "MuJoCo", "ready": mission.error is None})
                else:
                    relative = "index.html" if path == "/" else path.removeprefix("/static/")
                    file = (static / relative).resolve()
                    if not file.is_relative_to(static) or not file.is_file():
                        self.json(404, {"error": "File not found."})
                    else:
                        mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
                        self.send(200, file.read_bytes(), mime)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                LOG.exception("Request failed")
                self.json(500, {"error": "Request failed. Check the server log."})

        def do_POST(self):
            if urlparse(self.path).path != "/api/control":
                self.json(404, {"error": "Endpoint not found."})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                self.json(200, mission.control(payload))
            except (ValueError, TypeError) as exc:
                self.json(400, {"error": str(exc)})

    class ViewerServer(ThreadingHTTPServer):
        request_queue_size = 64

    server = ViewerServer((host, port), Handler)
    worker = threading.Thread(target=mission.tick, daemon=True, name="vacuum-physics")
    worker.start()
    LOG.info("Maple House ready at http://%s:%s. Mission is paused.", host, port)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        mission.stop.set()
        worker.join(timeout=5)
        server.server_close()
