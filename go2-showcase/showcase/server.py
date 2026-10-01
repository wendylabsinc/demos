"""Small operator server. Motion always requires an explicit start request."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import queue
from concurrent.futures import Future
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

from showcase.models import Command

DEMOS = {
    "object_search": "Find my keys",
    "obstacle_course": "Solve the changing course",
    "cinematographer": "Follow the shot",
    "dance": "Learn a dance",
    "person_search": "Find the missing person",
}

GOALS = {
    "object_search": "Find a set of keys using the robot camera. Explore if you cannot see them. Report what you actually see. Do not claim success without visible keys.",
    "obstacle_course": "Navigate through the room without touching obstacles. Look for an open passage and reach the far side. Reconsider each movement from the newest image and scan.",
    "cinematographer": "Find the person and keep them centered in the robot camera while maintaining safe distance. Stop if you lose them. Keep following until the operator stops.",
    "dance": "Observe the person's arm gestures through the camera and respond with a short safe turn or step that imitates them. Hold if no gesture is visible. Keep observing until stopped.",
    "person_search": "Search for a person through the camera. Explore unseen directions, and report the person only if you can actually see them.",
}


class Runtime:
    def __init__(self, transport_factory, hz=20, agent_factory=None):
        from concurrent.futures import ThreadPoolExecutor
        from showcase.ai import VisionAgent
        from showcase.action_executor import ActionExecutor
        self.transport_factory = transport_factory
        self.transport = None
        self.agent = (agent_factory or VisionAgent)()
        self.executor = ActionExecutor(budget_s=180, geofence_m=5)
        self.hz = hz
        self.lock = threading.RLock()
        self.observation = None
        self.command = Command()
        self.error = None
        self.alive = True
        self.requests = queue.Queue()
        self.ready = threading.Event()
        self.jpeg = None
        self.last_frame = 0
        self.sim_time = 0.0
        self.goal = ""
        self.history = []
        self.pending = None
        self.generation = 0
        self.last_decision = None
        self.inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="go2-vision-agent")
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        if not self.ready.wait(60):
            raise RuntimeError("Simulator initialization timed out")
        if self.transport is None:
            raise RuntimeError(self.error)

    def submit(self, operation):
        if threading.current_thread() is self.thread:
            return operation()
        result = Future()
        self.requests.put((operation, result))
        return result.result(timeout=30)

    def control_time(self):
        return self.sim_time if hasattr(self.transport, "advance") else time.monotonic()

    def _think(self):
        snapshot = self.transport.sensor_snapshot() if hasattr(self.transport, "sensor_snapshot") else {
            "pose": {"x": self.observation.x, "y": self.observation.y, "yaw": self.observation.yaw},
            "front_clearance": self.observation.front_clearance,
            "scan_sectors": list(self.observation.sectors),
            "healthy": self.observation.fresh and self.observation.scan_fresh,
        }
        frame = self.transport.camera_jpeg() if hasattr(self.transport, "camera_jpeg") else self.transport.frame_jpeg()
        if not frame:
            raise RuntimeError("A live robot camera frame is required for AI decisions")
        snapshot["executor"] = self.executor.status()
        self.pending = (self.inference.submit(self.agent.decide, self.goal, frame, snapshot, list(self.history[-12:])),
                        self.generation, time.monotonic(), (self.observation.x, self.observation.y, self.observation.yaw))

    def _accept_decision(self):
        future, generation, captured, pose = self.pending
        if not future.done():
            return
        self.pending = None
        if generation != self.generation:
            return
        decision = future.result()
        if time.monotonic() - captured > float(os.getenv("GO2_AI_MAX_DECISION_AGE", "30")):
            raise RuntimeError("The model response is too old; motion remains stopped")
        if math.hypot(self.observation.x-pose[0], self.observation.y-pose[1]) > .1:
            raise RuntimeError("The robot moved during inference; discard the old decision")
        data = asdict(decision)
        self.last_decision = data
        self.history.append({"decision": data, "pose": pose, "inference_seconds": round(time.monotonic()-captured, 3)})
        self.history = self.history[-100:]
        self.executor.begin(decision, self.observation, self.control_time())
        print(json.dumps({"event": "model_decision", "model": getattr(self.agent, "model", "configured"), **data}), flush=True)

    def loop(self):
        try:
            self.transport = self.transport_factory()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self.ready.set()
        while self.alive:
            began = time.monotonic()
            with self.lock:
                try:
                    while not self.requests.empty():
                        operation, result = self.requests.get_nowait()
                        try:
                            result.set_result(operation())
                        except Exception as exc:
                            result.set_exception(exc)
                    if not self.alive:
                        break
                    self.observation = self.transport.read()
                    state = self.executor.status()["state"]
                    if self.pending:
                        self._accept_decision()
                    self.command = self.executor.step(self.observation, self.control_time())
                    new_state = self.executor.status()["state"]
                    if state == "running" and new_state == "waiting" and self.history:
                        self.history[-1]["execution"] = {"status": self.executor.status(),
                            "pose": [self.observation.x, self.observation.y, self.observation.yaw]}
                    state = new_state
                    if state in ("waiting", "waiting_model") and self.pending is None and self.goal:
                        self._think()
                    self.transport.send(self.command)
                    if state in ("fault", "complete", "disabled"):
                        self.transport.stop()
                    if hasattr(self.transport, "advance"):
                        self.transport.advance(1 / self.hz)
                        self.sim_time += 1 / self.hz
                    if hasattr(self.transport, "frame_jpeg") and began-self.last_frame >= .5:
                        self.jpeg = self.transport.camera_jpeg() if hasattr(self.transport, "camera_jpeg") else self.transport.frame_jpeg()
                        self.last_frame = began
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.executor.stop(self.error)
                    self.generation += 1
                    self.command = Command()
                    try:
                        self.transport.stop()
                    except Exception:
                        pass
            time.sleep(max(0, 1 / self.hz - (time.monotonic()-began)))

    def start(self, payload):
        return self.submit(lambda: self._start(payload))

    def _start(self, payload):
        instruction = payload.get("instruction") or GOALS.get(payload.get("demo"))
        if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 2000:
            raise ValueError("Describe what you want the robot to do")
        if self.pending and not self.pending[0].done():
            raise RuntimeError("The previous model request is still finishing; the robot is stopped")
        self.transport.stop()
        self.generation += 1
        self.pending = None
        self.goal = instruction.strip()
        self.history = []
        self.last_decision = None
        self.error = None
        self.observation = self.transport.read()
        self.executor.enable(self.observation, self.control_time())
        if self.executor.status()["state"] == "fault":
            raise RuntimeError(self.executor.status().get("reason", "Sensors are not ready"))
        self.transport.arm()
        return self.status()

    def stop(self):
        return self.submit(self._stop)

    def _stop(self):
        self.generation += 1
        self.executor.stop("Operator stop")
        self.transport.stop()
        self.command = Command()

    def status(self):
        with self.lock:
            return {
                "app": "sh.wendy.go2-showcase", "mode": "vision_language_agent",
                "demos": DEMOS, "mission": self.executor.status(), "instruction": self.goal,
                "observation": asdict(self.observation) if self.observation else None,
                "command": asdict(self.command), "transport": self.transport.status(),
                "ai": {"model": getattr(self.agent, "model", os.getenv("GO2_AI_MODEL", "configured")),
                       "thinking": bool(self.pending and not self.pending[0].done()),
                       "last_decision": self.last_decision, "decisions": len(self.history)},
                "history": self.history[-12:], "error": self.error,
            }

    def close(self):
        def shutdown():
            self._stop()
            self.transport.close()
            self.alive = False
        self.submit(shutdown)
        self.thread.join(timeout=2)
        self.inference.shutdown(wait=False, cancel_futures=True)


def handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, code, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, Path(__file__).with_name("dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif path in ("/status", "/health"):
                data = runtime.status()
                self.respond(503 if path == "/health" and data["error"] else 200, data)
            elif path == "/camera.jpg":
                with runtime.lock:
                    if runtime.jpeg:
                        self.respond(200, runtime.jpeg, "image/jpeg")
                    else:
                        self.respond(404, {"error": "Camera preview is unavailable"})
            else:
                self.respond(404, {"error": "Unknown route"})

        def do_POST(self):
            # A local/LAN operator app. Cross-origin browser mutation is rejected.
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host"):
                self.respond(403, {"error": "Use the operator page on this host"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 4096:
                    raise ValueError("Request is too large")
                payload = json.loads(self.rfile.read(size) or b"{}")
                if self.path == "/start":
                    data = runtime.start(payload)
                elif self.path == "/stop":
                    runtime.stop()
                    data = runtime.status()
                elif self.path == "/scene":
                    if not hasattr(runtime.transport, "configure_scene"):
                        raise ValueError("Scene controls are available in simulation")
                    if runtime.executor.status()["state"] not in ("disabled", "fault", "complete"):
                        raise ValueError("Stop the mission before resetting the scene")
                    runtime.submit(lambda: runtime.transport.configure_scene(payload))
                    data = runtime.status()
                elif self.path == "/actor":
                    if getattr(runtime.transport, "profile", "") != "sim":
                        raise ValueError("Actor controls are available in simulation")
                    xy = payload.get("position", [2, 0])
                    if len(xy) != 2 or not all(math.isfinite(float(v)) and abs(float(v)) <= 4 for v in xy):
                        raise ValueError("Keep the actor inside the simulated room")
                    if payload.get("kind") == "person":
                        runtime.submit(lambda: runtime.transport.set_person(*map(float, xy)))
                    elif payload.get("kind") == "barrier":
                        runtime.submit(lambda: runtime.transport.set_barrier(*map(float, xy)))
                    elif payload.get("kind") == "person_pose" and hasattr(runtime.transport, "set_person_pose"):
                        runtime.submit(lambda: runtime.transport.set_person_pose(payload.get("pose", "neutral")))
                    else:
                        raise ValueError("Unknown actor")
                    data = runtime.status()
                else:
                    self.respond(404, {"error": "Unknown route"})
                    return
                self.respond(200, data)
            except (ValueError, RuntimeError, TypeError, KeyError) as exc:
                runtime.stop()
                self.respond(409, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["sim", "ros-sim", "hardware"], default=os.getenv("GO2_PROFILE", "sim"))
    parser.add_argument("--host", default=os.getenv("GO2_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("GO2_HTTP_PORT", "8788")))
    args = parser.parse_args()
    if args.profile == "sim":
        from sim.transport import SimTransport
        transport_factory = SimTransport
    else:
        from showcase.transport import build_transport
        transport_factory = lambda: build_transport(args.profile)
    runtime = Runtime(transport_factory)
    server = ThreadingHTTPServer((args.host, args.port), handler(runtime))
    print(json.dumps({"event": "ready", "profile": args.profile, "url": f"http://{args.host}:{args.port}", "motion": "disabled"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()
