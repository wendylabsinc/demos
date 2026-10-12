"""Local operator release button through the native Wendy MCP transport.

Run on the operator Mac: python3 operator_panel.py. No motion starts on load.
"""

import json
from operator_state import describe_state, ERROR_CASES
from placement_overlay import PlacementGuide
from pathlib import Path
import queue
import secrets
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE = "unitree-g1-nx-2.local:50052"
PORT = 8197
DEPLOYING = Path("/tmp/wendy-g1-nx2-deployment.lock")
TOKEN = secrets.token_urlsafe(32)

# Keep the browser layout/behavior readable separately from the transport.
PAGE = (
    Path(__file__)
    .with_name("operator_ui.html")
    .read_text(encoding="utf-8")
    .replace("ERROR_CASES_JSON", json.dumps(ERROR_CASES))
    .replace(
        "UI_VERSION_VALUE",
        json.loads(Path(__file__).with_name("wendy.json").read_text())["version"],
    )
)


class Wendy:
    """Serialize fixed operator requests through the native Wendy MCP transport."""

    def __init__(self):
        self.process = subprocess.Popen(
            ["wendy", "mcp", "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.sequence = 0
        threading.Thread(target=self.read, daemon=True).start()
        try:
            self.rpc(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "g1-operator-release", "version": "1"},
                },
            )
            self.process.stdin.write(
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
                + "\n"
            )
            self.process.stdin.flush()
            self.tool("wendy_status", {})
            self.tool("device_connect", {"device": DEVICE})
        except Exception:
            self.process.terminate()
            raise

    def read(self):
        for line in self.process.stdout:
            try:
                self.responses.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    def rpc(self, method, params):
        self.sequence += 1
        ident = self.sequence
        self.process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
            )
            + "\n"
        )
        self.process.stdin.flush()
        while True:
            row = self.responses.get(timeout=30)
            if row.get("id") == ident:
                if "error" in row:
                    raise RuntimeError(
                        row["error"].get("message", "Wendy transport failed")
                    )
                return row["result"]

    def tool(self, name, arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(str(result.get("content")))
        return result

    def request(self, release=False, *, action=None):
        # Fixed target, fixed app, fixed endpoints: never accept command text.
        action = "release" if release else (action or "status")
        paths = {
            "status": "status",
            "release": "release",
            "start": "demo/start",
            "stop": "stop",
        }
        if action not in paths:
            raise ValueError("Unknown operator action")
        if action in ("start", "start_force") and DEPLOYING.is_file():
            raise RuntimeError(
                "Demo update in progress; Start will be available after verification"
            )
        data = (
            ",data=b'{\"can_supported\":true}',headers={'Content-Type':'application/json'}"
            if action == "release"
            else (
                ",data=b'{}',headers={'Content-Type':'application/json'}"
                if action != "status"
                else ""
            )
        )
        script = (
            """import json,math,time,urllib.request,urllib.error
url='http://127.0.0.1:8102/"""
            + paths[action]
            + """'
request=urllib.request.Request(url"""
            + data
            + """)
try:
 response=urllib.request.urlopen(request,timeout=12);s=json.load(response)
 out={k:s.get(k) for k in ['build_info','controller_mode','act_policy','act_policy_parity','act_inference_budget_ms','pose_alignment_max_skew_ms','act_training_placement_enforced','phase','error','motion_commands_sent','adapter_initialized','telemetry_healthy','telemetry_error','camera_max_age_ms','yolo_confidence_strictly_above','target_identity','yolo_required','demo_running','demo_result','demo_error']}
 out['hand_feedback']={}
 for side in ['left','right']:
  sample=(s.get('measured') or {}).get(side) or {}
  received=sample.get('time')
  age_ms=(time.monotonic()-received)*1000 if isinstance(received,(int,float)) and math.isfinite(received) else None
  out['hand_feedback'][side]={'faults':sample.get('faults'),'temperatures':sample.get('temperatures'),'age_ms':age_ms}
 def finite_json(value):
  if isinstance(value,float) and not math.isfinite(value):return None
  if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
  if isinstance(value,list):return [finite_json(v) for v in value]
  return value
 print(json.dumps(finite_json(out),allow_nan=False))
except urllib.error.HTTPError as e:
 print(json.dumps({'error':json.load(e).get('error')}))
"""
        )
        with self.lock:
            result = self.tool(
                "container_exec",
                {
                    "app_name": "DEV-g1-handoff-physical",
                    "command": ["python3", "-c", script],
                    "timeout_seconds": 20,
                    "max_bytes": 8000,
                },
            )
        payload = result.get("structuredContent")
        if payload is None:
            payload = json.loads(
                next(c["text"] for c in result["content"] if c["type"] == "text")
            )
        if payload.get("exit_code") != 0 or payload.get("truncated"):
            raise RuntimeError(payload.get("stderr") or "Controller unavailable")
        state = json.loads(payload["stdout"])
        state["deployment_in_progress"] = DEPLOYING.is_file()
        state["presentation"] = describe_state(state)
        if state.get("error") and "phase" not in state:
            raise RuntimeError(state["error"])
        if action == "release" and (
            state.get("phase") != "PASSIVE" or state.get("adapter_initialized")
        ):
            raise RuntimeError(
                state.get("error") or "Release not confirmed; controller retains hold"
            )
        return state


class Handler(BaseHTTPRequestHandler):
    """Serve the local page and validate each explicit Start, Stop or Release."""

    def send(self, data, status=200, kind="application/json"):
        body = data.encode() if isinstance(data, str) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def valid(self):
        return (
            self.headers.get("Host") == f"127.0.0.1:{PORT}"
            and self.headers.get("X-Operator-Token") == TOKEN
            and self.headers.get("Origin", f"http://127.0.0.1:{PORT}")
            == f"http://127.0.0.1:{PORT}"
        )

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{PORT}":
            self.send({"error": "Local operator only"}, 403)
            return
        if self.path == "/":
            self.send(
                PAGE.replace("TOKEN_VALUE", TOKEN), kind="text/html; charset=utf-8"
            )
            return
        if (
            self.path not in ("/status", "/camera-health", "/placement")
            or not self.valid()
        ):
            self.send({"error": "Invalid operator request"}, 403)
            return
        try:
            if self.path == "/placement":
                guide = getattr(self.server, "placement", None)
                self.send(
                    guide.read()
                    if guide is not None
                    else {"valid": False, "error": "Placement guide unavailable"}
                )
            elif self.path == "/camera-health":
                with urllib.request.urlopen(
                    "http://10.10.20.238:8001/health", timeout=2
                ) as response:
                    health = json.load(response)
                self.send(
                    {
                        k: health.get(k)
                        for k in ["healthy", "frame_id", "frame_age_s", "fps"]
                    }
                )
            else:
                self.send(self.server.wendy.request())
        except Exception as exc:
            self.send({"error": str(exc)}, 503)

    def do_POST(self):
        if self.path not in ("/release", "/start", "/stop") or not self.valid():
            self.send({"error": "Invalid operator request"}, 403)
            return
        if self.path == "/start" and DEPLOYING.is_file():
            self.send(
                {
                    "error": "Demo update in progress; Start will be available after verification"
                },
                409,
            )
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 1024:
                raise ValueError("Invalid operator request")
            data = json.loads(self.rfile.read(size))
            if self.path == "/release":
                if data.get("can_supported") is not True:
                    raise ValueError("Support must be confirmed before release")
                self.send(self.server.wendy.request(release=True))
            else:
                self.send(self.server.wendy.request(action=self.path[1:]))
        except Exception as exc:
            self.send({"error": str(exc)}, 409)

    def log_message(self, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.wendy = Wendy()
    server.placement = PlacementGuide(Wendy).start()
    print(f"G1 operator panel: http://127.0.0.1:{PORT}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.placement.close()
        server.wendy.process.terminate()
        server.server_close()


if __name__ == "__main__":
    main()
