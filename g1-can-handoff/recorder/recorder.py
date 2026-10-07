"""Passive DDS flight recorder. No publishers, RPC clients, or motion API.

Records every message delivered to its callbacks; DDS/network loss before the
callback cannot be ruled out. Subscriber queueLen=0 avoids the SDK's silent
bounded callback queue; our own overflow counter is durable and visible.
"""
from __future__ import annotations

import dataclasses
import base64
import gzip
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import threading
import time
import uuid
try:
    import orjson
except ImportError:
    orjson = None
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TOPICS = {
    "rt/lowstate": "LowState_",
    "rt/dex3/left/state": "HandState_",
    "rt/dex3/right/state": "HandState_",
    "rt/lf/dex3/left/state": "HandState_",
    "rt/lf/dex3/right/state": "HandState_",
    "rt/arm_sdk": "LowCmd_",
    "rt/dex3/left/cmd": "HandCmd_",
    "rt/dex3/right/cmd": "HandCmd_",
}
REQUIRED = ["rt/lowstate", "rt/dex3/left/state", "rt/dex3/right/state"]
COMMAND_FOR_STATE = {
    "rt/lowstate": "rt/arm_sdk",
    "rt/dex3/left/state": "rt/dex3/left/cmd",
    "rt/dex3/right/state": "rt/dex3/right/cmd",
}


@lru_cache(maxsize=128)
def field_names(cls):
    return tuple(f.name for f in dataclasses.fields(cls))


def plain(value):
    kind = type(value)
    if kind is float:
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if kind in (str, int, bool, type(None)):
        return value
    if kind in (list, tuple, bytes, bytearray):
        return [plain(v) for v in value]
    if kind is dict:
        return {str(k): plain(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {name: plain(getattr(value, name)) for name in field_names(kind)}
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError(f"Unsupported SDK field type: {type(value).__name__}")


def encode_message(message):
    # Native dataclass traversal avoids Python per-field work starving DDS
    # callbacks. Fall back on explicit nonfinite tags whenever JSON null occurs;
    # orjson otherwise maps nonfinite floats to null, losing diagnostic detail.
    if orjson is None:
        return plain(message)
    encoded = orjson.dumps(message, default=plain)
    if b"null" in encoded:
        return plain(message)
    return orjson.loads(encoded)


def tracking(state, command):
    """Receive-time alignment, not proof of hardware command acceptance."""
    age = (state["monotonic_ns"] - command["monotonic_ns"]) / 1e9
    if not 0 <= age <= 0.2:
        return None
    measured = state["message"]["motor_state"]
    targets = command["message"]["motor_cmd"]
    indices = range(15, 29) if state["topic"] == "rt/lowstate" else range(7)
    errors = {}
    for i in indices:
        if i >= len(measured) or i >= len(targets):
            continue
        q, goal = measured[i].get("q"), targets[i].get("q")
        if isinstance(q, (float, int)) and isinstance(goal, (float, int)):
            errors[str(i)] = q - goal
    return {"command_age_s": age, "measured_minus_command_rad": errors,
            "alignment": "latest_observed_command_by_callback_time",
            "hardware_acceptance_verified": False}


class Recorder:
    def __init__(self, root, segment_s=60, min_free_mb=2048, max_session_mb=10240):
        self.session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        self.root = Path(root) / self.session
        self.root.mkdir(parents=True)
        self.segment_s, self.min_free = segment_s, min_free_mb * 1024**2
        self.max_bytes = max_session_mb * 1024**2
        self.queue = queue.Queue(maxsize=8192)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.received, self.written, self.dropped = Counter(), Counter(), Counter()
        self.latest_ns, self.last_written_ns, self.commands, self.fault_codes = {}, {}, {}, {}
        self.error, self.segments, self.current_file = None, [], None
        self.started = time.monotonic()
        self.sequence = 0
        self.thread = threading.Thread(target=self.write_loop, name="archive-writer", daemon=True)
        self.thread.start()

    def submit(self, topic, message):
        # SDK creates a new dataclass per received DDS sample. Ownership of that
        # object passes to this recorder; encoding happens in the writer thread.
        wall, mono = time.time_ns(), time.monotonic_ns()
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
            self.received[topic] += 1
            self.latest_ns[topic] = mono
            if self.stop.is_set() or self.error:
                self.dropped[topic] += 1
                return
        try:
            self.queue.put_nowait((topic, message, wall, mono, sequence))
        except queue.Full:
            with self.lock:
                self.dropped[topic] += 1

    def status(self):
        now = time.monotonic_ns()
        with self.lock:
            ages = {t: (now - self.latest_ns[t]) / 1e9 if t in self.latest_ns else None for t in TOPICS}
            return {
                "schema": "g1-handoff-recorder/1", "session": self.session,
                "read_only": True, "motion_commands_sent": 0,
                "recording": not self.stop.is_set() and self.error is None,
                "error": self.error, "uptime_s": time.monotonic() - self.started,
                "received": dict(self.received), "written": dict(self.written),
                "dropped": dict(self.dropped), "queue_depth": self.queue.qsize(),
                "age_s": ages, "missing_or_stale_required": [t for t in REQUIRED if ages[t] is None or ages[t] > 0.25],
                "last_written_age_s": {t: (now-ns)/1e9 for t, ns in self.last_written_ns.items()},
                "latest_nonzero_raw_fault_fields": dict(self.fault_codes),
                "raw_faults_live_decoded": False,
                "storage_codec": "dds-cdr-base64/1; decode using export_cdr.py",
                "current_file": self.current_file, "closed_segments": list(self.segments),
                "coverage": "all callback-delivered messages; upstream DDS loss is not proven absent",
                "application_phase_and_exceptions": "not available: handoff hardware controller is absent",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "interface": os.getenv("RECORDER_INTERFACE", "enP8p1s0"),
                "domain": int(os.getenv("RECORDER_DOMAIN", "0")),
            }

    def save_manifest(self):
        temp = self.root / "manifest.tmp"
        temp.write_text(json.dumps(self.status(), indent=2) + "\n")
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        temp.replace(self.root / "manifest.json")

    def write_loop(self):
        handle, path = None, None
        opened, flushed = 0, 0
        try:
            while not self.stop.is_set() or not self.queue.empty():
                now = time.monotonic()
                if handle is None or now - opened >= self.segment_s:
                    if handle is not None:
                        self.seal(handle, path)
                    path = self.root / f"samples-{len(self.segments):06d}.jsonl.gz"
                    handle = gzip.open(path, "wt", compresslevel=1)
                    self.current_file = str(path)
                    opened = now
                try:
                    topic, msg, wall, mono, sequence = self.queue.get(timeout=0.1)
                except queue.Empty:
                    topic = None
                if topic:
                    row = {"topic": topic, "unix_ns": wall, "monotonic_ns": mono,
                           "receive_sequence": sequence}
                    if isinstance(msg, bytes):
                        row["storage_codec"] = "dds-cdr-base64/1"
                        row["cdr_base64"] = base64.b64encode(msg).decode("ascii")
                    else:
                        row["message"] = encode_message(msg)
                        if TOPICS[topic].endswith("Cmd_"):
                            self.commands[topic] = row
                        command = self.commands.get(COMMAND_FOR_STATE.get(topic))
                        if command:
                            row["tracking"] = tracking(row, command)
                        motors = row["message"].get("motor_state", [])
                        faults = {str(i): {k: m[k] for k in ("motorstate", "lost") if m.get(k) not in (None, 0)}
                                  for i, m in enumerate(motors)}
                        faults = {i: v for i, v in faults.items() if v}
                        if any(row["message"].get("error", [])):
                            faults["message_error"] = row["message"]["error"]
                        if motors:
                            with self.lock:
                                self.fault_codes[topic] = faults
                    handle.write((orjson.dumps(row).decode() if orjson is not None else json.dumps(row, separators=(",", ":"), allow_nan=False)) + "\n")
                    with self.lock:
                        self.written[topic] += 1
                        self.last_written_ns[topic] = mono
                    self.queue.task_done()
                if now - flushed >= 1:
                    handle.write(json.dumps({"type": "recorder_status", "unix_ns": time.time_ns(),
                                             "status": self.status()}, separators=(",", ":")) + "\n")
                    handle.flush()
                    with path.open("rb") as raw:
                        os.fsync(raw.fileno())
                    self.save_manifest()
                    if shutil.disk_usage(self.root).free < self.min_free:
                        raise OSError("Disk free space below configured recorder floor")
                    if sum(p.stat().st_size for p in self.root.glob("*.gz")) > self.max_bytes:
                        raise OSError("Session storage limit reached; prior evidence retained")
                    flushed = now
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.stop.set()
            print(json.dumps({"recorder_error": self.error}), flush=True)
        finally:
            if handle is not None:
                try:
                    handle.write(json.dumps({"type": "recorder_status", "unix_ns": time.time_ns(), "status": self.status()}) + "\n")
                    self.seal(handle, path)
                except Exception as exc:
                    self.error = self.error or f"seal: {exc}"
            try:
                self.save_manifest()
            except Exception as exc:
                print(json.dumps({"manifest_error": str(exc)}), flush=True)

    def seal(self, handle, path):
        handle.close()
        with path.open("rb") as raw:
            os.fsync(raw.fileno())
            hasher = hashlib.sha256()
            for block in iter(lambda: raw.read(1024 * 1024), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        with self.lock:
            self.segments.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})

    def close(self):
        self.stop.set()
        self.thread.join(timeout=30)


def main():
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.sub import DataReader, ddspy_take
    from cyclonedds.core import Qos, Policy
    from unitree_sdk2py.core.channel import ChannelConfigHasInterface
    from unitree_sdk2py.idl.unitree_hg.msg import dds_
    interface = os.getenv("RECORDER_INTERFACE", "enP8p1s0")
    domain_id = int(os.getenv("RECORDER_DOMAIN", "0"))
    # Pinned CycloneDDS 0.10.2 native binding returns original CDR buffers.
    # Avoid DataReader.take(), which deserializes every sample in Python.
    domain = Domain(domain_id, ChannelConfigHasInterface.replace('$__IF_NAME__$', interface))
    participant = DomainParticipant(domain_id)
    qos = Qos(Policy.History.KeepLast(4096), Policy.Reliability.BestEffort)
    readers = {name: DataReader(participant, Topic(participant, name, getattr(dds_, typ)), qos=qos)
               for name, typ in TOPICS.items()}
    recorder = Recorder(os.getenv("RECORDER_ROOT", "/records"),
                        float(os.getenv("RECORDER_SEGMENT_SECONDS", "60")),
                        float(os.getenv("RECORDER_MIN_FREE_MB", "2048")),
                        float(os.getenv("RECORDER_MAX_SESSION_MB", "10240")))
    capture_stop = threading.Event()
    def capture():
        try:
            while not capture_stop.is_set():
                for name, reader in readers.items():
                    result = ddspy_take(reader._ref, 128)
                    if isinstance(result, int):
                        raise RuntimeError(f"DDS raw take failed for {name}: {result}")
                    for data, info in result:
                        if info.valid_data:
                            recorder.submit(name, bytes(data))
                capture_stop.wait(0.002)
        except Exception as exc:
            recorder.error = f"raw capture: {type(exc).__name__}: {exc}"
            recorder.stop.set()
    capture_thread = threading.Thread(target=capture, name="raw-dds-capture", daemon=True)
    capture_thread.start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/status"):
                self.send_error(404)
                return
            body = json.dumps(recorder.status(), indent=2).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", int(os.getenv("RECORDER_PORT", "8101"))), Handler)
    def shutdown(*args):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(json.dumps({"started": recorder.session, "topics": TOPICS, "read_only": True}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        capture_stop.set()
        capture_thread.join(timeout=3)
        recorder.close()
        server.server_close()


if __name__ == "__main__":
    main()
