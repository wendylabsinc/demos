import dataclasses
import gzip
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from recorder import Recorder, plain, tracking


@dataclasses.dataclass
class Motor:
    q: float = 0.2
    motorstate: int = 7
    temperature: tuple = (41, 42)


@dataclasses.dataclass
class Hand:
    motor_state: list = dataclasses.field(
        default_factory=lambda: [Motor() for _ in range(7)]
    )
    error: tuple = (0, 4)


class Tests(unittest.TestCase):
    def test_nonfinite_is_explicit_and_json_is_strict(self):
        result = plain(Hand([Motor(float("nan"))]))
        self.assertEqual(result["motor_state"][0]["q"], {"nonfinite": "nan"})
        json.dumps(result, allow_nan=False)

    def test_complete_stream_fault_fields_and_sealed_hash(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = Recorder(root, min_free_mb=0)
            for _ in range(100):
                recorder.submit("rt/dex3/left/state", Hand())
            recorder.close()
            manifest = json.loads((recorder.root / "manifest.json").read_text())
            self.assertEqual(manifest["written"]["rt/dex3/left/state"], 100)
            self.assertFalse(manifest["recording"])
            self.assertFalse(manifest["dropped"])
            segment = manifest["closed_segments"][0]
            path = recorder.root / segment["file"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), segment["sha256"]
            )
            with gzip.open(path, "rt") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertTrue(any(row.get("type") == "recorder_status" for row in rows))
            rows = [row for row in rows if "topic" in row]
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["message"]["motor_state"][0]["motorstate"], 7)
            self.assertEqual(rows[0]["message"]["error"], [0, 4])
            self.assertIn("rt/lowstate", manifest["missing_or_stale_required"])

    def test_stale_command_does_not_create_tracking_evidence(self):
        state = {
            "topic": "rt/dex3/left/state",
            "monotonic_ns": 1_000_000_000,
            "message": {"motor_state": [{"q": 0.2}] * 7},
        }
        cmd = {"monotonic_ns": 900_000_000, "message": {"motor_cmd": [{"q": 0.1}] * 7}}
        self.assertAlmostEqual(
            tracking(state, cmd)["measured_minus_command_rad"]["0"], 0.1
        )
        cmd["monotonic_ns"] = 0
        self.assertIsNone(tracking(state, cmd))

    def test_disk_floor_stops_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = Recorder(root, min_free_mb=10**15)
            recorder.thread.join(3)
            self.assertTrue(recorder.stop.is_set())
            manifest = json.loads((recorder.root / "manifest.json").read_text())
            self.assertIn("Disk free space", manifest["error"])
            self.assertFalse(manifest["recording"])


if __name__ == "__main__":
    unittest.main()
