"""Replay the real 00:59 trial's periodic-YOLO gap at the motor write gate."""

import gzip
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from physical_core import PhysicalPlant
from runtime import Runtime

ROOT = Path(__file__).parent / "offline/evidence/20261008T005915Z"


class LatestTrialTests(unittest.TestCase):
    def test_latest_rgb_depth_gap_fits_500ms_without_disabling_uncertainty_or_future_guard(
        self,
    ):
        root = (
            Path(__file__).parent
            / "offline/evidence/exposure-failure-20261008-preserved/trial-01"
        )
        import hashlib

        manifest = json.loads((root / "manifest.json").read_text())
        for name, metadata in manifest["files"].items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(),
                metadata["sha256"],
            )
        rows = [
            json.loads(line)
            for line in gzip.decompress(
                (root / "events.jsonl.gz").read_bytes()
            ).splitlines()
        ]
        failure = next(r for r in rows if r["phase"] == "primary_failure")
        results = [
            r for r in rows if r["phase"] == "perception_result" and r.get("geometry")
        ]
        before = [r for r in results if r["monotonic_ns"] < failure["monotonic_ns"]][-1]
        late = [r for r in results if r["monotonic_ns"] > failure["monotonic_ns"]][0]
        self.assertEqual((before["frame_id"], late["frame_id"]), (41858, 41860))
        meta = next(
            r["metadata"]
            for r in rows
            if r["phase"] == "phase_camera_evidence"
            and r["metadata"]["frame_id"] == before["frame_id"]
        )
        self.assertLess(abs(meta["color_time"] - meta["depth_time"]), 0.001)
        timestamp = meta["depth_time"]
        stop_at = failure["monotonic_ns"] / 1e9
        self.assertGreater(stop_at - timestamp, 0.4)
        self.assertLess(stop_at - timestamp, 0.5)
        self.assertLess(late["monotonic_ns"] / 1e9 - timestamp, 0.5)
        runtime = Runtime.__new__(Runtime)
        runtime.lock = threading.RLock()
        runtime.tracking_error = None
        runtime.tracking_observation = SimpleNamespace(timestamp=timestamp)
        with (
            patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "400"}),
            patch("runtime.time.monotonic", return_value=stop_at),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracking exposure stale"):
                runtime.tracking_guard()
        with patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "500"}):
            with patch("runtime.time.monotonic", return_value=stop_at):
                runtime.tracking_guard()
            with patch("runtime.time.monotonic", return_value=timestamp + 0.5001):
                with self.assertRaisesRegex(RuntimeError, "tracking exposure stale"):
                    runtime.tracking_guard()
            runtime.tracking_error = "uncertain appearance"
            with patch("runtime.time.monotonic", return_value=stop_at):
                with self.assertRaisesRegex(RuntimeError, "tracking uncertain"):
                    runtime.tracking_guard()

    def test_recorded_recheck_arrives_after_timeout_and_cannot_resume_motor_writes(
        self,
    ):
        rows = [
            json.loads(line)
            for line in gzip.decompress(
                (ROOT / "events.jsonl.gz").read_bytes()
            ).splitlines()
        ]
        failure = next(e for e in rows if e["phase"] == "primary_failure")
        self.assertIn("401.7 ms", failure["error"])
        results = [
            e for e in rows if e["phase"] == "perception_result" and e.get("geometry")
        ]
        before = [e for e in results if e["monotonic_ns"] < failure["monotonic_ns"]][-1]
        late = [e for e in results if e["monotonic_ns"] > failure["monotonic_ns"]][0]
        self.assertEqual((before["frame_id"], late["frame_id"]), (62257, 62259))
        self.assertEqual(late["geometry"]["tracking"]["reason"], "periodic_recheck")
        self.assertLess(late["exposure_age_ms"], 400.0)
        self.assertLess((late["monotonic_ns"] - failure["monotonic_ns"]) / 1e6, 12.0)
        camera = next(
            e
            for e in rows
            if e["phase"] == "phase_camera_evidence"
            and e["metadata"]["frame_id"] == before["frame_id"]
        )
        timestamp = camera["metadata"]["depth_time"]
        stop_at = failure["monotonic_ns"] / 1e9
        self.assertGreater((stop_at - timestamp) * 1000, 400.0)
        runtime = Runtime.__new__(Runtime)
        runtime.lock = threading.RLock()
        runtime.tracking_error = None
        runtime.tracking_observation = SimpleNamespace(timestamp=timestamp)
        plant = PhysicalPlant.__new__(PhysicalPlant)
        plant.cancelled = False
        plant.cancel = threading.Event()
        plant.perception_guard = runtime.tracking_guard
        plant.command_lock = threading.RLock()
        plant.clock = time
        plant.command_arm = Mock()
        plant.sender = Mock()
        plant.read = Mock()
        with (
            patch.dict("os.environ", HANDOFF_CAMERA_MAX_AGE_MS="400"),
            patch("runtime.time.monotonic", return_value=before["monotonic_ns"] / 1e9),
        ):
            runtime.tracking_guard()
        # This is the historical 400 ms stop, independent of the runner's
        # current deployment budget (500 ms in the today profile).
        with (
            patch.dict("os.environ", HANDOFF_CAMERA_MAX_AGE_MS="400"),
            patch("runtime.time.monotonic", return_value=stop_at),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracking exposure stale"):
                plant._execute_poses(np.zeros(7), [np.zeros(7)])
        # Freeze latches cancellation. A late, individually fresh result
        # cannot clear that latch and resume either arm or finger writes.
        plant.cancelled = True
        camera = next(
            e
            for e in rows
            if e["phase"] == "phase_camera_evidence"
            and e["metadata"]["frame_id"] == late["frame_id"]
        )
        runtime.tracking_observation = SimpleNamespace(
            timestamp=camera["metadata"]["depth_time"]
        )
        with patch("runtime.time.monotonic", return_value=late["monotonic_ns"] / 1e9):
            runtime.tracking_guard()
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                plant._execute_poses(np.zeros(7), [np.zeros(7)])
        plant.command_arm.assert_not_called()
        plant.sender.ctrl_dual_hand.assert_not_called()
        plant.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
