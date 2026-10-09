"""Fresh accepted tracking is required before the next physical command."""

import os
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("HANDOFF_DATA", tempfile.mkdtemp(prefix="tracking-test-"))
from runtime import Runtime
from physical_core import PhysicalPlant


class MotionTrackingTests(unittest.TestCase):
    def test_recorded_reach_settle_failure_reproduces_with_no_contact_inference(self):
        fixture = Path(__file__).parent / "offline/fixtures/reach-settle-20261009.json"
        recorded = json.loads(fixture.read_text())
        failure = next(
            e for e in recorded["events"] if e["phase"] == "arm_settle_failed"
        )
        goal = np.asarray(failure["goal_q"])
        measured = np.asarray(failure["measured_q"])
        np.testing.assert_allclose(measured - goal, failure["error_rad"])
        self.assertEqual(recorded["last_target"]["phase_name"], "reach")
        self.assertAlmostEqual(
            float(np.max(np.abs(measured - goal))), 0.07511741998107052
        )

        # Drive the actual settling loop with the captured feedback. No SDK or
        # physics world is involved; this reproduces the guard, not contact.
        class Clock:
            now = 0.0

            def monotonic(self):
                return self.now

            def sleep(self, seconds):
                self.now += seconds

        plant = PhysicalPlant.__new__(PhysicalPlant)
        plant.clock = Clock()
        plant.command_lock = threading.RLock()
        plant.check = Mock(return_value={"body": {"q": [0.0] * 15 + measured.tolist()}})
        plant.command_arm = Mock()
        plant.event = Mock()
        plant.sync = Mock()
        plant.arm_tau_command = np.asarray(failure["gravity_feedforward_nm"])
        with self.assertRaisesRegex(
            RuntimeError, "Physical arm did not settle at the IK target"
        ):
            plant._execute_poses(goal, [])
        self.assertGreaterEqual(plant.clock.now, 2.0)
        self.assertEqual(plant.event.call_args.args, ("arm_settle_failed",))
        np.testing.assert_allclose(
            plant.event.call_args.kwargs["error_rad"], failure["error_rad"]
        )
        plant.sync.assert_not_called()

    def runtime(self, age=0.2, error=None):
        runtime = Runtime.__new__(Runtime)
        runtime.lock = threading.RLock()
        runtime.tracking_error = error
        runtime.tracking_observation = SimpleNamespace(timestamp=100 - age)
        return runtime

    def test_fresh_observation_passes_and_stale_future_or_uncertain_blocks(self):
        with patch("runtime.time.monotonic", return_value=100):
            self.runtime(0.399).tracking_guard()
            for age, error in [
                (0.401, None),
                (-0.001, None),
                (0.1, "identity uncertain"),
            ]:
                with (
                    self.subTest(age=age, error=error),
                    self.assertRaises(RuntimeError),
                ):
                    self.runtime(age, error).tracking_guard()

    def test_stale_tracking_blocks_arm_and_finger_writes_at_next_command_tick(self):
        runtime = self.runtime(0.401)
        plant = PhysicalPlant.__new__(PhysicalPlant)
        plant.cancelled = False
        plant.cancel = threading.Event()
        plant.perception_guard = runtime.tracking_guard
        plant.clock = time
        plant.command_lock = threading.RLock()
        plant.read = Mock()
        plant.command_arm = Mock()
        plant.sender = Mock()
        with (
            patch("runtime.time.monotonic", return_value=100),
            self.assertRaisesRegex(RuntimeError, "stale"),
        ):
            plant._execute_poses(np.zeros(7), [np.zeros(7)])
        plant.command_arm.assert_not_called()
        plant.sender.ctrl_dual_hand.assert_not_called()
        plant.read.assert_not_called()

    def test_synchronous_phase_observation_updates_watchdog_after_full_validation(self):
        runtime = self.runtime(0.39)
        runtime.observation_lock = threading.RLock()
        runtime.tracking_baseline = SimpleNamespace(center=np.array([0.3, 0.15, 0.823]))
        runtime.pose_history = Mock()
        runtime.event = Mock()
        observation = SimpleNamespace(
            center=np.array([0.301, 0.15, 0.823]), timestamp=99.9
        )
        runtime.perception = Mock()
        runtime.perception.observe.return_value = (observation, {"frame_id": 2})
        with patch("runtime.time.monotonic", return_value=100):
            self.assertEqual(
                runtime.observe_frame({}, "approach", True)[0], observation
            )
            self.assertIs(runtime.tracking_observation, observation)
            runtime.tracking_guard()
            for center, timestamp in [
                (np.array([0.32, 0.15, 0.823]), 99.95),
                (observation.center, 99.5),
                (observation.center, 99.9),
            ]:
                runtime.perception.observe.return_value = (
                    SimpleNamespace(center=center, timestamp=timestamp),
                    {},
                )
                with self.assertRaises(RuntimeError):
                    runtime.observe_frame({}, "align", True)
                self.assertIs(runtime.tracking_observation, observation)
            runtime.tracking_error = "Identity lost"
            runtime.perception.observe.return_value = (
                SimpleNamespace(center=observation.center, timestamp=99.99),
                {},
            )
            with self.assertRaisesRegex(RuntimeError, "Identity lost"):
                runtime.observe_frame({}, "align", True)
            self.assertIs(runtime.tracking_observation, observation)

    def test_one_frame_prefetch_overlaps_processing_without_rewriting_exposure(self):
        runtime = self.runtime()
        runtime.tracking_stop = threading.Event()
        runtime.tracking_thread = None
        runtime.pending_frame = None
        runtime.frame_prefetch = ThreadPoolExecutor(max_workers=1)
        second_started = threading.Event()
        frames = []

        def capture(**kwargs):
            frame = {"frame_id": len(frames) + 1, "timestamp": 90 + len(frames) * 0.1}
            frames.append(frame)
            if len(frames) == 2:
                second_started.set()
            return frame

        runtime.camera = SimpleNamespace(frame=capture)
        try:
            first = runtime.next_tracking_frame()
            self.assertTrue(second_started.wait(1))
            self.assertIs(first, frames[0])
            self.assertEqual(first["timestamp"], 90)
            self.assertIs(runtime.next_tracking_frame(), frames[1])
        finally:
            runtime.stop_tracking()
        self.assertIsNone(runtime.pending_frame)
        self.assertIsNone(runtime.frame_prefetch)


if __name__ == "__main__":
    unittest.main()
