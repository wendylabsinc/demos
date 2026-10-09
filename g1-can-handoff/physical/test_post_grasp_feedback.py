"""No post-grasp visual gates, truthful completion, retained robot failures."""
import threading
import time
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from pickup import execute
from perception_policy import post_grasp_camera_checks
from test_pickup import Plant
os.environ.setdefault("HANDOFF_DATA", tempfile.mkdtemp(prefix="handoff-test-"))
from runtime import Runtime


class PostGraspFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.observation = SimpleNamespace(center=np.array([.35, .18, .811]), table_height=.75)
        self.prepared = (self.observation, {}, [self.observation.center.copy() for _ in range(4)], np.eye(3))

    def test_confirmed_grasp_finishes_without_any_post_grasp_frame_or_can_check(self):
        plant = Plant()
        grasped = False
        phases = []
        def observe(phase, *args, **kwargs):
            self.assertFalse(grasped, "Camera/depth/identity queried after grasp")
            phases.append(phase)
            return self.observation, {}
        def event(phase, **fields):
            nonlocal grasped
            if phase == "grasped":
                grasped = True
                # No visual rise/center/frame metadata and no further can or
                # contact validation are available after this exact boundary.
                plant.observe_can = Mock(side_effect=AssertionError("Can validation after grasp"))
                plant.contacts = Mock(side_effect=AssertionError("Contact revalidation after grasp"))
        result = execute(plant, self.prepared, observe, event, post_grasp_camera_checks=False)
        self.assertEqual(phases, ["pre-open", "pre-stage", "approach", "align", "reach", "pre-close"])
        self.assertEqual(plant.actions.count("move"), 4)
        self.assertEqual(plant.actions.count("hand"), 2)
        self.assertTrue(result["sequence_completed"])
        self.assertFalse(result["lift_verified"])
        self.assertIsNone(result["visual_rise_m"])
        self.assertIsNone(result["hold_verified_s"])

    def test_no_grasp_or_robot_failure_still_prevents_completion(self):
        for reason in ("missing grasp", "Operator cancelled trial", "body feedback stale", "winding temperature reaches guard"):
            with self.subTest(reason=reason):
                plant = Plant(contacts=("thumb",) if reason == "missing grasp" else ("thumb", "index"))
                def event(phase, **fields):
                    if phase == "grasped":
                        plant.move = Mock(side_effect=RuntimeError(reason))
                with self.assertRaises(RuntimeError):
                    execute(plant, self.prepared, lambda *a, **kw: (self.observation, {}), event,
                            post_grasp_camera_checks=False)

    def test_late_continuous_camera_failure_cannot_freeze_after_grasp(self):
        runtime = Runtime.__new__(Runtime)
        runtime.lock = threading.RLock()
        runtime.cancel = threading.Event()
        runtime.tracking_stop = threading.Event()
        runtime.post_grasp_perception_disabled = False
        runtime.grasp_started = False
        runtime.active = True
        runtime.phase = "close"
        runtime.plant = Mock()
        runtime.event = Mock()
        runtime.record_failure = Mock()
        entered = threading.Event()
        unblock = threading.Event()
        calls = 0
        def observe(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                entered.set()
                self.assertTrue(unblock.wait(2))
                raise RuntimeError("Camera disconnected after grasp")
            runtime.tracking_observation = SimpleNamespace(timestamp=time.monotonic())
            return runtime.tracking_observation, {}
        runtime.observe_frame = observe
        runtime.start_tracking({}, self.observation)
        self.assertTrue(entered.wait(2))
        runtime.finish_grasp_perception(runtime.plant)
        runtime.tracking_error = "stale result already queued"
        runtime.tracking_guard()  # The old guard reference must also become inert.
        unblock.set()
        runtime.tracking_thread.join(2)
        self.assertFalse(runtime.tracking_thread.is_alive())
        self.assertIsNone(runtime.plant.perception_guard)
        runtime.plant.freeze.assert_not_called()
        runtime.record_failure.assert_not_called()
        self.assertTrue(runtime.active)

    def test_configuration_is_explicit_and_validated(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(post_grasp_camera_checks())
        for value, expected in [("0", False), ("1", True)]:
            with patch.dict("os.environ", HANDOFF_POST_GRASP_CAMERA_CHECKS=value):
                self.assertIs(post_grasp_camera_checks(), expected)
        with patch.dict("os.environ", HANDOFF_POST_GRASP_CAMERA_CHECKS="true"):
            with self.assertRaises(ValueError):
                post_grasp_camera_checks()
