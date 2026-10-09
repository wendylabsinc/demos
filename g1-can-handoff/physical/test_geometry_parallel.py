"""Scheduled exact-frame YOLO overlaps world work without extra inference."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from camera_input import geometry
import test_target_tracker as tracker_tests
from test_yolo_coke import FrameFixture
from yolo_client import identity


class ParallelGeometryTests(FrameFixture, unittest.TestCase):
    setUp = tracker_tests.TrackerTests.setUp
    next_frame = tracker_tests.TrackerTests.next_frame
    select = tracker_tests.TrackerTests.select
    accept = tracker_tests.TrackerTests.accept
    seed = tracker_tests.TrackerTests.seed

    def check_overlap(self, periodic):
        if periodic:
            self.seed()
            self.tracker.state["semantic_timestamp"] -= 1.01
        frame = self.next_frame()
        original = self.detector.detect
        started = threading.Event()
        transformed = threading.Event()
        test = self

        class CheckedRays(np.ndarray):
            def __mul__(self, other):
                test.assertTrue(
                    started.wait(0.2), "YOLO must start before world transform"
                )
                transformed.set()
                return super().__mul__(other).view(np.ndarray)

        frame["rays"] = frame["rays"].view(CheckedRays)

        def detect(exact_frame):
            self.assertIs(exact_frame, frame)
            started.set()
            self.assertTrue(
                transformed.wait(0.2), "world work must run while inference waits"
            )
            return original(exact_frame)

        before = self.detector.calls
        with patch.object(self.detector, "detect", side_effect=detect):
            observation, vision = geometry(
                frame, self.config(), require_table=False, tracker=self.tracker
            )
        self.assertEqual(self.detector.calls, before + 1)
        self.assertEqual(
            vision["tracking"]["reason"],
            "periodic_recheck" if periodic else "initial_identity",
        )
        self.assertEqual(vision["image_sha256"], identity(frame)["image_sha256"])
        np.testing.assert_allclose(observation.center[:2], [0.35, 0.18], atol=1e-6)

    def test_initial_semantic_runs_in_parallel(self):
        self.check_overlap(False)

    def test_periodic_semantic_runs_in_parallel(self):
        self.check_overlap(True)

    def test_strong_visual_track_geometry_does_not_call_yolo(self):
        self.seed()
        frame = self.next_frame()
        before = self.detector.calls
        _, vision = geometry(
            frame, self.config(), require_table=False, tracker=self.tracker
        )
        self.assertEqual(vision["tracking"]["mode"], "visual_depth")
        self.assertEqual(self.detector.calls, before)

    def test_parallel_response_keeps_exact_frame_validation(self):
        frame = self.next_frame()
        original = self.detector.detect

        def wrong_image(exact_frame):
            result = original(exact_frame)
            result["source"]["image_sha256"] = "wrong"
            return result

        with patch.object(self.detector, "detect", side_effect=wrong_image):
            with self.assertRaisesRegex(RuntimeError, "exact RGB-D"):
                geometry(
                    frame, self.config(), require_table=False, tracker=self.tracker
                )
        self.assertIsNone(self.tracker.pending)


if __name__ == "__main__":
    unittest.main()
