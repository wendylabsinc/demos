"""Portable encoded RGB-D/HTTP YOLO replay across semantic-to-visual tracking.

The synthetic fixture is a grey cylinder, deliberately independent of red
color detection. Hardware frames and timing regressions live in other tests.
"""

import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from camera_input import geometry
from target_tracker import TargetTracker
import test_camera_pipeline as pipeline
from test_yolo_coke import FrameFixture
from yolo_client import MODEL_SHA256, SCHEMA, identity


class ReplayPerceptionTests(FrameFixture, unittest.TestCase):
    def setUp(self):
        fixture = self.frame()
        self.clock = 100.0
        self.http = pipeline.FakeRGBDHTTP(
            fixture["color_bgr"], fixture["depth_m"], fixture["rays"], "nx2-test"
        )
        self.http.advance(201, 99.95)
        self.camera = self.http.camera()
        self.tracker = TargetTracker()
        self.requests = []
        self.boxes = [
            dict(
                xyxy=[60, 25, 100, 85],
                confidence=0.11,
                class_id=0,
                class_name="Coca-Cola",
            )
        ]
        self.response_change = lambda result: result

        def yolo(request):
            source = json.loads(request.get_header("X-frame-identity"))
            raw = np.frombuffer(request.data, dtype=np.uint8).reshape(source["shape"])
            self.requests.append((copy.deepcopy(source), raw.copy()))
            result = dict(
                schema=SCHEMA,
                source=source,
                model_sha256=MODEL_SHA256,
                backend="pytorch_cuda",
                truncated=False,
                detections=copy.deepcopy(self.boxes),
            )
            return json.dumps(self.response_change(result)).encode()

        self.http.yolo = yolo
        self.network = patch("urllib.request.urlopen", side_effect=self.http.urlopen)
        self.network.start()
        self.timer = patch(
            "camera_input.time.monotonic", side_effect=lambda: self.clock
        )
        self.timer.start()

    def tearDown(self):
        self.camera.close()
        self.network.stop()
        self.timer.stop()

    def observe(self):
        frame = self.camera.frame(materialize=False)
        observation, vision = geometry(
            frame, self.config(), require_table=False, tracker=self.tracker
        )
        self.tracker.accept(observation, vision, frame)
        return frame, observation, vision

    def next_image(self, color=None):
        self.clock += 0.1
        self.http.advance(self.http.fid + 1, self.clock - 0.05, color)

    def test_decoded_camera_pixels_and_exposure_bind_every_uncertainty_recheck(self):
        first, _, initial = self.observe()
        self.assertEqual(initial["tracking"]["reason"], "initial_identity")
        self.next_image()
        _, _, visual = self.observe()
        self.assertEqual(visual["tracking"]["mode"], "visual_depth")
        self.assertEqual(len(self.requests), 1)
        changed = self.http.color.copy()
        changed[25:85, 60:100] = [0, 255, 0]
        self.next_image(changed)
        frame, observation, rechecked = self.observe()
        self.assertEqual(rechecked["tracking"]["mode"], "yolo")
        self.assertEqual(
            rechecked["tracking"]["semantic_frame_id"], frame["meta"]["frame_id"]
        )
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1][0], identity(frame))
        np.testing.assert_array_equal(self.requests[-1][1], frame["color_bgr"])
        np.testing.assert_allclose(observation.center[:2], [0.35, 0.18], atol=1e-6)
        self.assertNotEqual(frame["timestamp"], first["timestamp"])

    def test_uncertain_camera_view_with_absent_yolo_keeps_prior_identity_uncommitted(
        self,
    ):
        self.observe()
        previous = self.tracker.state
        changed = self.http.color.copy()
        changed[25:85, 60:100] = [0, 255, 0]
        self.next_image(changed)
        self.boxes = []
        with self.assertRaisesRegex(RuntimeError, "absent/ambiguous"):
            self.observe()
        self.assertIs(self.tracker.state, previous)
        self.assertIsNone(self.tracker.pending)
        self.assertEqual(self.requests[-1][0]["frame_id"], self.http.fid)

    def test_replayed_previous_yolo_response_cannot_replace_current_decoded_frame(self):
        self.observe()
        old_source = copy.deepcopy(self.requests[-1][0])
        previous = self.tracker.state
        changed = self.http.color.copy()
        changed[25:85, 60:100] = [0, 255, 0]
        self.next_image(changed)

        def stale_identity(result):
            result["source"] = old_source
            return result

        self.response_change = stale_identity
        with self.assertRaisesRegex(RuntimeError, "exact RGB-D"):
            self.observe()
        self.assertIs(self.tracker.state, previous)
        self.assertIsNone(self.tracker.pending)


if __name__ == "__main__":
    unittest.main()
