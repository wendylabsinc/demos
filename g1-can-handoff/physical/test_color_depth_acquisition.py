"""The two exact initial-YOLO failure frames, replayed without a classifier.

Archived exposure times are retained. Only the test clock is pinned; these
fixtures never authorize a live command or claim present camera freshness.
"""

import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
from PIL import Image

from camera_input import geometry
from target_tracker import TargetTracker

ROOT = Path(__file__).parent / "offline/evidence/low-yolo-20261008"


def recorded_frame(number):
    with zipfile.ZipFile(ROOT / f"camera-{number}-planning.source.zip") as archive:
        metadata = json.loads(archive.read("metadata.json"))
        color = np.array(
            Image.open(io.BytesIO(archive.read("color.jpg"))).convert("RGB")
        )[..., ::-1].copy()
        depth = np.array(
            Image.open(io.BytesIO(archive.read("depth.png"))), dtype=np.float32
        )
    depth *= metadata["calibration"]["depth_scale_m"]
    intrinsics = metadata["calibration"]["intrinsics"]
    yy, xx = np.indices(depth.shape)
    rays = np.stack(
        (
            (xx - intrinsics["ppx"]) / intrinsics["fx"],
            (yy - intrinsics["ppy"]) / intrinsics["fy"],
            np.ones(depth.shape),
        ),
        axis=2,
    )
    source = metadata["source_metadata"]
    return dict(
        color_bgr=color,
        depth_m=depth,
        rays=rays,
        timestamp=source["depth_time"],
        meta=dict(
            camera_serial=source["camera_serial"], frame_id=metadata["source_frame_id"]
        ),
    )


class RecordedColorDepthTests(unittest.TestCase):
    def test_both_failed_views_acquire_and_track_without_yolo(self):
        config = json.loads((ROOT / "last-scene.json").read_text())
        detector = Mock()
        detector.detect.side_effect = AssertionError(
            "Classifier is not a control dependency"
        )
        tracker = TargetTracker(detector, identity_mode="color_depth")
        for number in (115503, 115616):
            frame = recorded_frame(number)
            with (
                self.subTest(frame=number),
                patch(
                    "yolo_client.time.monotonic", return_value=frame["timestamp"] + 0.05
                ),
            ):
                observation, vision = geometry(frame, config, tracker=tracker)
                tracker.accept(observation, vision, frame)
                self.assertLess(vision["fit_error_m"], 0.006)
                np.testing.assert_allclose(
                    observation.center[:2], [0.316, 0.165], atol=0.003
                )
                self.assertEqual(observation.timestamp, frame["timestamp"])
        detector.detect.assert_not_called()

    def test_missing_target_still_blocks_without_asking_yolo(self):
        frame = recorded_frame(115616)
        frame["color_bgr"][:] = 0
        config = json.loads((ROOT / "last-scene.json").read_text())
        with patch(
            "yolo_client.time.monotonic", return_value=frame["timestamp"] + 0.05
        ):
            with self.assertRaisesRegex(RuntimeError, "Color/depth.*absent"):
                geometry(
                    frame, config, tracker=TargetTracker(identity_mode="color_depth")
                )

    def test_stale_recording_still_blocks_in_color_mode(self):
        frame = recorded_frame(115616)
        config = json.loads((ROOT / "last-scene.json").read_text())
        with patch("yolo_client.time.monotonic", return_value=frame["timestamp"] + 1):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                geometry(
                    frame, config, tracker=TargetTracker(identity_mode="color_depth")
                )


if __name__ == "__main__":
    unittest.main()
