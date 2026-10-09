"""Exercise the deployed camera owner's real methods with immutable fixtures.

Extract only hardware-independent methods so offline tests cannot import USB,
FastAPI or RealSense. The method bodies are compiled directly from ship source.
"""

import ast
from collections import OrderedDict
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
import zipfile
import cv2
import numpy as np


def owner_class():
    source = Path(__file__).parent / "camera_owner/src/coke_detector/camera_service.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "CameraRuntime"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("__init__", "recorder_rgbd_snapshot", "tracking_rgbd_snapshot")
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    scope = dict(
        threading=threading,
        OrderedDict=OrderedDict,
        time=time,
        np=np,
        cv2=cv2,
        io=io,
        json=json,
        zipfile=zipfile,
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope["CameraRuntime"]


class TrackingCameraOwnerTests(unittest.TestCase):
    def owner(self):
        owner = owner_class()(SimpleNamespace(rgbd_enabled=True, jpeg_quality=90))
        owner._stream_id = "current"
        return owner

    def frame(self, fid):
        return dict(
            color_bgr=np.zeros((4, 6, 3), np.uint8),
            depth_m=np.ones((4, 6), np.float32),
            meta=dict(
                frame_id=fid,
                camera_serial="real-camera",
                captured_at_unix_ns=100 + fid,
                aligned_to="color",
                depth_scale_m=0.001,
                intrinsics={"width": 6, "height": 4},
                depth_time=99.95,
                color_time=99.95,
                stream_id="current",
                timestamp_valid=True,
            ),
        )

    def test_newest_bundle_contains_unchanged_exposure_metadata_and_depth(self):
        owner = self.owner()
        owner._rgbd_frames.update({1: self.frame(1), 2: self.frame(2)})
        raw, meta = owner.tracking_rgbd_snapshot(after_frame_id=1, stream_id="current")
        self.assertEqual(meta["source_frame_id"], 2)
        self.assertEqual(meta["source_metadata"], owner._rgbd_frames[2]["meta"])
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            self.assertEqual(
                set(bundle.namelist()), {"metadata.json", "depth.png", "color.jpg"}
            )
            decoded = cv2.imdecode(
                np.frombuffer(bundle.read("depth.png"), np.uint8), cv2.IMREAD_UNCHANGED
            )
            np.testing.assert_array_equal(decoded, np.full((4, 6), 1000, np.uint16))

    def test_same_frame_waits_for_capture_notification_instead_of_replaying(self):
        owner = self.owner()
        owner._rgbd_frames[1] = self.frame(1)
        done = threading.Event()
        result = []

        def read():
            result.append(owner.tracking_rgbd_snapshot(1, "current"))
            done.set()

        thread = threading.Thread(target=read)
        thread.start()
        try:
            self.assertFalse(done.wait(0.02))
            with owner._frame_ready:
                owner._rgbd_frames[2] = self.frame(2)
                owner._frame_ready.notify_all()
            self.assertTrue(done.wait(1))
            self.assertEqual(result[0][1]["source_frame_id"], 2)
        finally:
            thread.join(1)

    def test_stream_restart_and_missing_progress_fail(self):
        owner = self.owner()
        owner._rgbd_frames[1] = self.frame(1)
        with self.assertRaisesRegex(RuntimeError, "stream changed"):
            owner.tracking_rgbd_snapshot(1, "old")
        with self.assertRaisesRegex(RuntimeError, "200 ms"):
            owner.tracking_rgbd_snapshot(1, "current")


if __name__ == "__main__":
    unittest.main()
