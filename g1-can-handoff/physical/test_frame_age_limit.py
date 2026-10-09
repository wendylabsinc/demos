"""Replay recorded exposure stops and enforce the bounded configured budget."""

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from yolo_client import fresh, identity
from yolo.frame_limits import frame_age_limit_s


class FrameAgeLimitTests(unittest.TestCase):
    def frame(self):
        return {
            "timestamp": 10.0,
            "color_bgr": np.zeros((2, 2, 3), np.uint8),
            "meta": {"frame_id": 1, "camera_serial": "test"},
        }

    def service(self):
        root = Path(__file__).parent / "yolo"
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location(
            "age_test_service", root / "service.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_saved_341ms_fault_is_accepted_but_401ms_and_future_rejected(self):
        service = self.service()
        frame = self.frame()
        source = identity(frame)
        for now, allowed in [
            (10.3412, True),
            (10.3999, True),
            (10.4001, False),
            (9.9999, False),
        ]:
            with (
                self.subTest(now=now),
                patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "400"}),
                patch("time.monotonic", return_value=now),
            ):
                if allowed:
                    fresh(frame)
                    service.decode_frame(source, frame["color_bgr"].tobytes())
                else:
                    with self.assertRaises(RuntimeError):
                        fresh(frame)
                    with self.assertRaises(ValueError):
                        service.decode_frame(source, frame["color_bgr"].tobytes())

    def test_bad_or_unbounded_configuration_fails_closed(self):
        for value in ["0", "49", "501", "inf", "nan", "bad"]:
            with (
                self.subTest(value=value),
                patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": value}),
            ):
                with self.assertRaises(ValueError):
                    frame_age_limit_s()

    def test_500ms_budget_accepts_recorded_stop_but_rejects_older_or_future_frames(
        self,
    ):
        service = self.service()
        frame = self.frame()
        source = identity(frame)
        for now, allowed in [
            (10.4017, True),
            (10.4999, True),
            (10.5001, False),
            (9.9999, False),
        ]:
            with (
                self.subTest(now=now),
                patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "500"}),
                patch("time.monotonic", return_value=now),
            ):
                if allowed:
                    fresh(frame)
                    service.decode_frame(source, frame["color_bgr"].tobytes())
                else:
                    with self.assertRaises(RuntimeError):
                        fresh(frame)
                    with self.assertRaises(ValueError):
                        service.decode_frame(source, frame["color_bgr"].tobytes())

    def test_stricter_override_rejects_saved_fault(self):
        with (
            patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "300"}),
            patch("time.monotonic", return_value=10.3412),
        ):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                fresh(self.frame())

    def test_motion_tracking_guard_uses_the_same_bounded_budget(self):
        from runtime import Runtime

        runtime = Runtime.__new__(Runtime)
        runtime.lock = threading.Lock()
        runtime.tracking_observation = SimpleNamespace(timestamp=10.0)
        runtime.tracking_error = None
        for limit, now, allowed in [
            ("400", 10.4017, False),
            ("500", 10.4017, True),
            ("500", 10.4999, True),
            ("500", 10.5001, False),
            ("500", 9.9999, False),
        ]:
            with (
                self.subTest(limit=limit, now=now),
                patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": limit}),
                patch("time.monotonic", return_value=now),
            ):
                if allowed:
                    runtime.tracking_guard()
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "tracking exposure stale"
                    ):
                        runtime.tracking_guard()
        runtime.tracking_error = "uncertain appearance"
        with (
            patch.dict("os.environ", {"HANDOFF_CAMERA_MAX_AGE_MS": "500"}),
            patch("time.monotonic", return_value=10.01),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracking uncertain"):
                runtime.tracking_guard()


if __name__ == "__main__":
    unittest.main()
