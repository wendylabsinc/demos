"""Regress the real [casing 53, winding 85] stop and retain both sensor guards."""

import copy, json, time, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from physical_core import (
    snapshot_guard,
    body_temperature_limit_c,
    body_winding_temperature_limit_c,
)


class BodyTemperatureSensors(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            "os.environ",
            {
                "HANDOFF_BODY_TEMPERATURE_LIMIT_C": "85",
                "HANDOFF_BODY_CASING_TEMPERATURE_LIMIT_C": "85",
                "HANDOFF_BODY_WINDING_TEMPERATURE_LIMIT_C": "95",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.snapshot = {
            name: dict(
                time=10,
                q=[0] * n,
                dq=[0] * n,
                tau_est=[0] * n,
                imu_rpy=[0, 0, 0],
                faults=[0] * n,
                temperature_max=35,
            )
            for name, n in [("body", 29), ("left", 7), ("right", 7)]
        }
        self.snapshot["body"]["temperatures"] = [[35, 35] for _ in range(29)]

    def test_recorded_winding_85_with_casing_53_passes(self):
        fixture = json.loads(
            (
                Path(__file__).parent / "offline/fixtures/body-thermal-20261009.json"
            ).read_text()
        )
        self.snapshot["body"].update(
            temperatures=fixture["temperatures"], temperature_max=85
        )
        snapshot_guard(self.snapshot, 10.1)
        self.assertEqual(fixture["temperatures"][16], [53, 85])

    def test_winding_boundary_is_independent_of_casing(self):
        self.snapshot["body"]["temperatures"][16] = [53, 94.99]
        snapshot_guard(self.snapshot, 10.1)
        self.snapshot["body"]["temperatures"][16] = [53, 95]
        with self.assertRaisesRegex(
            ValueError,
            "body winding temperature 95 C at left_shoulder_roll_joint reaches the 95 C guard",
        ):
            snapshot_guard(self.snapshot, 10.1)

    def test_casing_boundary_still_stops_at_85(self):
        self.snapshot["body"]["temperatures"][16] = [84.99, 35]
        snapshot_guard(self.snapshot, 10.1)
        self.snapshot["body"]["temperatures"][16] = [85, 35]
        with self.assertRaisesRegex(
            ValueError, "body casing temperature 85 C at left_shoulder_roll_joint"
        ):
            snapshot_guard(self.snapshot, 10.1)

    def test_requested_120_winding_guard_accepts_95_but_stops_at_120(self):
        with patch.dict('os.environ', {'HANDOFF_BODY_WINDING_TEMPERATURE_LIMIT_C':'120'}):
            self.snapshot['body']['temperatures'][16]=[53,95]
            snapshot_guard(self.snapshot,10.1)
            self.snapshot['body']['temperatures'][16]=[53,119.99]
            snapshot_guard(self.snapshot,10.1)
            self.snapshot['body']['temperatures'][16]=[53,120]
            with self.assertRaisesRegex(ValueError,'body winding temperature 120 C at left_shoulder_roll_joint reaches the 120 C guard'):
                snapshot_guard(self.snapshot,10.1)
            self.snapshot['body']['temperatures'][16]=[85,95]
            with self.assertRaisesRegex(ValueError,'body casing temperature 85 C'):
                snapshot_guard(self.snapshot,10.1)

    def test_every_joint_and_both_sensors_checked(self):
        for joint in range(29):
            for sensor, limit in ((0, 85), (1, 95)):
                with self.subTest(joint=joint, sensor=sensor):
                    snap = copy.deepcopy(self.snapshot)
                    snap["body"]["temperatures"][joint][sensor] = limit
                    with self.assertRaisesRegex(
                        ValueError, "body (casing|winding) temperature"
                    ):
                        snapshot_guard(snap, 10.1)

    def test_malformed_or_nonfinite_raw_sensor_arrays_fail(self):
        for raw in (
            None,
            [],
            [[35]] * 29,
            [[35, 35]] * 28,
            [[35, 35]] * 30,
            [[35, float("nan")]] * 29,
            [[float("inf"), 35]] * 29,
        ):
            with self.subTest(raw=raw):
                snap = copy.deepcopy(self.snapshot)
                snap["body"]["temperatures"] = raw
                with self.assertRaisesRegex(ValueError, "Invalid body casing/winding"):
                    snapshot_guard(snap, 10.1)

    def test_unidentified_legacy_sensor_cannot_use_higher_winding_guard(self):
        del self.snapshot["body"]["temperatures"]
        self.snapshot["body"]["temperature_max"] = 85
        with self.assertRaisesRegex(
            ValueError, "body temperature 85 C reaches the 85 C guard"
        ):
            snapshot_guard(self.snapshot, 10.1)

    def test_hand_guard_and_faults_remain(self):
        self.snapshot["left"]["temperature_max"] = 75
        with self.assertRaisesRegex(ValueError, "left temperature"):
            snapshot_guard(self.snapshot, 10.1)
        self.snapshot["left"]["temperature_max"] = 35
        self.snapshot["body"]["faults"][16] = 512
        with self.assertRaisesRegex(ValueError, "body motor fault"):
            snapshot_guard(self.snapshot, 10.1)

    def test_limits_cannot_exceed_vendor_protection_defaults(self):
        for name, fn, cap in (
            ("HANDOFF_BODY_CASING_TEMPERATURE_LIMIT_C", body_temperature_limit_c, 85),
            (
                "HANDOFF_BODY_WINDING_TEMPERATURE_LIMIT_C",
                body_winding_temperature_limit_c,
                120,
            ),
        ):
            for value in (str(cap + 0.01), "nan", "inf", "0", "-1", "invalid"):
                with (
                    self.subTest(name=name, value=value),
                    patch.dict("os.environ", {name: value}),
                ):
                    with self.assertRaisesRegex(ValueError, f"at most {cap} C"):
                        fn()

    def test_legacy_casing_alias_remains_supported(self):
        with patch.dict(
            "os.environ", {"HANDOFF_BODY_TEMPERATURE_LIMIT_C": "75"}, clear=True
        ):
            self.assertEqual(body_temperature_limit_c(), 75)
            self.assertEqual(body_winding_temperature_limit_c(), 95)


if __name__ == "__main__":
    unittest.main()
