import copy
import time
import unittest
from unittest.mock import patch
import test_model_calibration
from scene_refresh import refresh_scene


class SceneRefreshTests(unittest.TestCase):
    def setUp(self):
        override = patch.dict('os.environ', {'HANDOFF_POSE_ALIGNMENT_MAX_SKEW_MS': '20'})
        override.start()
        self.addCleanup(override.stop)
        fixture = test_model_calibration.ModelCalibrationTests()
        fixture.setUp()
        self.config = fixture.config
        self.snapshot = fixture.snapshot
        self.frame = {"meta": {"camera_serial": "camera"}}
        self.fresh = copy.deepcopy(self.config)
        self.fresh.update(
            commissioned=False,
            calibration_method="factory_mount_and_table_plane_candidate",
            source_frame_id=101,
        )

    def test_configured_40_ms_refresh_accepts_35_ms_skew(self):
        self.frame['timestamp'] = 100.0
        pose = dict(timestamp=100.035, rpy=[0.0] * 3, waist_q=[0.0] * 3)
        with patch.dict('os.environ', {'HANDOFF_POSE_ALIGNMENT_MAX_SKEW_MS': '40'}), patch('scene_refresh.candidate', return_value=self.fresh):
            result = refresh_scene(self.config, self.frame, self.snapshot, 'camera', exposure_pose=pose)
        self.assertAlmostEqual(result['calibration_exposure_pose']['skew_ms'], 35.0)

    def test_expired_scene_replaced_by_new_observation_and_current_pose(self):
        self.config["accepted_at_unix_s"] = time.time() - 3600
        self.snapshot["body"]["q"][12] = 0.1
        before = copy.deepcopy(self.config)
        with patch("scene_refresh.candidate", return_value=self.fresh) as observe:
            result = refresh_scene(self.config, self.frame, self.snapshot, "camera")
        observe.assert_called_once_with(self.frame, self.snapshot)
        self.assertEqual(result["source_frame_id"], 101)
        self.assertEqual(result["reference_base_q"][12], 0.1)
        self.assertFalse(result["independent_validation"])
        self.assertTrue(result["scene_refresh_automatic"])
        self.assertEqual(self.config, before)

    def test_refresh_does_not_create_operator_authorization(self):
        for field in (
            "commissioned",
            "operator_accepted",
            "factory_mount_position_confirmed",
            "stationary_base_confirmed",
        ):
            config = dict(self.config)
            config[field] = False
            with patch("scene_refresh.candidate") as observe:
                with self.assertRaises(ValueError):
                    refresh_scene(config, self.frame, self.snapshot, "camera")
                observe.assert_not_called()

    def test_bad_new_geometry_and_wrong_camera_are_rejected(self):
        self.fresh["table_plane_rmse_m"] = 0.01
        with patch("scene_refresh.candidate", return_value=self.fresh):
            with self.assertRaisesRegex(ValueError, "table_plane"):
                refresh_scene(self.config, self.frame, self.snapshot, "camera")
        self.frame["meta"]["camera_serial"] = "other"
        with patch("scene_refresh.candidate") as observe:
            with self.assertRaisesRegex(ValueError, "identity"):
                refresh_scene(self.config, self.frame, self.snapshot, "camera")
            observe.assert_not_called()

    def test_reference_and_candidate_use_exposure_pose_without_mutating_live_snapshot(
        self,
    ):
        self.frame["timestamp"] = 100.0
        self.snapshot["body"]["imu_rpy"] = [0.01, 0.04, 0.02]
        self.snapshot["body"]["q"] = list(range(29))
        before = copy.deepcopy(self.snapshot)
        exposure = dict(
            timestamp=100.003, rpy=[0.02, 0.025, 0.03], waist_q=[0.001, 0.002, 0.003]
        )
        with patch("scene_refresh.candidate", return_value=self.fresh) as observe:
            result = refresh_scene(
                self.config, self.frame, self.snapshot, "camera", exposure_pose=exposure
            )
        used = observe.call_args.args[1]
        self.assertEqual(used["body"]["imu_rpy"], exposure["rpy"])
        self.assertEqual(used["body"]["q"][12:15], exposure["waist_q"])
        self.assertEqual(used["body"]["q"][:12], before["body"]["q"][:12])
        self.assertEqual(used["body"]["q"][15:], before["body"]["q"][15:])
        self.assertEqual(used["body"]["time"], before["body"]["time"])
        self.assertEqual(result["reference_imu_rpy"], exposure["rpy"])
        self.assertEqual(result["reference_base_q"][12:15], exposure["waist_q"])
        self.assertEqual(self.snapshot, before)
        self.assertAlmostEqual(result["calibration_exposure_pose"]["skew_ms"], 3.0)

    def test_exposure_pose_rejects_skew_or_nonfinite_before_candidate(self):
        self.frame["timestamp"] = 100.0
        pose = dict(timestamp=100.0, rpy=[0.0, 0.0, 0.0], waist_q=[0.0, 0.0, 0.0])
        bad = []
        for timestamp in (99.979, 100.021, float("nan")):
            bad.append(dict(pose, timestamp=timestamp))
        bad.extend(
            [
                dict(pose, rpy=[0.0, float("nan"), 0.0]),
                dict(pose, waist_q=[0.0, 0.0]),
                dict(pose, waist_q=[0.0, 0.0, float("inf")]),
            ]
        )
        for exposure in bad:
            with (
                self.subTest(exposure=exposure),
                patch("scene_refresh.candidate") as observe,
            ):
                with self.assertRaises(ValueError):
                    refresh_scene(
                        self.config,
                        self.frame,
                        self.snapshot,
                        "camera",
                        exposure_pose=exposure,
                    )
                observe.assert_not_called()

    def test_exposure_pose_does_not_bypass_live_snapshot_health(self):
        self.frame["timestamp"] = 100.0
        pose = dict(timestamp=100.0, rpy=[0.0, 0.0, 0.0], waist_q=[0.0, 0.0, 0.0])
        for changes in ({"time": time.monotonic() - 1.0}, {"temperature_max": 78.0}):
            snapshot = copy.deepcopy(self.snapshot)
            snapshot["body"].update(changes)
            with patch("scene_refresh.candidate") as observe:
                with self.assertRaises(ValueError):
                    refresh_scene(
                        self.config, self.frame, snapshot, "camera", exposure_pose=pose
                    )
                observe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
