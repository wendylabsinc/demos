import copy
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pose_alignment import PoseHistory, corrected_config


class PoseAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.reference = [0.02, 0.035, 2.9]
        pelvis = np.eye(4)
        pelvis[:3, :3] = Rotation.from_euler(
            "xyz", [*self.reference[:2], 0]
        ).as_matrix()
        pelvis[:3, 3] = [0.1, -0.2, 0.8]
        camera = pelvis.copy()
        camera[:3, 3] += [0.4, 0.05, 0.5]
        self.config = dict(
            calibration_method="operator_accepted_factory_table",
            reference_imu_rpy=self.reference,
            reference_base_q=[0.0] * 15,
            camera_to_world=camera.tolist(),
            pelvis_to_world=pelvis.tolist(),
            table_height_m=0.762,
            operator_accepted=True,
            assumptions=["original provenance"],
        )
        self.pose = dict(timestamp=100.0, rpy=list(self.reference), waist_q=[0.0] * 3)

    def test_nearest_uses_exposure_not_wall_clock_and_copies_sample(self):
        history = PoseHistory()
        history.add(100.0, [0, 0, 0], [0, 0, 0])
        history.add(100.01, [1, 2, 3], [4, 5, 6])
        chosen = history.nearest(100.008)
        self.assertEqual(chosen["timestamp"], 100.01)
        chosen["rpy"][0] = 99
        self.assertEqual(history.nearest(100.008)["rpy"][0], 1)

    def test_old_and_future_samples_outside_twenty_ms_rejected(self):
        history = PoseHistory()
        with self.assertRaises(RuntimeError):
            history.nearest(100)
        history.add(100, [0] * 3, [0] * 3)
        for exposure in (99.979, 100.021):
            with self.assertRaisesRegex(RuntimeError, "20 ms"):
                history.nearest(exposure)
        for exposure in (99.98, 100.02):
            self.assertEqual(history.nearest(exposure)["timestamp"], 100)

    def test_history_bounded_and_timestamps_advance(self):
        history = PoseHistory(max_samples=2)
        for timestamp in (1.0, 2.0, 3.0):
            history.add(timestamp, [0] * 3, [0] * 3)
        with self.assertRaises(RuntimeError):
            history.nearest(1)
        with self.assertRaises(ValueError):
            history.add(3, [0] * 3, [0] * 3)
        for capacity in (0, 513, True):
            with self.assertRaises(ValueError):
                PoseHistory(capacity)

    def test_identity_and_original_accepted_config_unchanged(self):
        before = copy.deepcopy(self.config)
        result = corrected_config(self.config, self.pose)
        for key in ("camera_to_world", "pelvis_to_world"):
            np.testing.assert_allclose(result[key], self.config[key], atol=1e-14)
        result["assumptions"].append("changed copy")
        self.assertEqual(self.config, before)
        self.assertEqual(result["reference_imu_rpy"], self.reference)

    def test_known_rotation_about_pelvis_pivot_updates_both_transforms(self):
        self.pose["rpy"][1] -= 0.013
        delta = (
            Rotation.from_euler("xyz", [*self.pose["rpy"][:2], 0]).as_matrix()
            @ Rotation.from_euler("xyz", [*self.reference[:2], 0]).as_matrix().T
        )
        result = corrected_config(self.config, self.pose)
        pivot = np.asarray(self.config["pelvis_to_world"])[:3, 3]
        for key in ("camera_to_world", "pelvis_to_world"):
            original = np.asarray(self.config[key])
            corrected = np.asarray(result[key])
            np.testing.assert_allclose(
                corrected[:3, :3], delta @ original[:3, :3], atol=1e-14
            )
            np.testing.assert_allclose(
                corrected[:3, 3], pivot + delta @ (original[:3, 3] - pivot), atol=1e-14
            )
        np.testing.assert_allclose(np.asarray(result["pelvis_to_world"])[:3, 3], pivot)
        self.assertEqual(result["table_height_m"], self.config["table_height_m"])

    def test_yaw_is_relative_and_wraps(self):
        self.config["reference_imu_rpy"][2] = np.pi - 0.01
        self.pose["rpy"][2] = -np.pi + 0.01
        result = corrected_config(self.config, self.pose)
        self.assertAlmostEqual(result["pose_alignment"]["delta_rpy_rad"][2], 0.02)

    def test_attitude_and_unmodeled_waist_changes_rejected(self):
        for component in range(3):
            pose = copy.deepcopy(self.pose)
            pose["rpy"][component] += 0.04001
            with self.assertRaisesRegex(RuntimeError, "attitude"):
                corrected_config(self.config, pose)
        self.pose["waist_q"][1] = 0.00201
        with self.assertRaisesRegex(RuntimeError, "Waist"):
            corrected_config(self.config, self.pose)

    def test_nonfinite_data_and_wrong_calibration_rejected(self):
        history = PoseHistory()
        with self.assertRaises(ValueError):
            history.add(float("nan"), [0] * 3, [0] * 3)
        with self.assertRaises(ValueError):
            history.nearest(float("nan"))
        self.pose["rpy"][0] = float("nan")
        with self.assertRaises(ValueError):
            corrected_config(self.config, self.pose)
        self.config["calibration_method"] = "unaccepted"
        with self.assertRaisesRegex(ValueError, "accepted"):
            corrected_config(self.config, self.pose)


if __name__ == "__main__":
    unittest.main()
