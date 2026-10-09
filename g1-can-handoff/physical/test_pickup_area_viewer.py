"""Placement projection and visibility checks; no network or robot required."""

from copy import deepcopy
import unittest

import numpy as np
from pickup_area_viewer import project_area


class PickupAreaTests(unittest.TestCase):
    def snapshot(self):
        return dict(
            demo_running=False,
            phase="PASSIVE",
            body_age_s=0.01,
            body=dict(time=100.0, imu_rpy=[0, 0, 0], q=[0] * 29),
            bounds={"0": [0.27, 0.45], "1": [0.09, 0.29]},
            camera_serial="test",
            camera_health=dict(healthy=True, frame_age_s=0.03),
            intrinsics=dict(
                fx=100.0,
                fy=100.0,
                ppx=0.0,
                ppy=0.0,
                width=640,
                height=480,
                coeffs=[0] * 5,
            ),
            config=dict(
                calibration_method="operator_accepted_factory_table",
                operator_accepted=True,
                stationary_base_confirmed=True,
                camera_serial="test",
                reference_imu_rpy=[0, 0, 0],
                reference_base_q=[0] * 15,
                table_height_m=1.0,
                camera_to_world=np.eye(4).tolist(),
                pelvis_to_world=np.eye(4).tolist(),
            ),
        )

    def test_projects_all_four_workspace_corners_on_table(self):
        result = project_area(self.snapshot())
        np.testing.assert_allclose(
            result["polygon"], [[27, 9], [45, 9], [45, 29], [27, 29]]
        )
        self.assertEqual(result["bounds_m"], {"0": [0.27, 0.45], "1": [0.09, 0.29]})

    def test_uses_camera_translation_and_current_attitude(self):
        snapshot = self.snapshot()
        snapshot["config"]["camera_to_world"][0][3] = 0.1
        np.testing.assert_allclose(
            project_area(snapshot)["polygon"], [[17, 9], [35, 9], [35, 29], [17, 29]]
        )
        before = project_area(snapshot)["polygon"]
        snapshot["body"]["imu_rpy"][2] = 0.01
        self.assertFalse(np.allclose(project_area(snapshot)["polygon"], before))

    def test_hides_active_stale_mismatched_and_unsupported_projection(self):
        changes = [
            {"demo_running": True},
            {"body_age_s": 0.201},
            {"camera_serial": "other"},
            {"camera_health": dict(healthy=False, frame_age_s=0.03)},
        ]
        for fields in changes:
            snapshot = self.snapshot()
            snapshot.update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                project_area(snapshot)
        snapshot = self.snapshot()
        snapshot["intrinsics"]["coeffs"][0] = 0.01
        with self.assertRaisesRegex(ValueError, "distortion"):
            project_area(snapshot)
        snapshot = self.snapshot()
        snapshot["config"]["operator_accepted"] = False
        with self.assertRaisesRegex(ValueError, "accepted"):
            project_area(snapshot)


if __name__ == "__main__":
    unittest.main()
