"""Lift continuation retains its IK branch and restores the FK query state."""

import unittest
from unittest.mock import Mock
import numpy as np
import test_arm_gravity


class CartesianLiftTests(unittest.TestCase):
    def plant(self):
        plant, _, _, _ = test_arm_gravity.ArmGravityTests().plant()
        plant.set_contact_phase("lift")
        return plant

    def test_local_ik_keeps_previous_branch_for_small_displacement(self):
        plant = self.plant()
        start = plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]].copy()
        position = plant.ik.data.site_xpos[plant.ik.site].copy() + [0, 0, 0.005]
        rotation = plant.ik.data.site_xmat[plant.ik.site].reshape(3, 3).copy()
        goal = plant._continuous_lift_ik(position, rotation)
        self.assertLessEqual(np.max(np.abs(goal - start)), 0.2000001)
        self.assertLess(
            np.linalg.norm(plant.ik.ik_data.site_xpos[plant.ik.site] - position), 0.008
        )

    def test_failed_segment_restores_model_and_never_commands(self):
        plant = self.plant()
        original = plant.ik.data.qpos.copy()
        plant._joint_poses = Mock(side_effect=RuntimeError("obstacle"))
        rotation = plant.ik.data.site_xmat[plant.ik.site].reshape(3, 3).copy()
        with self.assertRaisesRegex(RuntimeError, "obstacle"):
            plant._cartesian_lift_poses(
                plant.ik.data.site_xpos[plant.ik.site] + [0, 0, 0.005],
                rotation=rotation,
            )
        np.testing.assert_array_equal(plant.ik.data.qpos, original)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_ik_failure_subdivides_then_checks_every_accepted_segment(self):
        plant = self.plant()
        original = plant.ik.data.qpos.copy()
        solve = plant._continuous_lift_ik
        calls = []

        def flaky(position, rotation):
            calls.append(np.array(position))
            if len(calls) == 1:
                raise RuntimeError(
                    "Continuous Cartesian lift IK cannot preserve the grasp pose"
                )
            return solve(position, rotation)

        plant._continuous_lift_ik = flaky
        checked = plant._joint_poses
        plant._joint_poses = Mock(wraps=checked)
        start = plant.ik.data.site_xpos[plant.ik.site].copy()
        rotation = plant.ik.data.site_xmat[plant.ik.site].reshape(3, 3).copy()
        plant._cartesian_lift_poses(start + [0, 0, 0.005], rotation=rotation)
        self.assertEqual(len(calls), 3)
        self.assertEqual(plant._joint_poses.call_count, 2)
        np.testing.assert_allclose(calls[1], start + [0, 0, 0.0025])
        np.testing.assert_array_equal(plant.ik.data.qpos, original)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_persistent_ik_failure_is_bounded(self):
        plant = self.plant()
        plant._continuous_lift_ik = Mock(
            side_effect=RuntimeError(
                "Continuous Cartesian lift IK cannot preserve the grasp pose"
            )
        )
        with self.assertRaisesRegex(RuntimeError, "cannot preserve"):
            plant._cartesian_lift_poses(
                plant.ik.data.site_xpos[plant.ik.site] + [0, 0, 0.01]
            )
        self.assertLessEqual(plant._continuous_lift_ik.call_count, 5)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_rejects_downward_or_unbounded_lift(self):
        plant = self.plant()
        start = plant.ik.data.site_xpos[plant.ik.site].copy()
        for delta in ([0, 0, -0.02], [0, 0, 0.21]):
            with self.assertRaisesRegex(RuntimeError, "at most 0.20"):
                plant._cartesian_lift_poses(start + delta)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_downward_alignment_is_continuous_checked_and_read_only(self):
        plant = self.plant()
        plant.set_contact_phase("align")
        before = plant.ik.data.qpos.copy()
        start = plant.ik.data.site_xpos[plant.ik.site].copy()
        rotation = plant.ik.data.site_xmat[plant.ik.site].reshape(3, 3).copy()
        plant._joint_poses = Mock(wraps=plant._joint_poses)
        plant._cartesian_transfer_poses(start + [0, 0, -0.005], rotation=rotation)
        plant._joint_poses.assert_called()
        np.testing.assert_array_equal(plant.ik.data.qpos, before)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_descent_cannot_be_requested_during_lift(self):
        plant = self.plant()
        start = plant.ik.data.site_xpos[plant.ik.site].copy()
        with self.assertRaisesRegex(RuntimeError, "descent requires"):
            plant._cartesian_lift_poses(start + [0, 0, -0.005], allow_descent=True)
        plant.adapter.ctrl_dual_arm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
