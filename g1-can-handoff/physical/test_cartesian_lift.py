"""Lift continuation retains its IK branch and restores the FK query state."""
import unittest
from unittest.mock import Mock
import numpy as np
import test_arm_gravity


class CartesianLiftTests(unittest.TestCase):
    def plant(self):
        plant,_,_,_=test_arm_gravity.ArmGravityTests().plant()
        plant.set_contact_phase('lift')
        return plant

    def test_local_ik_keeps_previous_branch_for_small_displacement(self):
        plant=self.plant()
        start=plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]].copy()
        position=plant.ik.data.site_xpos[plant.ik.site].copy()+[0,0,.005]
        rotation=plant.ik.data.site_xmat[plant.ik.site].reshape(3,3).copy()
        goal=plant._continuous_lift_ik(position,rotation)
        self.assertLessEqual(np.max(np.abs(goal-start)),.2000001)
        self.assertLess(np.linalg.norm(plant.ik.ik_data.site_xpos[plant.ik.site]-position),.008)

    def test_failed_segment_restores_model_and_never_commands(self):
        plant=self.plant()
        original=plant.ik.data.qpos.copy()
        plant._joint_poses=Mock(side_effect=RuntimeError('obstacle'))
        rotation=plant.ik.data.site_xmat[plant.ik.site].reshape(3,3).copy()
        with self.assertRaisesRegex(RuntimeError,'obstacle'):
            plant._cartesian_lift_poses(plant.ik.data.site_xpos[plant.ik.site]+[0,0,.005],rotation=rotation)
        np.testing.assert_array_equal(plant.ik.data.qpos,original)
        plant.adapter.ctrl_dual_arm.assert_not_called()

    def test_rejects_downward_or_unbounded_lift(self):
        plant=self.plant()
        start=plant.ik.data.site_xpos[plant.ik.site].copy()
        for delta in ([0,0,-.02],[0,0,.21]):
            with self.assertRaisesRegex(RuntimeError,'at most 0.20'):
                plant._cartesian_lift_poses(start+delta)
        plant.adapter.ctrl_dual_arm.assert_not_called()

if __name__=='__main__':unittest.main()
