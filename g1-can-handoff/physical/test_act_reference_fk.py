"""Scratch ACT FK/Jacobian parity with full dynamics; no hardware IO."""
import unittest
from unittest.mock import patch
import numpy as np
import mujoco
import test_act_residual


class ACTReferenceFKTests(unittest.TestCase):
    def test_kinematics_and_com_match_full_forward_for_500_nominal_targets(self):
        plant, io, policy, _ = test_act_residual.ACTTests().plant()
        model = plant.ik.model
        reduced = mujoco.MjData(model)
        full = mujoco.MjData(model)
        rng = np.random.default_rng(819714)
        for _ in range(500):
            full.qpos[:] = plant.ik.data.qpos
            full.qpos[policy.qadr[:7]] = rng.uniform(policy.bounds[:, 0], policy.bounds[:, 1])
            full.qvel[:] = rng.normal(0, .1, model.nv)
            reduced.qpos[:] = full.qpos
            reduced.qvel[:] = full.qvel
            mujoco.mj_forward(model, full)
            mujoco.mj_kinematics(model, reduced)
            mujoco.mj_comPos(model, reduced)
            np.testing.assert_array_equal(reduced.site_xpos, full.site_xpos)
            np.testing.assert_array_equal(reduced.site_xmat, full.site_xmat)
            full_jac = [np.zeros((3, model.nv)), np.zeros((3, model.nv))]
            reduced_jac = [np.zeros((3, model.nv)), np.zeros((3, model.nv))]
            mujoco.mj_jacSite(model, full, *full_jac, plant.ik.site)
            mujoco.mj_jacSite(model, reduced, *reduced_jac, plant.ik.site)
            for expected, actual in zip(full_jac, reduced_jac):
                np.testing.assert_array_equal(actual, expected)
            self.assertEqual(io.commands, 0)

    def test_update_uses_reduced_scratch_pipeline_without_changing_live_geometry(self):
        plant, io, policy, _ = test_act_residual.ACTTests().plant()
        initial_qpos = plant.ik.data.qpos.copy()
        initial_site = plant.ik.data.site_xpos.copy()
        with patch.object(mujoco, 'mj_forward', wraps=mujoco.mj_forward) as full:
            target = policy.update(plant.left_arm, io.read())
        self.assertEqual(full.call_count, 0)
        self.assertTrue(np.isfinite(target).all())
        np.testing.assert_array_equal(plant.ik.data.qpos, initial_qpos)
        np.testing.assert_array_equal(plant.ik.data.site_xpos, initial_site)
        self.assertEqual(io.commands, 0)


if __name__ == '__main__':
    unittest.main()
