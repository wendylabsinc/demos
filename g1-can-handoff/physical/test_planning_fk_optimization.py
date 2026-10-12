"""FK/contact equivalence checks for planning computation stages; no publishers."""
import unittest
import numpy as np
import mujoco
from g1_can_handoff.sim import scene

class PlanningStagesTests(unittest.TestCase):
    def test_fk_and_collision_stages_match_full_forward_at_200_joint_poses(self):
        plant = scene(height=0.762)
        model = plant.model
        full, reduced = mujoco.MjData(model), mujoco.MjData(model)
        adr = plant.qadr[plant.arm]
        bounds = model.jnt_range[plant.joints[plant.arm]]
        rng = np.random.default_rng(713)
        for _ in range(200):
            full.qpos[:] = plant.data.qpos
            full.qpos[adr] = rng.uniform(bounds[:, 0], bounds[:, 1])
            full.qvel[:] = rng.normal(0, 0.1, model.nv)
            reduced.qpos[:] = full.qpos
            reduced.qvel[:] = full.qvel
            mujoco.mj_forward(model, full)
            mujoco.mj_kinematics(model, reduced)
            np.testing.assert_array_equal(full.site_xpos, reduced.site_xpos)
            np.testing.assert_array_equal(full.site_xmat, reduced.site_xmat)
            np.testing.assert_array_equal(full.geom_xpos, reduced.geom_xpos)
            for a, b in [(model.geom('table').id, model.geom('can_shell').id)]:
                self.assertEqual(mujoco.mj_geomDistance(model, full, a, b, .1, None),
                                 mujoco.mj_geomDistance(model, reduced, a, b, .1, None))
            mujoco.mj_fwdPosition(model, reduced)
            self.assertEqual(full.ncon, reduced.ncon)
            for x, y in zip(full.contact, reduced.contact):
                self.assertEqual((x.geom1, x.geom2, x.dist), (y.geom1, y.geom2, y.dist))
            self.assertEqual(plant.collision_details(full), plant.collision_details(reduced))
