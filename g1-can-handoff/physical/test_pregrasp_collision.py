"""Can pose and phase-specific contact rules, using MuJoCo collision distances."""

import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from g1_can_handoff.sim import Plant, scene
from physical_core import PhysicalPlant
from sim_runtime import SimulationIO


class PregraspCollisionTests(unittest.TestCase):
    def fixture(self, other="left_hand_index_1_link", gap=0.005):
        xml = f"""<mujoco><worldbody>
        <body name="can" pos="0 0 1"><freejoint/>
          <geom type="cylinder" size=".033 .061" margin=".01" gap=".01"/></body>
        <body name="{other}" pos="{.033+.006+gap} 0 1">
          <geom type="sphere" size=".006"/></body>
        </worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        plant = Plant.__new__(Plant)
        plant.model = model
        plant.contact_phase = "align"
        return plant, data

    def test_positive_finger_clearance_below_10mm_is_rejected_before_grasp(self):
        plant, data = self.fixture(gap=0.005)
        self.assertGreater(data.ncon, 0)
        self.assertGreater(data.contact[0].dist, 0)
        for phase in ("pregrasp", "staging", "approach", "align"):
            plant.set_contact_phase(phase)
            self.assertTrue(plant.pose_collision(data), phase)

    def test_intentional_finger_contact_only_allowed_in_grasp_phases(self):
        plant, data = self.fixture(gap=-0.002)
        for phase in ("reach", "close", "lift", "hold", "offer"):
            plant.set_contact_phase(phase)
            self.assertFalse(plant.pose_collision(data), phase)
        plant.set_contact_phase("align")
        self.assertTrue(plant.pose_collision(data))

    def test_diagnostic_identifies_can_clearance_and_phase(self):
        plant, data = self.fixture(gap=0.005)
        detail = plant.collision_details(data)
        self.assertEqual(detail["phase"], "align")
        self.assertEqual(set(detail["bodies"]), {"can", "left_hand_index_1_link"})
        self.assertAlmostEqual(detail["distance_m"], 0.005, places=5)
        plant.set_contact_phase("reach")
        self.assertIsNone(plant.collision_details(data))

    def test_wrist_contact_is_not_exempt_even_during_grasp(self):
        plant, data = self.fixture(other="left_wrist_yaw_link", gap=-0.002)
        plant.set_contact_phase("close")
        self.assertTrue(plant.pose_collision(data))

    def test_camera_updates_collision_proxy_without_moving_simulated_object(self):
        world = scene()
        io = SimulationIO(world)
        model = scene()
        config = {"arm_limits_rad": world.model.jnt_range[io.body_ids[15:]].tolist()}
        plant = PhysicalPlant(
            model, io, io, io.read, config, lambda *a, **k: None, clock=io
        )
        initial = world.can_position().copy()
        center = np.array([0.42, 0.25, 0.83])
        plant.observe_can(SimpleNamespace(center=center))
        plant.sync(io.read())
        np.testing.assert_allclose(model.data.xpos[model.can_id], center)
        np.testing.assert_array_equal(world.can_position(), initial)


if __name__ == "__main__":
    unittest.main()
