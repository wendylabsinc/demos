"""Check planning decisions against analytic distances and explicit pose limits."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from g1_can_handoff.sim import Plant, scene
from g1_can_handoff.task import plan


class IKContractTests(unittest.TestCase):
    def test_table_distances_match_analytic_box_sphere_geometry(self):
        half=np.array([.1,.1,.025])
        radius=.02
        for center in ([0,0,.03], [0,0,.05], [0,0,.0549],
                       [.13,.14,.025], [.125,0,.025]):
            center=np.array(center)
            expected=np.linalg.norm(np.maximum(np.abs(center)-half,0))-radius
            model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
              <geom name="table" type="box" size=".1 .1 .025" margin=".01" gap=".01"/>
              <body name="left_hand_index_1_link" pos="{' '.join(map(str,center))}">
                <freejoint/><geom name="finger" type="sphere" size=".02"/>
              </body></worldbody></mujoco>''')
            data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            plant=Plant.__new__(Plant)
            plant.model,plant.contact_phase,plant.table_clearance_m=model,'reach',.01
            actual=mujoco.mj_geomDistance(model,data,model.geom('table').id,
                                        model.geom('finger').id,.1,None)
            with self.subTest(center=center.tolist()):
                self.assertAlmostEqual(actual,expected,places=10)
                self.assertEqual(plant.pose_collision(data),expected<.01)

    def test_nonadjacent_left_links_cannot_interpenetrate(self):
        # Sphere distance is center separation minus the two radii, independent
        # of the controller's contact policy or the G1 mesh approximation.
        for other in ('left_hip_roll_link', 'left_elbow_link'):
            model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
              <body name="left_hand_thumb_2_link"><freejoint/>
                <geom type="sphere" size=".02"/></body>
              <body name="{other}" pos=".03 0 0">
                <geom type="sphere" size=".02"/></body>
              </worldbody></mujoco>''')
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            plant = Plant.__new__(Plant)
            plant.model, plant.contact_phase = model, 'reach'
            self.assertAlmostEqual(data.contact[0].dist, .03-.02-.02, places=10)
            self.assertIsNotNone(plant.collision_details(data), other)

    def test_endpoint_fallback_does_not_accept_a_10mm_pose_error(self):
        plant = scene()
        q = plant.data.qpos[plant.qadr[plant.arm]].copy()
        position = plant.data.site_xpos[plant.site].copy()+[.01, 0, 0]
        rotation = plant.data.site_xmat[plant.site].reshape(3, 3).copy()
        # An optimizer stopping short cannot waive the public 8 mm pose limit.
        candidate = SimpleNamespace(x=q, fun=np.r_[[-.01, 0, 0], np.zeros(9)])
        with patch('g1_can_handoff.sim.least_squares', return_value=candidate):
            with self.assertRaisesRegex(ValueError, 'Target cannot be reached'):
                plant.solve(position, rotation)

    def test_endpoint_fallback_does_not_accept_excess_rotation_error(self):
        plant = scene()
        q = plant.data.qpos[plant.qadr[plant.arm]].copy()
        position = plant.data.site_xpos[plant.site].copy()
        rotation = plant.data.site_xmat[plant.site].reshape(3, 3).copy()
        angle = .11
        rotation = rotation@np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        residual = .15*(plant.data.site_xmat[plant.site].reshape(3, 3)-rotation).ravel()
        self.assertGreater(np.linalg.norm(residual), .02)
        candidate = SimpleNamespace(x=q, fun=np.r_[np.zeros(3), residual])
        with patch('g1_can_handoff.sim.least_squares', return_value=candidate):
            with self.assertRaisesRegex(ValueError, 'Target cannot be reached'):
                plant.solve(position, rotation)

    def test_stale_feedback_during_endpoint_search_aborts_immediately(self):
        observation = SimpleNamespace(center=np.array([.35, .18, .823]),
                                      table_height=.762, timestamp=time.monotonic())
        plant = Mock()
        plant.solve.side_effect = ValueError('body feedback is stale')
        with self.assertRaisesRegex(ValueError, 'body feedback is stale'):
            plan(plant, observation, include_offer=False)
        self.assertEqual(plant.solve.call_count, 1)
        plant.move.assert_not_called()
        plant.hand.assert_not_called()

    def test_collision_candidates_remain_retryable(self):
        from g1_can_handoff.errors import IKError
        observation=SimpleNamespace(center=np.array([.35,.18,.823]),
                                    table_height=.762,timestamp=time.monotonic())
        plant=Mock()
        plant.solve.side_effect=[IKError('Target pose collision: table')]+[np.zeros(7)]*4
        plan(plant,observation,include_offer=False)
        self.assertEqual(plant.solve.call_count,5)
        plant.move.assert_not_called()


if __name__ == '__main__':
    unittest.main()
