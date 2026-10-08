"""Collision-aware IK must recover a rejected real-geometry segment offline."""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from offline.planner_replay import fixture


class TableClearanceIKTests(unittest.TestCase):
    def test_endpoint_collision_recovers_a_different_redundant_solution(self):
        root = Path(__file__).parent
        capture = json.loads((root/'fixtures/table-clearance-ik.json').read_text())
        plant, _, _, _ = fixture(3, root/'offline/evidence/clearance-demos-20261008-preserved')
        endpoint = capture['endpoint']
        plant.ik.data.qpos[:] = endpoint['qpos']
        plant.set_contact_phase(endpoint['phase'])
        mujoco.mj_forward(plant.ik.model, plant.ik.data)
        before = plant.ik.data.qpos.copy()
        position, rotation = np.array(endpoint['position']), np.array(endpoint['rotation'])
        with self.assertRaisesRegex(ValueError, 'Target pose collision:.*table'):
            plant.ik.solve(position, rotation)
        goal = plant._pose_ik(position, rotation)
        self.assertIsNone(plant.ik.collision_details(plant.ik.ik_data))
        self.assertLessEqual(np.linalg.norm(plant.ik.ik_data.site_xpos[plant.ik.site]-position), .008)
        self.assertLessEqual(np.linalg.norm(.15*(plant.ik.ik_data.site_xmat[plant.ik.site].reshape(3, 3)-rotation)), .02)
        limits = np.asarray(plant.config['arm_limits_rad'])[:7]
        self.assertTrue(np.all(goal >= limits[:, 0]))
        self.assertTrue(np.all(goal <= limits[:, 1]))
        np.testing.assert_array_equal(plant.ik.data.qpos, before)
        self.assertIsNone(plant.adapter)
        self.assertIsNone(plant.sender)

    def test_alignment_recovers_finger_clearance_without_changing_the_grasp(self):
        root = Path(__file__).parent
        capture = json.loads((root/'fixtures/table-clearance-ik.json').read_text())
        plant, _, _, _ = fixture(3, root/'offline/evidence/clearance-demos-20261008-preserved')
        segment = capture['segments'][0]
        plant.ik.data.qpos[:] = segment['qpos']
        plant.set_contact_phase(segment['phase'])
        mujoco.mj_forward(plant.ik.model, plant.ik.data)
        before = plant.ik.data.qpos.copy()
        goal = plant._continuous_lift_ik(np.array(segment['position']), np.array(segment['rotation']))
        _, poses = plant._joint_poses(goal, .1)
        table = plant.ik.model.geom('table').id
        geoms = [g for g in range(plant.ik.model.ngeom)
                 if plant.ik.model.geom_contype[g] and
                 plant.ik.model.body(plant.ik.model.geom_bodyid[g]).name.startswith('left_')]
        data = mujoco.MjData(plant.ik.model)
        for q in poses:
            data.qpos[:] = before
            data.qpos[plant.ik.qadr[plant.ik.arm]] = q
            mujoco.mj_forward(plant.ik.model, data)
            self.assertGreaterEqual(min(mujoco.mj_geomDistance(plant.ik.model, data, table, g, .1, None)
                                        for g in geoms), .01)
        self.assertLessEqual(np.linalg.norm(data.site_xpos[plant.ik.site]-segment['position']), .008)
        self.assertLessEqual(np.linalg.norm(.15*(data.site_xmat[plant.ik.site].reshape(3, 3)-segment['rotation'])), .02)
        self.assertLessEqual(np.max(np.abs(goal-before[plant.ik.qadr[plant.ik.arm]])), .2000001)
        np.testing.assert_array_equal(plant.ik.data.qpos, before)
        self.assertIsNone(plant.adapter)
        self.assertIsNone(plant.sender)

    def test_high_table_wrist_collision_recovers_a_clear_complete_path(self):
        import copy
        import time
        from pickup import prepare
        from physical_core import PhysicalPlant, calibrated_scene
        from offline.planner_replay import clearance_auditor
        root=Path(__file__).parent
        captured,observation,_,events=fixture(3,root/'offline/evidence/clearance-demos-20261008-preserved')
        config=copy.deepcopy(captured.config)
        config['table_height_m']=.95
        observation.center[:]=[.35,.10,.95+config['can_height_m']/2]
        observation.table_height=.95
        plant=PhysicalPlant(calibrated_scene(config,.35,.10),None,None,
                            captured.read,config,captured.event,clock=captured.clock)
        before=plant.ik.data.qpos.copy()
        audit=clearance_auditor(plant)
        def observe(*args):
            observation.timestamp=time.monotonic()
            return observation,{'source':'synthetic_high_table_with_captured_posture'}
        prepare(plant,observe,plant.event)
        self.assertTrue(plant.path_preflight_ok)
        self.assertGreater(audit['samples'],100)
        self.assertGreaterEqual(audit['min_table_distance_m'],.01)
        np.testing.assert_array_equal(plant.ik.data.qpos,before)
        self.assertIsNone(plant.adapter)
        self.assertIsNone(plant.sender)


if __name__ == '__main__':
    unittest.main()
