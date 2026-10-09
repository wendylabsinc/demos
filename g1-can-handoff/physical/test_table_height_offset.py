"""The assumed table correction raises geometry and targets, not camera data."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from physical_core import calibrated_scene
from g1_can_handoff.task import plan
from pickup import prepare


class TableHeightOffsetTests(unittest.TestCase):
    def config(self):
        return dict(pelvis_to_world=np.eye(4).tolist(),table_height_m=.762,
                    table_center_xy_m=[.48,.18],table_half_size_xy_m=[.25,.32])

    def targets(self, model):
        observation=SimpleNamespace(center=np.array([.31,.15,.823]),
                                    table_height=.762,timestamp=100.)
        plant=SimpleNamespace(planning_table_height_offset_m=model.planning_table_height_offset_m,
                              set_contact_phase=lambda phase:None)
        with patch('g1_can_handoff.task.time.monotonic',return_value=100.):
            points,_=plan(plant,observation,validate=lambda *args:True,include_offer=False)
        np.testing.assert_array_equal(observation.center,[.31,.15,.823])
        self.assertEqual(observation.table_height,.762)
        return np.array(points)

    def test_default_raises_table_and_all_targets_by_one_cm(self):
        config=self.config()
        with patch.dict('os.environ',HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M='0.01'):
            raised=calibrated_scene(config)
        baseline=calibrated_scene(dict(self.config(),planning_table_height_offset_m=0.))
        table=raised.model.geom('table')
        self.assertAlmostEqual(table.pos[2]+table.size[2],.772)
        self.assertEqual(config['table_height_m'],.762)
        np.testing.assert_allclose(self.targets(raised)-self.targets(baseline),
                                   np.tile([0,0,.01],(5,1)),atol=1e-12)
        # The measured can proxy is not silently translated with the table.
        np.testing.assert_allclose(raised.can_position(),baseline.can_position())

    def test_replay_is_pinned_and_does_not_accumulate_offset(self):
        config=self.config()
        with patch.dict('os.environ',HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M='.01'):
            calibrated_scene(config)
        with patch.dict('os.environ',HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M='0'):
            replay=calibrated_scene(config)
        self.assertAlmostEqual(replay.model.geom('table').pos[2]+.025,.772)
        self.assertEqual(config['planning_table_height_m'],.772)

    def test_zero_restores_original_table_and_grasp(self):
        model=calibrated_scene(dict(self.config(),planning_table_height_offset_m=0.))
        self.assertAlmostEqual(model.model.geom('table').pos[2]+.025,.762)
        np.testing.assert_allclose(self.targets(model)[2],[.31,.15,.823])

    def test_seeded_sim_raises_only_controller_table_not_dynamics_world(self):
        from sim_runtime import setup_scene
        config=dict(self.config(),camera_to_world=np.eye(4).tolist(),
                    can_radius_m=.033,can_height_m=.122)
        setup=dict(config=config,can_center_world_m=[.31,.15,.823],
                   camera_intrinsics=dict(width=640,height=480,fx=500.,fy=500.,
                                          ppx=319.5,ppy=239.5))
        with patch.dict('os.environ',HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M='.01'):
            world=setup_scene(setup)
            controller=setup_scene(setup,planning=True)
        self.assertAlmostEqual(world.model.geom('table').pos[2]+.025,.762)
        self.assertAlmostEqual(controller.model.geom('table').pos[2]+.025,.772)
        np.testing.assert_allclose(world.can_position(),controller.can_position())
        self.assertNotIn('planning_table_height_offset_m',config)

    def test_prepare_validates_raised_path_but_checks_raw_observations(self):
        observation=SimpleNamespace(center=np.array([.31,.15,.823]),
                                    table_height=.762,timestamp=100.)
        validated=[];events=[]
        plant=SimpleNamespace(planning_table_height_offset_m=.01,
                              observe_can=lambda observation:None,
                              set_contact_phase=lambda phase:None,
                              preflight=lambda points,rotation:validated.append(np.array(points)))
        with patch('g1_can_handoff.task.time.monotonic',return_value=100.):
            prepared=prepare(plant,lambda *args:(observation,{}),
                             lambda phase,**fields:events.append((phase,fields)))
        self.assertIs(prepared[0],observation)
        self.assertAlmostEqual(validated[0][2][2],.833)
        fields=next(fields for phase,fields in events if phase=='plan')
        self.assertEqual(fields['measured_table_height_m'],.762)
        self.assertEqual(fields['planning_table_height_offset_m'],.01)
        self.assertEqual(fields['planning_table_height_m'],.772)

    def test_invalid_offset_and_adjusted_workspace_height_rejected(self):
        for value in ('nan','inf','-.001','.031','invalid'):
            with self.subTest(value=value),patch.dict('os.environ',HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M=value):
                with self.assertRaisesRegex(ValueError,'table height offset'):
                    calibrated_scene(self.config())
        with self.assertRaisesRegex(ValueError,'supported'):
            calibrated_scene(dict(self.config(),table_height_m=.95,planning_table_height_offset_m=.01))


if __name__=='__main__':unittest.main()
