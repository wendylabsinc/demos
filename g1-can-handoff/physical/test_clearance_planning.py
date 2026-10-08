"""Clearance and interpolation regressions; no hardware or command output."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock,patch
import mujoco
import numpy as np
from g1_can_handoff.sim import Plant
from physical_core import PhysicalPlant
from offline.planner_replay import fixture,ROOT,clearance_auditor


class ClearancePlanningTests(unittest.TestCase):
    def setUp(self):
        environment=patch.dict(os.environ,{'HANDOFF_TABLE_CLEARANCE_M':'.01'})
        environment.start();self.addCleanup(environment.stop)

    def probe(self,gap):
        model=mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
          <geom name="table" type="box" size=".1 .1 .025" margin=".01" gap=".01"/>
          <body name="left_hand_index_1_link" pos="0 0 {gap+.025+.006}">
            <freejoint/><geom type="sphere" size=".006"/></body></worldbody></mujoco>''')
        plant=Plant.__new__(Plant);plant.model=model;plant.contact_phase='reach';plant.table_clearance_m=.01
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        return plant,data

    def test_positive_table_gap_under_clearance_is_rejected_in_every_phase(self):
        for gap in (-.0014,0.,.005,.0099):
            plant,data=self.probe(gap)
            for phase in ('pregrasp','staging','approach','align','reach','close','lift','hold','offer'):
                with self.subTest(gap=gap,phase=phase):
                    self.assertGreater(data.ncon,0)
                    plant.set_contact_phase(phase)
                    self.assertIsNotNone(plant.collision_details(data))

    def test_table_gap_above_clearance_passes(self):
        plant,data=self.probe(.0101)
        self.assertIsNone(plant.collision_details(data))

    def test_align_and_reach_use_continuous_cartesian_paths(self):
        plant=PhysicalPlant.__new__(PhysicalPlant);plant.ik=Mock()
        plant._cartesian_transfer_poses=Mock(return_value='continuous')
        plant._joint_poses=Mock(return_value='joint')
        for phase in ('align','reach'):
            plant.ik.contact_phase=phase
            self.assertEqual(plant._move_poses(np.ones(3)),'continuous')
        plant._joint_poses.assert_not_called()

    def test_recorded_table_collision_still_rejected_with_clearance(self):
        plant,_,events,_=fixture()
        collision=next(e for e in reversed(events) if e['phase']=='trajectory_collision')
        plant.ik.data.qpos[plant.ik.qadr[plant.ik.fingers]]=0.
        plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]]=collision['joint_q']
        plant.set_contact_phase(collision['phase_name'])
        mujoco.mj_forward(plant.ik.model,plant.ik.data)
        detail=plant.ik.collision_details(plant.ik.data)
        self.assertIsNotNone(detail)
        self.assertIn('table',detail['geoms'])
        self.assertLess(detail['distance_m'],.01)

    def test_clearance_changes_proximity_checks_without_changing_measured_table(self):
        plant,observation,_,_=fixture()
        geom=plant.ik.model.geom('table')
        self.assertAlmostEqual(geom.pos[2]+geom.size[2],observation.table_height)
        self.assertAlmostEqual(geom.margin[0],.01)
        self.assertAlmostEqual(geom.gap[0],.01)

    def test_invalid_clearance_cannot_disable_table_checks(self):
        for value in ('0','-1','nan','inf','garbage','.031'):
            with self.subTest(value=value),patch.dict(os.environ,{'HANDOFF_TABLE_CLEARANCE_M':value}):
                with self.assertRaisesRegex(ValueError,'Table clearance'):fixture()

    def test_all_saved_frames_events_and_sources_are_hash_verified(self):
        for directory in sorted(ROOT.glob('trial-*')):
            manifest=json.loads((directory/'manifest.json').read_text())
            for name,metadata in manifest['files'].items():
                with self.subTest(trial=directory.name,file=name):
                    raw=(directory/name).read_bytes()
                    self.assertEqual(len(raw),metadata['bytes'])
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),metadata['sha256'])
            events=[json.loads(line) for line in gzip.decompress((directory/'events.jsonl.gz').read_bytes()).decode().splitlines()]
            self.assertTrue(any(e['phase']=='primary_failure' for e in events))
            self.assertFalse(any(e['phase']=='lift_verified' for e in events))

    def test_saved_scene_replans_all_four_phases_with_positive_clearance(self):
        from pickup import prepare
        from unittest.mock import patch
        for index in (1,2,3):
            with self.subTest(trial=index):
                plant,observation,_,logged=fixture(index)
                before=plant.ik.data.qpos.copy()
                self.assertIsNone(plant.adapter);self.assertIsNone(plant.sender)
                report=clearance_auditor(plant)
                with patch('g1_can_handoff.task.time.monotonic',return_value=100.):
                    _,_,points,rotation=prepare(plant,lambda *args:(observation,dict(source='saved_scene_replay')),plant.event)
                self.assertTrue(plant.path_preflight_ok)
                np.testing.assert_array_equal(plant.ik.data.qpos,before)
                self.assertGreaterEqual(points[2][2]-observation.center[2],0.)
                self.assertLessEqual(points[2][2]-observation.center[2],.0200001)
                self.assertGreater(report['samples'],100)
                self.assertGreaterEqual(report['min_table_distance_m'],.01)
                self.assertTrue(any(e['phase']=='cartesian_transfer_planned' for e in logged))

    def test_near_centre_hardware_failure_replans_without_elbow_torso_collision(self):
        from pickup import prepare
        root=Path(__file__).parent/'offline/evidence/clearance-demos-20261008-preserved'
        plant,observation,_,logged=fixture(3,root)
        before=plant.ik.data.qpos.copy()
        report=clearance_auditor(plant)
        with patch('g1_can_handoff.task.time.monotonic',return_value=100.):
            prepare(plant,lambda *args:(observation,dict(source='saved_scene_replay')),plant.event)
        self.assertTrue(plant.path_preflight_ok)
        self.assertGreater(report['samples'],100)
        self.assertGreaterEqual(report['min_table_distance_m'],.01)
        self.assertGreater(report['min_elbow_torso_distance_m'],0.)
        np.testing.assert_array_equal(plant.ik.data.qpos,before)
        self.assertIsNone(plant.adapter);self.assertIsNone(plant.sender)

    def test_saved_head_camera_offset_is_still_rejected(self):
        from camera_input import geometry
        path=Path(__file__).parent/'offline/evidence/head-camera-offset-20261008/frame.npz'
        with np.load(path,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata_json']))
            frame={k:data[k].copy() for k in ('color_bgr','depth_m','rays')}
        frame.update(meta=metadata['meta'],timestamp=metadata['timestamp'])
        with self.assertRaisesRegex(RuntimeError,'over 40 mm'):
            geometry(frame,metadata['config'],require_table=metadata['require_table'],
                     expected_center_xy=metadata['expected_center_xy'],synthetic_red=True)

    def test_infeasible_lift_does_not_spend_staging_search_budget(self):
        plant,observation,_,logged=fixture()
        centre=observation.center+np.array([0,0,.02])
        points=[centre+[0,.1,.16],centre+[0,.1,0],centre,centre+[0,0,.15]]
        original=plant._cartesian_lift_poses
        def fail_lift(*args,**kwargs):
            if plant.ik.contact_phase=='lift':
                raise RuntimeError('Continuous Cartesian lift IK cannot preserve the grasp pose')
            return original(*args,**kwargs)
        plant._cartesian_lift_poses=fail_lift
        with patch('staging.joint_path') as search:
            with self.assertRaisesRegex(RuntimeError,'Continuous Cartesian lift IK'):
                plant.preflight(points,np.eye(3))
            search.assert_not_called()
        self.assertTrue(any(e['phase']=='trajectory_collision' for e in logged))
        self.assertFalse(plant.path_preflight_ok)


if __name__=='__main__':unittest.main()
