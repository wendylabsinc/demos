"""Faults must stop the same pickup sequence used by both backends."""
import unittest
import time
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from pickup import execute,prepare


class Plant:
    def __init__(self, contacts=('thumb','index')):
        self.actions=[];self.groups=contacts
    def observe_can(self,observation):pass
    def set_contact_phase(self,phase):pass
    def validate_start(self):pass
    def stage(self):pass
    def hand(self,*args):self.actions.append('hand')
    def move(self,*args,**kw):self.actions.append('move')
    def step(self,*args):self.actions.append('step')
    def contacts(self):return self.groups


class SharedPickupTests(unittest.TestCase):
    def setUp(self):
        self.obs=SimpleNamespace(center=np.array([.35,.18,.811]),table_height=.75)
        self.prepared=(self.obs,{'top_quantile_z_m':.86},
                       [self.obs.center.copy() for _ in range(4)],np.eye(3))
    def observe(self,phase,require_table,expected_center_xy=None):
        return self.obs,dict(top_quantile_z_m=1.01,center_xy_m=[.35,.18])
    def test_full_path_rejection_searches_another_orientation(self):
        self.obs.timestamp=time.monotonic()
        plant=Mock();plant.preflight.side_effect=[RuntimeError('Continuous Cartesian lift IK cannot preserve the grasp pose'),None]
        events=[]
        result=prepare(plant,self.observe,lambda phase,**kw:events.append(phase))
        self.assertEqual(plant.preflight.call_count,2)
        self.assertGreater(result[2][2][2],self.obs.center[2])
        self.assertIn('grasp_candidate_rejected',events)
        plant.move.assert_not_called()

    def test_orientation_search_never_swallows_operational_faults(self):
        for reason in ('Operator cancelled trial','body temperature reaches guard','Base attitude moved'):
            self.obs.timestamp=time.monotonic()
            plant=Mock();plant.preflight.side_effect=RuntimeError(reason)
            with self.assertRaisesRegex(RuntimeError,reason):prepare(plant,self.observe,lambda *a,**k:None)
            self.assertEqual(plant.preflight.call_count,1)

    def test_orientation_candidates_are_bounded(self):
        self.obs.timestamp=time.monotonic()
        plant=Mock();plant.preflight.side_effect=RuntimeError('Interpolated physical trajectory intersects robot/table')
        with self.assertRaisesRegex(ValueError,'No reachable.*preflight rejected: Interpolated physical trajectory intersects robot/table'):
            prepare(plant,self.observe,lambda *a,**k:None)
        self.assertEqual(plant.preflight.call_count,30)
        plant.move.assert_not_called()
    def test_lift_requires_camera_evidence_even_with_finger_obstructions(self):
        plant=Plant()
        def static(phase,require_table,expected_center_xy=None):
            return self.obs,dict(top_quantile_z_m=.86,center_xy_m=[.35,.18])
        with self.assertRaisesRegex(RuntimeError,'camera'):
            execute(plant,self.prepared,static,lambda *a,**kw:None)
    def test_can_motion_aborts_before_first_arm_command(self):
        plant=Plant()
        moved=SimpleNamespace(center=self.obs.center+[.02,0,0],table_height=.75)
        with self.assertRaisesRegex(RuntimeError,'moved'):
            execute(plant,self.prepared,lambda *a:(moved,{}),lambda *a,**kw:None)
        self.assertNotIn('move',plant.actions)
    def test_can_pushed_during_reach_stops_before_closing(self):
        plant=Plant()
        moved=SimpleNamespace(center=self.obs.center+[.02,0,0],table_height=.75)
        def observe(phase,require_table,expected_center_xy=None):
            return (moved if phase=='pre-close' else self.obs),{}
        with self.assertRaisesRegex(RuntimeError,'refusing to close'):
            execute(plant,self.prepared,observe,lambda *a,**kw:None)
        self.assertEqual(plant.actions.count('hand'),1)

    def test_missing_obstruction_aborts_before_lift(self):
        plant=Plant(contacts=('thumb',))
        with self.assertRaisesRegex(RuntimeError,'obstruction'):
            execute(plant,self.prepared,self.observe,lambda *a,**kw:None)
        self.assertEqual(plant.actions.count('move'),3)
    def test_transient_finger_obstructions_do_not_authorize_lift(self):
        plant=Plant();samples=iter([('thumb','index'),('thumb','index'),()] * 7)
        plant.contacts=lambda:next(samples)
        with self.assertRaisesRegex(RuntimeError,'stable finger obstruction'):
            execute(plant,self.prepared,self.observe,lambda *a,**kw:None)
        self.assertEqual(plant.actions.count('move'),3)

    def test_slip_after_initial_lift_does_not_report_success(self):
        plant=Plant()
        def slipping(phase,require_table,expected_center_xy=None):
            height=.86 if phase=='hold-verification-2' else 1.01
            return self.obs,dict(top_quantile_z_m=height,center_xy_m=[.35,.18])
        with self.assertRaisesRegex(RuntimeError,'lift and hold'):
            execute(plant,self.prepared,slipping,lambda *a,**kw:None)

    def test_replayed_camera_frame_does_not_report_held_can(self):
        plant=Plant()
        def replay(phase,require_table,expected_center_xy=None):
            return self.obs,dict(top_quantile_z_m=1.01,center_xy_m=[.35,.18],frame_id=123)
        with self.assertRaisesRegex(RuntimeError,'repeated a frame'):
            execute(plant,self.prepared,replay,lambda *a,**kw:None)

    def test_shifted_grip_outside_40mm_fails_shared_hold_verifier(self):
        plant=Plant()
        def shifted(phase,require_table,expected_center_xy=None):
            return self.obs,dict(top_quantile_z_m=1.01,center_xy_m=[.30,.18])
        with self.assertRaisesRegex(RuntimeError,'lift and hold'):
            execute(plant,self.prepared,shifted,lambda *a,**kw:None)

    def test_verified_lift_holds_without_offer_or_release(self):
        plant=Plant();events=[]
        result=execute(plant,self.prepared,self.observe,lambda phase,**kw:events.append(phase))
        self.assertTrue(result['lift_verified'])
        self.assertEqual(events,['approach','align','reach','close','lift']+['lift_observation']*5+['lift_verified'])
        self.assertEqual(plant.actions.count('hand'),2)
        self.assertEqual(plant.actions.count('move'),4)

if __name__=='__main__':unittest.main()
