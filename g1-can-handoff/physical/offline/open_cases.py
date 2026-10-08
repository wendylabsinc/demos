"""Opt-in red acceptance case: cold SDK acquisition must precede the final plan.

This tests Runtime.run's ownership/planning contract with fake actuator endpoints.
It intentionally fails on the deployed 1538fd1 runtime. It is not xfailed/skipped.
Recorded geometry/pose data are exercised separately by test_recorded_failures.
"""
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from runtime import Runtime

class ColdOwnershipAcceptance(unittest.TestCase):
    def test_pickup_execution_uses_a_plan_measured_after_sdk_ownership(self):
        subject=Runtime.__new__(Runtime)
        subject.arm=None;subject.hand=None;subject.plant=None;subject.active=False
        subject.phase='PASSIVE';subject.cancel=threading.Event();subject.lock=threading.RLock()
        subject.operation_lock=threading.Lock();subject.observation_lock=threading.RLock()
        subject.camera=Mock();subject.event=Mock();subject.read=Mock();subject.hub=Mock()
        subject.motion_commands_sent=0
        subject.archive_guard=Mock(return_value=dict(age_s={},session='offline'))
        subject.config=Mock(return_value=dict(arm_limits_rad=[]));subject.make_plant=Mock()
        subject.start_tracking=Mock();subject.stop_tracking=Mock();subject.stop_perception=Mock()
        subject.wait_for_control_settle=Mock()
        state={'owned':False};planned=[];executed=[]
        def acquire(**kwargs):state['owned']=True;return Mock()
        def prepare(*args):
            result=(SimpleNamespace(center=[.38627 if not state['owned'] else .39629,.118,.823]),{})
            planned.append(state['owned']);return result
        def execute(plant,prepared,*args):
            executed.append(prepared[0].center[0])
            return dict(lift_verified=False,visual_rise_m=0,hold_verified_s=0)
        modules={'g1_arm':SimpleNamespace(G1_29_ArmController=acquire),
                 'g1_dex3':SimpleNamespace(Dex3Sender=Mock())}
        with patch.dict(sys.modules,modules),patch('perception_worker.PerceptionWorker'),\
             patch('runtime.snapshot_guard'),patch('runtime.calibration_start_guard'),\
             patch('runtime.http_json',side_effect=ConnectionError),\
             patch('pickup.prepare',side_effect=prepare),patch('pickup.execute',side_effect=execute):
            subject.run(True)
        self.assertTrue(planned[-1],'Final plan was measured before cold SDK ownership changed robot pose')
        self.assertAlmostEqual(executed[-1],.39629,places=5)

if __name__=='__main__':unittest.main()
