"""Phase scheduling regression, including the captured empty-YOLO failure.

This tests scheduling and retained robot guards, not model accuracy on a new
view or a physical pickup. The complete original event archive stays immutable.
"""
import gzip
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from perception_policy import perception_policy,semantic_recheck_interval_s
from perception_worker import worker_main
from physical_core import PhysicalPlant
from runtime import Runtime


class PerceptionPolicyTests(unittest.TestCase):
    def test_default_phase_policy_and_explicit_continuous_rollback(self):
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(perception_policy(),'phase_boundary')
            self.assertIsNone(semantic_recheck_interval_s())
        with patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY='continuous'):
            self.assertEqual(semantic_recheck_interval_s(),1.)
        with patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY='typo'):
            with self.assertRaisesRegex(ValueError,'PERCEPTION_POLICY'):perception_policy()

    def runtime(self):
        result=Runtime.__new__(Runtime)
        result.start_tracking=Mock();result.stop_tracking=Mock()
        return result

    def test_phase_policy_starts_no_background_perception_and_detaches_old_guard(self):
        runtime=self.runtime();plant=Mock()
        runtime.tracking_baseline=object();runtime.tracking_error='old failure'
        with patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY='phase_boundary'):
            runtime.configure_motion_perception(plant,{},object())
        runtime.start_tracking.assert_not_called()
        runtime.stop_tracking.assert_called_once()
        self.assertIsNone(plant.perception_guard)
        self.assertIsNone(runtime.tracking_baseline)
        self.assertIsNone(runtime.tracking_error)

    def test_continuous_rollback_retains_original_tracking_guard(self):
        runtime=self.runtime();plant=Mock();config={};baseline=object()
        with patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY='continuous'):
            runtime.configure_motion_perception(plant,config,baseline)
        runtime.start_tracking.assert_called_once_with(config,baseline)
        self.assertEqual(plant.perception_guard,runtime.tracking_guard)

    def test_spawned_worker_uses_same_policy_as_parent(self):
        for policy,interval in [('phase_boundary',None),('continuous',1.)]:
            with self.subTest(policy=policy),patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY=policy),\
                 patch('camera_input.Camera'),patch('target_tracker.TargetTracker') as tracker,\
                 patch('latest_frame.LatestFrameBuffer'):
                connection=Mock();connection.recv.return_value=None
                worker_main(connection,None)
                tracker.assert_called_once_with(recheck_interval_s=interval)

    def test_captured_recognition_failure_is_no_longer_a_mid_phase_command_guard(self):
        archive=Path(__file__).parent/'offline/evidence/issue-batch-demo-20261008-preserved/trial-01'
        raw=(archive/'events.jsonl.gz').read_bytes()
        expected=json.loads((archive/'manifest.json').read_text())['files']['events.jsonl.gz']['sha256']
        self.assertEqual(hashlib.sha256(raw).hexdigest(),expected)
        rows=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
        fault=next(row for row in rows if row['phase']=='perception_result' and row.get('error') and row.get('source',{}).get('frame_id')==97662)
        self.assertEqual(fault['detections'],[])
        self.assertIn('Coke YOLO target absent/ambiguous',fault['error'])
        for policy,blocked in [('continuous',True),('phase_boundary',False)]:
            with self.subTest(policy=policy),patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY=policy):
                runtime=self.runtime();runtime.lock=threading.RLock()
                runtime.tracking_error=None
                runtime.tracking_observation=SimpleNamespace(timestamp=100.)
                plant=PhysicalPlant.__new__(PhysicalPlant)
                plant.cancelled=False;plant.cancel=threading.Event()
                plant.clock=SimpleNamespace(monotonic=lambda:100.)
                plant.base_joints=[0]*15;plant.base_rpy=[0]*3;plant.right_arm=[0]*7
                plant.sender=None
                feedback={name:{'time':100.,'q':[0]*n,'dq':[0]*n,'tau_est':[0]*n,
                    'imu_rpy':[0]*3,'faults':[0]*n,'temperature_max':35}
                    for name,n in [('body',29),('left',7),('right',7)]}
                plant.read=lambda:feedback
                runtime.configure_motion_perception(plant,{},object())
                # Inject the native late asynchronous error at the write gate.
                # In phase mode no producer is scheduled to generate this error.
                runtime.tracking_error=fault['error']
                if blocked:
                    with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):plant.check()
                else:
                    plant.check()
                    feedback['body']['time']=99.
                    with self.assertRaisesRegex(ValueError,'Stale body'):plant.check()
                # Cancellation must still reject the next write in both modes.
                plant.cancel.set()
                with self.assertRaisesRegex(RuntimeError,'cancel'):plant.check()


if __name__=='__main__':unittest.main()
