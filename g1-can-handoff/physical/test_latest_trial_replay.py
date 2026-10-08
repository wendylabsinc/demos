"""Replay the real 00:59 trial's periodic-YOLO gap at the motor write gate."""
import gzip
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from physical_core import PhysicalPlant
from runtime import Runtime

ROOT=Path(__file__).parent/'offline/evidence/20261008T005915Z'


class LatestTrialTests(unittest.TestCase):
    def test_recorded_recheck_arrives_after_timeout_and_cannot_resume_motor_writes(self):
        rows=[json.loads(line) for line in gzip.decompress((ROOT/'events.jsonl.gz').read_bytes()).splitlines()]
        failure=next(e for e in rows if e['phase']=='primary_failure')
        self.assertIn('401.7 ms',failure['error'])
        results=[e for e in rows if e['phase']=='perception_result' and e.get('geometry')]
        before=[e for e in results if e['monotonic_ns']<failure['monotonic_ns']][-1]
        late=[e for e in results if e['monotonic_ns']>failure['monotonic_ns']][0]
        self.assertEqual((before['frame_id'],late['frame_id']),(62257,62259))
        self.assertEqual(late['geometry']['tracking']['reason'],'periodic_recheck')
        self.assertLess(late['exposure_age_ms'],400.)
        self.assertLess((late['monotonic_ns']-failure['monotonic_ns'])/1e6,12.)
        camera=next(e for e in rows if e['phase']=='phase_camera_evidence' and e['metadata']['frame_id']==before['frame_id'])
        timestamp=camera['metadata']['depth_time'];stop_at=failure['monotonic_ns']/1e9
        self.assertGreater((stop_at-timestamp)*1000,400.)
        runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock();runtime.tracking_error=None
        runtime.tracking_observation=SimpleNamespace(timestamp=timestamp)
        plant=PhysicalPlant.__new__(PhysicalPlant);plant.cancelled=False;plant.cancel=threading.Event()
        plant.perception_guard=runtime.tracking_guard;plant.command_lock=threading.RLock();plant.clock=time
        plant.command_arm=Mock();plant.sender=Mock();plant.read=Mock()
        with patch('runtime.time.monotonic',return_value=before['monotonic_ns']/1e9):runtime.tracking_guard()
        with patch('runtime.time.monotonic',return_value=stop_at):
            with self.assertRaisesRegex(RuntimeError,'tracking exposure stale'):
                plant._execute_poses(np.zeros(7),[np.zeros(7)])
        # Freeze latches cancellation. A late, individually fresh result
        # cannot clear that latch and resume either arm or finger writes.
        plant.cancelled=True
        camera=next(e for e in rows if e['phase']=='phase_camera_evidence' and e['metadata']['frame_id']==late['frame_id'])
        runtime.tracking_observation=SimpleNamespace(timestamp=camera['metadata']['depth_time'])
        with patch('runtime.time.monotonic',return_value=late['monotonic_ns']/1e9):
            runtime.tracking_guard()
            with self.assertRaisesRegex(RuntimeError,'cancelled'):
                plant._execute_poses(np.zeros(7),[np.zeros(7)])
        plant.command_arm.assert_not_called();plant.sender.ctrl_dual_hand.assert_not_called();plant.read.assert_not_called()


if __name__=='__main__':unittest.main()
