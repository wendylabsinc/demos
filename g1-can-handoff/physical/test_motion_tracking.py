"""Fresh accepted tracking is required before the next physical command."""
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import Mock,patch

import numpy as np

os.environ.setdefault('HANDOFF_DATA',tempfile.mkdtemp(prefix='tracking-test-'))
from runtime import Runtime
from physical_core import PhysicalPlant


class MotionTrackingTests(unittest.TestCase):
    def runtime(self,age=.2,error=None):
        runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock()
        runtime.tracking_error=error
        runtime.tracking_observation=SimpleNamespace(timestamp=100-age)
        return runtime

    def test_fresh_observation_passes_and_stale_future_or_uncertain_blocks(self):
        with patch('runtime.time.monotonic',return_value=100):
            self.runtime(.399).tracking_guard()
            for age,error in [(.401,None),(-.001,None),(.1,'identity uncertain')]:
                with self.subTest(age=age,error=error),self.assertRaises(RuntimeError):
                    self.runtime(age,error).tracking_guard()

    def test_stale_tracking_blocks_arm_and_finger_writes_at_next_command_tick(self):
        runtime=self.runtime(.401)
        plant=PhysicalPlant.__new__(PhysicalPlant)
        plant.cancelled=False;plant.cancel=threading.Event();plant.perception_guard=runtime.tracking_guard
        plant.clock=time;plant.command_lock=threading.RLock();plant.read=Mock()
        plant.command_arm=Mock();plant.sender=Mock()
        with patch('runtime.time.monotonic',return_value=100),self.assertRaisesRegex(RuntimeError,'stale'):
            plant._execute_poses(np.zeros(7),[np.zeros(7)])
        plant.command_arm.assert_not_called();plant.sender.ctrl_dual_hand.assert_not_called()
        plant.read.assert_not_called()

    def test_one_frame_prefetch_overlaps_processing_without_rewriting_exposure(self):
        runtime=self.runtime();runtime.tracking_stop=threading.Event();runtime.tracking_thread=None
        runtime.pending_frame=None;runtime.frame_prefetch=ThreadPoolExecutor(max_workers=1)
        second_started=threading.Event();frames=[]
        def capture(**kwargs):
            frame={'frame_id':len(frames)+1,'timestamp':90+len(frames)*.1}
            frames.append(frame)
            if len(frames)==2:second_started.set()
            return frame
        runtime.camera=SimpleNamespace(frame=capture)
        try:
            first=runtime.next_tracking_frame()
            self.assertTrue(second_started.wait(1))
            self.assertIs(first,frames[0]);self.assertEqual(first['timestamp'],90)
            self.assertIs(runtime.next_tracking_frame(),frames[1])
        finally:runtime.stop_tracking()
        self.assertIsNone(runtime.pending_frame);self.assertIsNone(runtime.frame_prefetch)


if __name__=='__main__':unittest.main()
