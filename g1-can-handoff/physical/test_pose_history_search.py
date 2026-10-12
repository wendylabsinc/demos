"""Nearest sample tie, capacity and rejection equivalence at ordered boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
from pose_alignment import PoseHistory

class PoseHistorySearchTests(unittest.TestCase):
    def test_dyadic_midpoint_keeps_earlier_pose_on_tie(self):
        with patch.dict('os.environ', {'HANDOFF_POSE_ALIGNMENT_MAX_SKEW_MS':'40'}):
            history=PoseHistory()
        history.add(100.,[0]*3,[0]*3)
        history.add(100.0625,[1]*3,[0]*3)
        self.assertEqual(history.nearest(100.03125)['timestamp'],100.)

    def test_ordered_search_matches_scan_after_capacity_rollover(self):
        with patch.dict('os.environ', {'HANDOFF_POSE_ALIGNMENT_MAX_SKEW_MS':'40'}):
            history=PoseHistory(max_samples=512)
        stamps=100.+np.cumsum(np.random.default_rng(67).uniform(.001,.03,1100))
        for timestamp in stamps:
            history.add(timestamp,[0]*3,[0]*3)
        samples=stamps[-512:]
        for exposure in np.linspace(samples[0]-.041,samples[-1]+.041,1000):
            expected=min(samples,key=lambda timestamp:abs(timestamp-exposure))
            if abs(expected-exposure)*1000 >40+1e-9:
                with self.assertRaisesRegex(RuntimeError,'40 ms'):
                    history.nearest(exposure)
            else:
                self.assertEqual(history.nearest(exposure)['timestamp'],expected)
