"""Recorded command curves must stay reproducible without robot access."""
import unittest
from offline.trajectory_parity import compare


class TrajectoryParityTests(unittest.TestCase):
    def test_captured_command_samples_replay_exactly(self):
        report=compare()
        self.assertEqual(report['compared_segments'],8)
        self.assertEqual(report['compared_samples'],1647)
        self.assertTrue(report['all_logged_targets_match'])
        self.assertFalse(report['physical_trajectory_verified'])
        for case in report['ik']:
            self.assertEqual(case['repeat_max_abs_error_rad'],0.)
            self.assertFalse(case['full_planning_snapshot_available'])


if __name__=='__main__':unittest.main()
