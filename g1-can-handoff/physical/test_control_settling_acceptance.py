"""Acceptance at the real control-settling seam with immutable native feedback."""
import unittest

from offline.control_settling_replay import cases, negative_cases, run_case


class NativeControlSettlingAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = cases()

    def test_original_two_failures_reject_at_point_zero_three_and_settle_at_one_three_times(self):
        for repeat in range(3):
            for trial in self.trials:
                with self.subTest(repeat=repeat, trial=trial['trial']):
                    old = run_case(trial, .03)
                    self.assertFalse(old['settled'])
                    self.assertIn('within 5 seconds', old['error'])
                    new = run_case(trial, 1.)
                    self.assertTrue(new['settled'], new['error'])
                    self.assertGreaterEqual(new['elapsed_s'], .5)
                    self.assertLess(new['elapsed_s'], 5.)
                    self.assertLess(new['events'][0]['max_speed_rad_s'], 1.)
                    self.assertEqual(new['events'][0]['max_speed_limit_rad_s'], 1.)
                    self.assertGreater(new['archive_checks'], 0)

    def test_native_feedback_negative_controls_reject_each_body_joint_and_all_existing_guards(self):
        for trial in self.trials:
            for name, mutation, expected in negative_cases():
                with self.subTest(trial=trial['trial'], scenario=name):
                    result = run_case(trial, 1., mutation)
                    self.assertFalse(result['settled'])
                    self.assertIn(expected, result['error'])
                    self.assertFalse(result['events'])

    def test_real_hand_feedback_includes_motor_and_device_fault_words(self):
        for trial in self.trials:
            rows = trial['rows']
            self.assertEqual({row['topic'] for row in rows},
                             {'rt/lowstate', 'rt/dex3/left/state', 'rt/dex3/right/state'})
            for row in rows:
                self.assertTrue(row['cdr_base64'])
                self.assertEqual(len(row['sample']['faults']), 29 if row['topic'] == 'rt/lowstate' else 9)


if __name__ == '__main__':
    unittest.main()
