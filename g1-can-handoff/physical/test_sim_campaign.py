"""Campaign pairing and denominator integrity without repeating dynamics."""

import unittest
from offline.sim_campaign import cases, summarize, wilson


class CampaignTests(unittest.TestCase):
    def test_design_has_unique_fresh_seeds_and_separate_stress_cases(self):
        design = cases()
        self.assertEqual(len(design), 64)
        self.assertEqual(len({c["seed"] for c in design}), 64)
        self.assertEqual(sum(c["scale"] == 2.0 for c in design), 16)
        self.assertTrue(all(abs(x) <= 0.015 for c in design for x in c["offset_xy_m"]))

    def test_missing_pair_is_not_counted_as_a_paired_win(self):
        def row(case, policy, passed, condition="fitted_noise"):
            return dict(
                case_id=case,
                policy=policy,
                condition=condition,
                lift_verified=passed,
                error=None if passed else "stop",
                simulated_time_s=50.0,
                max_tracking_error_rad=0.03,
            )

        report = summarize(
            [
                row("a", "legacy", False),
                row("a", "tuned", True),
                row("b", "tuned", True),
                row("c", "legacy", True, "double_noise"),
                row("c", "tuned", False, "double_noise"),
            ]
        )
        self.assertEqual(report["paired_outcomes"], dict(tuned_only=1, legacy_only=1))
        self.assertEqual(report["groups"]["fitted_noise"]["tuned"]["trials"], 2)
        self.assertEqual(report["groups"]["double_noise"]["tuned"]["lifts"], 0)

    def test_all_pass_interval_retains_nonzero_failure_uncertainty(self):
        low, high = wilson(32, 32)
        self.assertLess(low, 0.90)
        self.assertAlmostEqual(high, 1.0)
        self.assertIsNone(wilson(0, 0))


if __name__ == "__main__":
    unittest.main()
