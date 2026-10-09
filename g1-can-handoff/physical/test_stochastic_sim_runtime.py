"""The tuned simulator still uses noisy perception and checked physical paths."""

import json
from pathlib import Path
import unittest
from sim_runtime import run

ROOT = Path(__file__).parent


class StochasticRuntimeTests(unittest.TestCase):
    def test_tuned_full_simulation_lifts_with_fresh_noise_without_yolo_or_hardware(
        self,
    ):
        setup = json.loads(
            (ROOT / "offline/validation/position-noise-20261008/setup.json").read_text()
        )
        result = run(
            setup=setup,
            identity_mode="color_depth",
            noise_seed=200,
            planner_mode="tuned",
        )
        self.assertTrue(result["lift_verified"], result.get("error"))
        self.assertFalse(result["physical_pickup_verified"])
        self.assertEqual(result["perception"]["yolo_calls"], 0)
        self.assertEqual(result["position_noise"]["mode"], "stochastic")
        self.assertEqual(result["planner_settings"]["grasp_candidate_budget"], 3)
        self.assertEqual(result["planner_settings"]["ik_position_tolerance_m"], 0.005)
        selections = [
            e for e in result["events"] if e["phase"] == "noise_aware_grasp_selected"
        ]
        self.assertGreaterEqual(len(selections), 2)
        self.assertTrue(all(e["candidate_count"] >= 1 for e in selections))
        cameras = [
            e["vision"]["simulation_position_noise"]
            for e in result["events"]
            if e["phase"] == "camera"
        ]
        self.assertEqual(cameras[0]["seed"], 200)
        self.assertTrue(cameras[-1]["extrapolated"])
        self.assertGreater(len(result["position_noise"]["generated_jumps"]), 0)
        self.assertEqual(result["event_counts"]["lift_observation"], 5)
        self.assertLess(result["max_tracking_error_rad"], 0.15)
        self.assertEqual(
            result["simulation_contact_audit"]["pregrasp_contact_steps"], 0
        )


if __name__ == "__main__":
    unittest.main()
