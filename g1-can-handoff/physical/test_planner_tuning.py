"""Candidate selection cannot reuse the wrong path or weaken physical bounds."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from g1_can_handoff.perception import Observation
from g1_can_handoff.task import plan
from planner_tuning import configure


class PlannerTuningTests(unittest.TestCase):
    def plant(self):
        return SimpleNamespace(ik=SimpleNamespace(IK_SETTINGS={"x_scale": 1.0}))

    def test_defaults_preserve_existing_policy_and_snapshots_capture_effective_settings(
        self,
    ):
        plant = self.plant()
        config = {}
        configure(plant, config)
        self.assertEqual(plant.ik.ik_reference_weight, 0.0)
        self.assertEqual(plant.ik.ik_position_tolerance, 0.012)
        self.assertEqual(plant.grasp_candidate_budget, 1)
        self.assertEqual(plant.ik.IK_SETTINGS["x_scale"], 1.0)
        self.assertEqual(config["grasp_orientation_step_rad"], 0.0)
        # A captured setting wins over later environment changes on replay.
        with patch.dict(
            "os.environ",
            HANDOFF_IK_REFERENCE_WEIGHT_M_PER_RAD=".005",
            HANDOFF_IK_POSITION_TOLERANCE_MM="1",
        ):
            configure(self.plant(), config)
        self.assertEqual(config["ik_reference_weight_m_per_rad"], 0.0)
        self.assertEqual(config["ik_position_tolerance_m"], 0.012)

    def test_invalid_settings_never_relax_bounds(self):
        for settings in (
            {"ik_reference_weight_m_per_rad": 0.01},
            {"ik_position_tolerance_m": 0.013},
            {"grasp_candidate_budget": 4},
            {"grasp_candidate_budget": True},
            {"grasp_orientation_step_rad": 0.4},
            {"ik_variable_scale": "random"},
            {"planning_position_cov_m2": np.diag([-1.0, 1.0, 0]).tolist()},
        ):
            with self.assertRaises(ValueError):
                configure(self.plant(), settings)

    def test_tuned_physical_profile_matches_sim_and_replays_without_environment(self):
        import json
        from pathlib import Path

        expected = json.loads(
            (
                Path(__file__).parent / "g1_can_handoff/noise_aware_planner.json"
            ).read_text()
        )
        config = {}
        with patch.dict("os.environ", HANDOFF_PLANNER_PROFILE="tuned"):
            configure(self.plant(), config)
        for key, value in expected.items():
            self.assertEqual(config[key], value)
        with patch.dict("os.environ", HANDOFF_PLANNER_PROFILE="legacy"):
            configure(self.plant(), config)
        self.assertEqual(config["planner_profile"], "tuned")
        self.assertEqual(config["grasp_candidate_budget"], 3)
        with self.assertRaisesRegex(ValueError, "profile"):
            configure(self.plant(), {"planner_profile": "typo"})

    def test_selects_and_revalidates_lowest_sensitivity_candidate(self):
        scored = []
        validated = []
        plant = SimpleNamespace(
            grasp_candidate_budget=3,
            grasp_orientation_step_rad=0.1,
            set_contact_phase=lambda phase: None,
            grasp_noise_score=lambda: (3.0, 1.0, 2.0)[len(scored) - 1],
        )

        def validate(points, rotation):
            validated.append(rotation.copy())
            scored.append(rotation.copy())
            return True

        observation = Observation(
            np.array([0.31, 0.15, 0.823]), 0.762, 100.0, 0.003, 100
        )
        with patch("g1_can_handoff.task.time.monotonic", return_value=100.0):
            points, rotation = plan(
                plant, observation, validate=validate, include_offer=False
            )
        self.assertEqual(len(validated), 4)
        np.testing.assert_array_equal(rotation, validated[1])
        np.testing.assert_array_equal(validated[-1], rotation)
        self.assertTrue(np.allclose(points[2], observation.center))

    def test_final_revalidation_cannot_overrun_original_planning_deadline(self):
        clock = {"time": 100.0}
        plant = SimpleNamespace(
            grasp_candidate_budget=2,
            grasp_orientation_step_rad=0.1,
            set_contact_phase=lambda phase: None,
            grasp_noise_score=lambda: 0.0,
        )
        observation = Observation(
            np.array([0.31, 0.15, 0.823]), 0.762, 100.0, 0.003, 100
        )
        calls = []

        def validate(*args):
            calls.append(1)
            if len(calls) == 3:
                clock["time"] += 30.001
            return True

        with patch(
            "g1_can_handoff.task.time.monotonic", side_effect=lambda: clock["time"]
        ):
            with self.assertRaisesRegex(ValueError, "search time limit"):
                plan(plant, observation, validate=validate, include_offer=False)

    def test_final_revalidation_failure_and_cancellation_do_not_return_a_path(self):
        plant = SimpleNamespace(
            grasp_candidate_budget=2,
            grasp_orientation_step_rad=0.1,
            set_contact_phase=lambda phase: None,
            grasp_noise_score=lambda: 0.0,
        )
        observation = Observation(
            np.array([0.31, 0.15, 0.823]), 0.762, 100.0, 0.003, 100
        )
        attempts = []

        def validate(*args):
            attempts.append(1)
            return len(attempts) < 3

        with patch("g1_can_handoff.task.time.monotonic", return_value=100.0):
            with self.assertRaisesRegex(ValueError, "final path validation"):
                plan(plant, observation, validate=validate, include_offer=False)

            def cancel(*args):
                raise RuntimeError("Operator cancelled trial")

            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                plan(plant, observation, validate=cancel, include_offer=False)


if __name__ == "__main__":
    unittest.main()
