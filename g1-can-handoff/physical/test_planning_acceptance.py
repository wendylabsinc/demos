"""Native failed-scene acceptance: actual IK, every sampled path, no DDS."""

import unittest
import time
import importlib.util
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
from offline.planner_replay import fixture, clearance_auditor
from pickup import prepare

ROOT = Path(__file__).parent / "offline/evidence/clearance-demos-20261008-preserved"


class PlanningAcceptanceTests(unittest.TestCase):
    def test_recorded_scene_is_red_with_original_planner_and_green_with_current(self):
        from offline.planning_failures_replay import run

        path = (
            Path(__file__).parent
            / "offline/validation/issue-planning-20261008/source_task_11a582c.py"
        )
        spec = importlib.util.spec_from_file_location("recorded_task_11a582c", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        with patch("pickup.plan", old.plan):
            red = run(5, historical=True)
        self.assertFalse(red["passed"])
        self.assertIn(
            "preflight rejected: Interpolated physical trajectory intersects",
            red["error"],
        )
        green = run(5)
        self.assertTrue(green["passed"], green.get("error"))
        self.assertGreaterEqual(green["clearance"]["min_table_distance_m"], 0.01)
        self.assertGreater(green["clearance"]["min_elbow_torso_distance_m"], 0)
        self.assertFalse(green["actuation"])

    def test_without_full_path_validator_unreachable_endpoints_still_reject(self):
        from g1_can_handoff.task import plan

        plant = Mock(spec=["solve", "set_contact_phase", "move", "hand"])
        plant.solve.side_effect = ValueError("Target cannot be reached")
        observation = SimpleNamespace(
            center=np.array([0.35, 0.18, 0.823]),
            table_height=0.762,
            timestamp=time.monotonic(),
        )
        with self.assertRaisesRegex(ValueError, "No reachable grasp"):
            plan(plant, observation, include_offer=False)
        self.assertGreater(plant.solve.call_count, 0)
        plant.move.assert_not_called()
        plant.hand.assert_not_called()

    def test_full_path_rejection_never_returns_a_plan(self):
        from g1_can_handoff.task import plan

        plant = Mock(spec=["solve", "set_contact_phase", "move", "hand"])
        observation = SimpleNamespace(
            center=np.array([0.35, 0.18, 0.823]),
            table_height=0.762,
            timestamp=time.monotonic(),
        )
        validate = Mock(return_value=False)
        with self.assertRaisesRegex(ValueError, "No reachable grasp"):
            plan(plant, observation, validate=validate, include_offer=False)
        self.assertEqual(validate.call_count, 30)
        plant.move.assert_not_called()
        plant.hand.assert_not_called()

    def test_continuous_full_path_validator_selects_branch_without_disconnected_endpoint_search(
        self,
    ):
        plant, observation, _, logged = fixture(3, ROOT)
        report = clearance_auditor(plant)
        observation.timestamp = time.monotonic()
        # Physical preflight owns the sequential IK branch and validates every
        # intermediate arm/finger pose. A separate per-endpoint multi-seed
        # search must not discard that fully checked continuous branch.
        with patch.object(
            plant,
            "solve",
            side_effect=AssertionError("Redundant disconnected endpoint solve"),
        ):
            prepare(
                plant,
                lambda *args: (observation, {"source": "saved_scene_replay"}),
                plant.event,
            )
        self.assertTrue(plant.path_preflight_ok)
        self.assertGreater(report["samples"], 100)
        self.assertGreaterEqual(report["min_table_distance_m"], 0.01)
        self.assertGreater(report["min_elbow_torso_distance_m"], 0)
        self.assertTrue(any(e["phase"] == "cartesian_transfer_planned" for e in logged))
        self.assertIsNone(plant.adapter)
        self.assertIsNone(plant.sender)


if __name__ == "__main__":
    unittest.main()
