"""Ownership acceptance cases: settle, refresh, and plan before execution.

This tests Runtime.run's ownership/planning contract with fake actuator endpoints.
The original case fails on deployed 1538fd1. Rejection cases retain ownership
without executing an obsolete pre-ownership plan or automatically releasing.
Recorded geometry/pose data are exercised separately by test_recorded_failures.
"""

import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from runtime import Runtime


class ColdOwnershipAcceptance(unittest.TestCase):
    def subject(self):
        subject = Runtime.__new__(Runtime)
        subject.arm = None
        subject.hand = None
        subject.plant = None
        subject.active = False
        subject.phase = "PASSIVE"
        subject.cancel = threading.Event()
        subject.lock = threading.RLock()
        subject.operation_lock = threading.Lock()
        subject.observation_lock = threading.RLock()
        subject.camera = Mock()
        subject.event = Mock()
        subject.read = Mock()
        subject.hub = Mock()
        subject.motion_commands_sent = 0
        subject.archive_guard = Mock(return_value=dict(age_s={}, session="offline"))
        subject.config = Mock(return_value=dict(arm_limits_rad=[]))
        subject.make_plant = Mock()
        subject.start_tracking = Mock()
        subject.stop_tracking = Mock()
        subject.stop_perception = Mock()
        subject.wait_for_control_settle = Mock()
        return subject

    def test_pickup_execution_uses_a_plan_measured_after_sdk_ownership(self):
        subject = self.subject()
        state = {"owned": False}
        planned = []
        executed = []

        def acquire(**kwargs):
            state["owned"] = True
            return Mock()

        def prepare(*args):
            result = (
                SimpleNamespace(
                    center=[0.38627 if not state["owned"] else 0.39629, 0.118, 0.823]
                ),
                {},
            )
            planned.append(state["owned"])
            return result

        def execute(plant, prepared, *args):
            executed.append(prepared[0].center[0])
            return dict(lift_verified=False, visual_rise_m=0, hold_verified_s=0)

        modules = {
            "g1_arm": SimpleNamespace(G1_29_ArmController=acquire),
            "g1_dex3": SimpleNamespace(Dex3Sender=Mock()),
        }
        with (
            patch.dict(sys.modules, modules),
            patch("perception_worker.PerceptionWorker"),
            patch("runtime.snapshot_guard"),
            patch("runtime.calibration_start_guard"),
            patch("runtime.http_json", side_effect=ConnectionError),
            patch("pickup.prepare", side_effect=prepare),
            patch("pickup.execute", side_effect=execute),
        ):
            subject.run(True)
        self.assertTrue(
            planned[-1],
            "Final plan was measured before cold SDK ownership changed robot pose",
        )
        self.assertAlmostEqual(executed[-1], 0.39629, places=5)
        subject.start_tracking.assert_not_called()
        self.assertIsNone(subject.make_plant.return_value.perception_guard)

    def failed_owned_attempt(self, failure):
        subject = self.subject()
        state = {"owned": False}

        def acquire(**kwargs):
            state["owned"] = True
            return Mock()

        def prepare(*args):
            if state["owned"] and failure == "perception":
                raise RuntimeError("Owned perception failed")
            return (SimpleNamespace(center=[0.38627, 0.118, 0.823]), {})

        if failure == "cancel":
            subject.wait_for_control_settle.side_effect = subject.cancel.set
        if failure == "settle":
            subject.wait_for_control_settle.side_effect = RuntimeError(
                "Control handover did not settle within 5 seconds"
            )
        modules = {
            "g1_arm": SimpleNamespace(G1_29_ArmController=acquire),
            "g1_dex3": SimpleNamespace(Dex3Sender=Mock()),
        }
        with (
            patch.dict(sys.modules, modules),
            patch("perception_worker.PerceptionWorker"),
            patch("runtime.snapshot_guard"),
            patch("runtime.calibration_start_guard"),
            patch("runtime.http_json", side_effect=ConnectionError),
            patch("pickup.prepare", side_effect=prepare) as planning,
            patch("pickup.execute") as execute,
        ):
            with self.assertRaises(RuntimeError):
                subject.run(True)
            execute.assert_not_called()
            if failure != "perception":
                self.assertEqual(planning.call_count, 1)
        self.assertEqual(subject.phase, "FROZEN")
        subject.plant.freeze.assert_called_once()
        subject.arm.shutdown.assert_not_called()
        subject.hand.relax.assert_not_called()

    def test_failed_owned_perception_cannot_execute_the_preview_plan(self):
        self.failed_owned_attempt("perception")

    def test_cancel_after_settling_cannot_start_owned_replanning(self):
        self.failed_owned_attempt("cancel")

    def test_unstable_ownership_holds_without_executing_or_releasing(self):
        self.failed_owned_attempt("settle")


if __name__ == "__main__":
    unittest.main()
