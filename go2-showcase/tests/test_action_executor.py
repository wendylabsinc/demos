import math
import unittest

from showcase.action_executor import ActionExecutor, MotionAction
from showcase.models import Observation


def observed(**kwargs):
    values = dict(fresh=True, scan_fresh=True, front_clearance=5,
                  sectors=tuple((i*math.pi/4, 5) for i in range(-4, 5)))
    values.update(kwargs)
    return Observation(**values)


class ExecutorTests(unittest.TestCase):
    def test_no_actions_without_enable(self):
        runner = ActionExecutor()
        self.assertEqual(runner.begin(MotionAction("forward"), observed(), 0).vx, 0)
        self.assertEqual(runner.state, "disabled")
        runner.enable(observed(), 0)
        self.assertEqual(runner.step(observed(), .1).vx, 0)
        self.assertEqual(runner.state, "waiting")

    def test_segments_expire_and_require_new_model_decision(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        self.assertEqual(runner.begin(MotionAction("forward", .3), observed(), .1).vx, .55)
        self.assertEqual(runner.step(observed(), .2).vx, .55)
        self.assertEqual(runner.step(observed(), .4).vx, 0)
        self.assertEqual(runner.step(observed(), .5).vx, 0)

    def test_stale_input_fault_latches(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        runner.begin(MotionAction("forward"), observed(), .1)
        self.assertEqual(runner.step(observed(fresh=False), .2).vx, 0)
        self.assertEqual(runner.state, "fault")
        self.assertEqual(runner.begin(MotionAction("forward"), observed(), .3).vx, 0)
        runner.enable(observed(), .4)
        self.assertEqual(runner.begin(MotionAction("forward"), observed(), .5).vx, .55)

    def test_invalid_decisions_fault(self):
        decisions = [MotionAction("forward", 2), MotionAction("backward"),
                     MotionAction("forward", float("nan")), MotionAction("forward", 0),
                     {"action": "forward", "duration": True}, {"action": []}]
        for decision in decisions:
            runner = ActionExecutor()
            runner.enable(observed(), 0)
            self.assertEqual(runner.begin(decision, observed(), .1).vx, 0)
            self.assertEqual(runner.state, "fault")

    def test_rotation_needs_full_scan(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        self.assertEqual(runner.begin(MotionAction("turn_left"), observed(), .1).yaw_rate, .45)
        self.assertEqual(runner.step(observed(sectors=((0, 5),)), .2).yaw_rate, 0)
        self.assertEqual(runner.state, "fault")

    def test_new_obstacle_stops_during_segment(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        runner.begin(MotionAction("forward", 1), observed(), .1)
        self.assertEqual(runner.step(observed(front_clearance=.8), .2).vx, 0)
        self.assertFalse(runner.enabled)

    def test_geofence_and_time_limits(self):
        runner = ActionExecutor(geofence_m=2)
        runner.enable(observed(), 0)
        self.assertEqual(runner.begin(MotionAction("forward"), observed(x=1), .1).vx, 0)
        self.assertEqual(runner.state, "fault")
        runner = ActionExecutor(budget_s=1)
        runner.enable(observed(), 0)
        runner.step(observed(), .6)
        self.assertEqual(runner.begin(MotionAction("forward"), observed(), 1).vx, 0)
        self.assertEqual(runner.state, "fault")

    def test_model_completion_is_explicitly_unverified(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        command = runner.begin({"action": "complete", "duration": 0,
                                "reason": "I see the keys", "observations": "keys near chair"}, observed(), .1)
        self.assertEqual(command.vx, 0)
        self.assertFalse(runner.enabled)
        self.assertEqual(runner.state, "complete")
        self.assertTrue(runner.model_completion["reported"])
        self.assertFalse(runner.model_completion["verified"])
        self.assertEqual(runner.model_completion["reason"], "I see the keys")

    def test_hold_and_operator_stop(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        self.assertEqual(runner.begin(MotionAction("hold", .5), observed(), .1).yaw_rate, 0)
        runner.stop()
        self.assertEqual(runner.step(observed(), .2).vx, 0)
        self.assertFalse(runner.enabled)

    def test_clock_gap_and_travel_budget(self):
        runner = ActionExecutor()
        runner.enable(observed(), 0)
        runner.step(observed(), 1.1)
        self.assertEqual(runner.state, "fault")
        runner = ActionExecutor(travel_budget_m=1)
        runner.enable(observed(), 0)
        runner.step(observed(x=.7), .1)
        runner.step(observed(), .2)
        self.assertEqual(runner.state, "fault")


if __name__ == "__main__":
    unittest.main()
