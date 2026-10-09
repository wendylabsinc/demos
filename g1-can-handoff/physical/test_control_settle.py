import threading
import os
import unittest
from unittest.mock import Mock, patch
from control_settle import wait_for_stable_control


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class ControlSettleTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.cancel = threading.Event()
        self.event = Mock()
        self.archive = Mock()
        environment = patch.dict(
            os.environ, {"HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S": "1"}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def read(self):
        return {
            name: dict(
                q=[0.0] * count,
                dq=[0.0] * count,
                tau_est=[0.0] * count,
                time=self.clock.now,
                temperature_max=25.0,
                faults=[0] * count,
                imu_rpy=[0.0] * 3,
            )
            for name, count in [("body", 29), ("left", 7), ("right", 7)]
        }

    def wait(self, read=None):
        return wait_for_stable_control(
            read or self.read, self.cancel, self.event, self.archive, self.clock
        )

    def test_only_returns_after_half_second_of_advancing_stable_feedback(self):
        self.wait()
        self.assertGreaterEqual(self.clock.now - 100.0, 0.5)
        self.assertLess(self.clock.now - 100.0, 0.6)
        self.archive.assert_called()
        self.assertEqual(self.event.call_args.args, ("control_settled",))

    def test_transition_discards_moving_history_before_stable_window(self):
        def moving():
            s = self.read()
            s["body"]["q"][15] = min(self.clock.now - 100.0, 0.7) * 0.2
            s["body"]["dq"][15] = 0.2 if self.clock.now < 100.7 else 0.0
            return s

        self.wait(moving)
        self.assertGreater(self.clock.now, 101.1)
        self.assertLessEqual(
            max(self.event.call_args.kwargs["pose_span_rad"][18:]), 0.015
        )

    def test_ongoing_motion_times_out_even_when_feedback_is_fresh(self):
        def moving():
            s = self.read()
            s["body"]["q"][15] = (self.clock.now - 100.0) * 0.2
            s["body"]["dq"][15] = 0.2
            return s

        with self.assertRaisesRegex(RuntimeError, "within 5 seconds"):
            self.wait(moving)
        self.assertLess(self.clock.now - 100.0, 5.03)
        self.event.assert_not_called()

    def test_stable_positions_with_recorded_leg_velocity_noise_can_settle(self):
        def noisy():
            s = self.read()
            s["body"]["dq"][0] = 0.06
            return s

        self.wait(noisy)
        self.assertAlmostEqual(self.event.call_args.kwargs["max_speed_rad_s"], 0.06)
        self.assertEqual(self.event.call_args.kwargs["max_speed_limit_rad_s"], 1.0)

    def test_speed_limit_is_strict_and_applies_to_every_joint(self):
        for joint in range(29):
            for speed in (-1.0, 1.0, 1.1):
                with self.subTest(joint=joint, speed=speed):
                    self.clock = Clock()
                    self.event.reset_mock()

                    def fast():
                        s = self.read()
                        s["body"]["dq"][joint] = speed
                        return s

                    with self.assertRaisesRegex(RuntimeError, "within 5 seconds"):
                        self.wait(fast)
                    self.event.assert_not_called()

    def test_every_joint_below_one_can_settle_if_positions_are_stable(self):
        def noisy():
            s = self.read()
            s["body"]["dq"] = [0.999] * 29
            return s

        self.wait(noisy)

    def test_environment_can_lower_limit_but_invalid_values_fail_closed(self):
        with patch.dict(os.environ, {"HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S": ".03"}):

            def noisy():
                s = self.read()
                s["body"]["dq"][0] = 0.06
                return s

            with self.assertRaisesRegex(RuntimeError, "within 5 seconds"):
                self.wait(noisy)
        for value in ("nan", "inf", "0", "-1", "1.1", "garbage"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ, {"HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S": value}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "settling speed limit"):
                    self.wait()

    def test_duplicate_samples_do_not_establish_a_stable_window(self):
        s = self.read()
        with self.assertRaisesRegex(ValueError, "Stale body"):
            self.wait(lambda: s)
        self.event.assert_not_called()

    def test_cancellation_and_unhealthy_motor_samples_abort_immediately(self):
        self.cancel.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.wait()
        self.cancel.clear()

        def hot():
            s = self.read()
            s["left"]["temperature_max"] = 75.0
            return s

        with self.assertRaisesRegex(ValueError, "temperature"):
            self.wait(hot)
        self.event.assert_not_called()
