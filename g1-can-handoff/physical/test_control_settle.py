import threading
import unittest
from unittest.mock import Mock
from control_settle import wait_for_stable_control


class Clock:
    def __init__(self):self.now=100.
    def monotonic(self):return self.now
    def sleep(self,seconds):self.now+=seconds


class ControlSettleTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.cancel=threading.Event();self.event=Mock();self.archive=Mock()
    def read(self):
        return {name:dict(q=[0.]*count,dq=[0.]*count,tau_est=[0.]*count,
                         time=self.clock.now,temperature_max=25.,faults=[0]*count,imu_rpy=[0.]*3)
                for name,count in [('body',29),('left',7),('right',7)]}
    def wait(self,read=None):return wait_for_stable_control(read or self.read,self.cancel,self.event,self.archive,self.clock)

    def test_only_returns_after_half_second_of_advancing_stable_feedback(self):
        self.wait();self.assertGreaterEqual(self.clock.now-100.,.5)
        self.assertLess(self.clock.now-100.,.6);self.archive.assert_called()
        self.assertEqual(self.event.call_args.args,('control_settled',))

    def test_transition_discards_moving_history_before_stable_window(self):
        def moving():
            s=self.read();s['body']['q'][15]=min(self.clock.now-100.,.7)*.2
            s['body']['dq'][15]=.2 if self.clock.now<100.7 else 0.
            return s
        self.wait(moving);self.assertGreater(self.clock.now,101.2)

    def test_ongoing_motion_times_out_even_when_feedback_is_fresh(self):
        def moving():
            s=self.read();s['body']['dq'][15]=.04;return s
        with self.assertRaisesRegex(RuntimeError,'within 5 seconds'):self.wait(moving)
        self.assertLess(self.clock.now-100.,5.03);self.event.assert_not_called()

    def test_duplicate_samples_do_not_establish_a_stable_window(self):
        s=self.read()
        with self.assertRaisesRegex(ValueError,'Stale body'):self.wait(lambda:s)
        self.event.assert_not_called()

    def test_cancellation_and_unhealthy_motor_samples_abort_immediately(self):
        self.cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):self.wait()
        self.cancel.clear()
        def hot():
            s=self.read();s['left']['temperature_max']=75.;return s
        with self.assertRaisesRegex(ValueError,'temperature'):self.wait(hot)
        self.event.assert_not_called()
