"""Contact-stop grasp tests, including two chronological hardware failures."""
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock
import numpy as np
from gentle_grasp import GentleGrasp
from physical_core import PhysicalPlant, sim_hand_to_bridge
from g1_can_handoff.task import CLOSED


class GentleGraspTests(unittest.TestCase):
    def test_two_real_failures_switch_before_heating_and_cap_proportional_effort(self):
        path = Path(__file__).parent / 'offline/validation/grip-transition-signals-20261009/trace.json.gz'
        for case in json.loads(gzip.decompress(path.read_bytes()))['cases']:
            with self.subTest(case=case['name']):
                rows = [r for r in case['rows'] if 0 <= r['t'] < case['stop_s']]
                grasp = GentleGrasp(rows[0]['q'], sim_hand_to_bridge(CLOSED), kp=2)
                trigger = None
                first_thumb_hold = None
                for row in rows:
                    if not row['cmd']:
                        continue
                    target = grasp.update(row['t'], row['q'], row['cmd']['q'])
                    for joint in grasp.holding_joints:
                        self.assertLessEqual(abs(2 * (target[joint] - row['q'][joint])), 0.500001)
                    if 2 in grasp.holding_joints and first_thumb_hold is None:
                        first_thumb_hold = row
                    if grasp.grasped:
                        trigger = row
                        break
                self.assertIsNotNone(trigger)
                self.assertLess(trigger['t'], case['stop_s'] - 2)
                self.assertIsNotNone(first_thumb_hold)
                self.assertLess(first_thumb_hold['temperatures'][2][1], 50)

    def test_motion_pauses_shorter_than_point_two_do_not_grasp(self):
        g = GentleGrasp([0] * 7, sim_hand_to_bridge(CLOSED), kp=2)
        for t in [0, 0.1, 0.19]:
            g.update(t, [0, .1, .3, -.4, -.4, -.4, -.4], CLOSED)
        self.assertFalse(g.grasped)

    def test_idle_or_empty_fully_closed_hand_is_not_object_contact(self):
        for measured in ([0] * 7, sim_hand_to_bridge(CLOSED)):
            g = GentleGrasp(measured, sim_hand_to_bridge(CLOSED), kp=2)
            for t in np.arange(0, .8, .02):
                g.update(t, measured, sim_hand_to_bridge(CLOSED))
            self.assertFalse(g.grasped)

    def test_repeated_feedback_cannot_count_as_point_two_seconds(self):
        g = GentleGrasp([0] * 7, sim_hand_to_bridge(CLOSED), kp=2)
        for _ in range(100):
            g.update(1.0, [0, .1, .3, -.4, -.4, -.4, -.4], CLOSED)
        self.assertFalse(g.grasped)

    def test_slow_but_still_moving_fingers_are_not_grasped(self):
        g = GentleGrasp([0] * 7, sim_hand_to_bridge(CLOSED), kp=2)
        for t in np.arange(0, 1.0, .02):
            q = np.array([0, t*.04, t*.04, -t*.04, -t*.04, -t*.04, -t*.04])
            g.update(float(t), q, sim_hand_to_bridge(CLOSED))
        self.assertFalse(g.grasped)

    def test_simulated_finger_gains_match_physical_holding_effort(self):
        from g1_can_handoff.sim import scene
        from sim_runtime import SimulationIO
        world = scene(); io = SimulationIO(world)
        self.assertEqual(io.kp, 2)
        np.testing.assert_array_equal(world.kp[io.all_hand_ids], 2)
        np.testing.assert_array_equal(world.kd[io.all_hand_ids], .2)

    def test_hold_does_not_chase_fingers_farther_closed(self):
        g = GentleGrasp([0] * 7, sim_hand_to_bridge(CLOSED), kp=2)
        q = np.array([0, .1, .3, -.4, -.4, -.4, -.4])
        for t in np.arange(0, .5, .02):
            target = g.update(float(t), q, sim_hand_to_bridge(CLOSED))
        self.assertTrue(g.grasped)
        fixed = target.copy()
        inward = q + np.array([0, .1, .1, -.1, -.1, -.1, -.1])
        np.testing.assert_array_equal(g.hold(inward), fixed)
        outward = q - np.array([0, .1, .1, -.1, -.1, -.1, -.1])
        reduced = g.hold(outward)
        self.assertTrue(np.all(np.abs(2 * (reduced - outward)) <= .500001))

    def test_hand_closure_returns_gentle_grasp_and_logs_offsets_without_raising(self):
        p = PhysicalPlant.__new__(PhysicalPlant)
        p.command_lock = threading.RLock()
        p.left_hand = np.zeros(7); p.right_hand = np.zeros(7)
        clock = SimpleNamespace(now=0.0)
        clock.monotonic = lambda: clock.now
        clock.sleep = lambda dt: setattr(clock, 'now', clock.now + dt)
        p.clock = clock; p.grip = None
        limit = np.array([0, .13, .397, -.5, -.75, -.48, -.92])
        def read():
            q = np.where(limit >= 0, np.minimum(p.left_hand, limit), np.maximum(p.left_hand, limit))
            return {'left': {'q': q.tolist(), 'time': clock.now, 'dq': [0]*7, 'tau_est':[0]*7}}
        p.check = read; p.sync = Mock(); p.hold_arm = Mock(); p.check_live_geometry = Mock()
        p.sender = SimpleNamespace(kp=2, ctrl_dual_hand=Mock()); p.event = Mock()
        p.hand(CLOSED, 1.2)
        self.assertTrue(p.grip.grasped)
        self.assertLess(clock.now, 1.4)
        self.assertLessEqual(abs(2 * (p.left_hand[2] - .397)), .500001)
        groups = p.contacts()
        self.assertGreaterEqual(len(groups), 2)
        self.assertEqual(p.contact_evidence['kind'], 'gentle_contact_hold')
        # A finger offset while gripping is logged, not an arm-tracking exception.
        self.assertIn('finger_position_error_rad', p.contact_evidence)
        self.assertTrue(any(c.args[0] == 'gentle_grasp_started' for c in p.event.call_args_list))

    def test_feedback_failure_prevents_finger_commands(self):
        p = PhysicalPlant.__new__(PhysicalPlant)
        p.command_lock = threading.RLock(); p.left_hand = np.zeros(7)
        p.grip = None; p.check = Mock(side_effect=ValueError('Stale left measured feedback'))
        p.clock = SimpleNamespace(monotonic=lambda:0); p.sender=Mock(kp=2)
        with self.assertRaisesRegex(ValueError, 'Stale left'):
            p.hand(CLOSED)
        p.sender.ctrl_dual_hand.assert_not_called()

    def test_cancelled_planning_refresh_sends_no_finger_command(self):
        p = PhysicalPlant.__new__(PhysicalPlant)
        p.command_lock = threading.RLock(); p.cancelled = False
        p.cancel = threading.Event(); p.cancel.set(); p.sender = Mock()
        p.refresh_grip({})
        p.sender.ctrl_dual_hand.assert_not_called()

    def test_partial_contact_reduces_effort_without_claiming_remaining_fingers_stopped(self):
        g = GentleGrasp([0] * 7, sim_hand_to_bridge(CLOSED), kp=2)
        for t in np.arange(0, .7, .02):
            q = np.array([0, .1, .3, -.3, -.3, -t*.3, -t*.3])
            target = g.update(float(t), q, sim_hand_to_bridge(CLOSED))
        self.assertIn(2, g.holding_joints)
        self.assertFalse(g.grasped)
        self.assertLessEqual(abs(2 * (target[2] - q[2])), .500001)

if __name__ == '__main__':
    unittest.main()
