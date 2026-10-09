import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from g1_can_handoff.sim import scene
from physical_core import BODY_NAMES, PhysicalPlant, ARM_GRAVITY_LIMITS_NM


class ArmGravityTests(unittest.TestCase):
    def plant(self, enabled=True):
        model = scene()
        ids = np.array([model.model.joint(n).id for n in BODY_NAMES])
        q = model.data.qpos[model.model.jnt_qposadr[ids]].tolist()
        clock = SimpleNamespace(now=10.)
        clock.monotonic = lambda: clock.now
        def read():
            return {name:dict(q=q if name=='body' else [0]*n, dq=[0]*n, tau_est=[0]*n,
                              faults=[0]*n, temperature_max=30., temperatures=[[30.,30.]]*n,
                              imu_rpy=[0]*3, time=clock.now)
                    for name,n in [('body',29),('left',7),('right',7)]}
        adapter = Mock()
        config = dict(arm_limits_rad=model.model.jnt_range[ids[15:]].tolist(),
                      arm_gravity_compensation=enabled)
        p = PhysicalPlant(model,adapter,None,read,config,lambda *a,**k:None,clock=clock)
        return p, adapter, read, clock

    def test_feedforward_is_nonzero_capped_and_slew_limited(self):
        p, adapter, read, clock = self.plant()
        previous = np.zeros(14)
        for _ in range(80):
            clock.now += .02
            p.command_arm(np.asarray(read()['body']['q'][15:22]), read())
            q, torque = adapter.ctrl_dual_arm.call_args.args
            self.assertEqual(q.shape, (14,))
            self.assertTrue(np.all(np.abs(torque) <= ARM_GRAVITY_LIMITS_NM))
            self.assertLessEqual(np.max(np.abs(torque-previous)), .10000001)
            previous = torque.copy()
        self.assertGreater(np.max(np.abs(previous)), .5)
        clock.now += 100
        p.command_arm(np.asarray(read()['body']['q'][15:22]), read())
        self.assertLessEqual(np.max(np.abs(adapter.ctrl_dual_arm.call_args.args[1]-previous)), .10000001)

    def test_disabled_compensation_sends_zero_torque(self):
        p, adapter, read, clock = self.plant(False)
        clock.now += .02
        p.command_arm(np.asarray(read()['body']['q'][15:22]), read())
        np.testing.assert_array_equal(adapter.ctrl_dual_arm.call_args.args[1], np.zeros(14))

    def test_hand_wait_retains_arm_target_and_refreshes_bounded_gravity(self):
        p, adapter, read, clock = self.plant()
        p.sender = Mock()
        target = np.asarray(read()['body']['q'][15:22]) + .01
        clock.now += .02
        p.command_arm(target, read())
        previous = p.arm_tau_command.copy()
        for _ in range(20):
            clock.now += .02
            p.maintain_hand()
            q, torque = adapter.ctrl_dual_arm.call_args.args
            np.testing.assert_array_equal(q[:7], target)
            self.assertTrue(np.all(np.abs(torque) <= ARM_GRAVITY_LIMITS_NM))
            self.assertLessEqual(np.max(np.abs(torque-previous)), .10000001)
            previous = torque.copy()
        self.assertGreater(np.max(np.abs(previous)), .5)

    def test_arm_tracking_fault_during_hand_wait_sends_no_commands(self):
        p, adapter, read, clock = self.plant()
        p.sender = Mock()
        p.left_arm += .16
        with self.assertRaisesRegex(RuntimeError, 'tracking error'):
            p.maintain_hand()
        adapter.ctrl_dual_arm.assert_not_called()
        p.sender.ctrl_dual_hand.assert_not_called()


if __name__ == '__main__':
    unittest.main()
