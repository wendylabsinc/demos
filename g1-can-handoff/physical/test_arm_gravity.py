import unittest
import importlib.util
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from g1_can_handoff.sim import scene
from physical_core import BODY_NAMES, PhysicalPlant, ARM_GRAVITY_LIMITS_NM

if os.getenv('ARM_GRAVITY_RUNTIME_SOURCE'):
    spec=importlib.util.spec_from_file_location('historical_physical_core',os.environ['ARM_GRAVITY_RUNTIME_SOURCE'])
    historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
    PhysicalPlant=historical.PhysicalPlant


class ArmGravityTests(unittest.TestCase):
    def plant(self, enabled=True):
        model = scene()
        ids = np.array([model.model.joint(n).id for n in BODY_NAMES])
        q = model.data.qpos[model.model.jnt_qposadr[ids]].tolist()
        clock = SimpleNamespace(now=10.)
        clock.monotonic = lambda: clock.now
        def read():
            return {name:dict(q=q if name=='body' else [0]*n, dq=[0]*n, tau_est=[0]*n,
                              faults=[0]*n, temperature_max=30., imu_rpy=[0]*3, time=clock.now)
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

    def test_hand_open_and_hold_refresh_the_same_arm_reference_and_feedforward(self):
        p,adapter,read,clock=self.plant();p.sender=SimpleNamespace(ctrl_dual_hand=Mock())
        clock.sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)
        reference=np.asarray(read()['body']['q'][15:22]).copy()
        p.hand(np.zeros(7),.8)
        self.assertGreater(adapter.ctrl_dual_arm.call_count,30)
        q,torque=adapter.ctrl_dual_arm.call_args.args
        np.testing.assert_array_equal(q[:7],reference)
        self.assertGreater(np.max(np.abs(torque)),.5)
        count=adapter.ctrl_dual_arm.call_count;p.step(.1)
        self.assertGreater(adapter.ctrl_dual_arm.call_count,count)
        np.testing.assert_array_equal(adapter.ctrl_dual_arm.call_args.args[0][:7],reference)

    def test_recalibrated_model_inherits_torque_and_reference_without_another_ramp(self):
        p,adapter,read,clock=self.plant();p.sender=SimpleNamespace(ctrl_dual_hand=Mock())
        clock.sleep=lambda seconds:setattr(clock,'now',clock.now+seconds)
        p.hand(np.zeros(7),.8)
        new=PhysicalPlant(scene(),adapter,p.sender,read,p.config,lambda *a,**k:None,clock=clock)
        new.inherit_control_hold(p)
        np.testing.assert_array_equal(new.arm_tau_command,p.arm_tau_command)
        new.hold_arm(read())
        self.assertLessEqual(np.max(np.abs(new.arm_tau_command-p.arm_tau_command)),.10000001)
        np.testing.assert_array_equal(new.left_arm,p.left_arm)
        self.assertGreater(np.max(np.abs(new.arm_tau_command)),.5)

    def test_different_or_stopped_controller_cannot_supply_a_hold(self):
        p,_,_,_=self.plant();other,adapter,_,_=self.plant()
        with self.assertRaisesRegex(RuntimeError,'different or stopped'):other.inherit_control_hold(p)
        other.adapter=p.adapter;other.sender=p.sender;other.clock=p.clock;p.cancelled=True
        with self.assertRaisesRegex(RuntimeError,'different or stopped'):other.inherit_control_hold(p)
        adapter.ctrl_dual_arm.assert_not_called()

    def test_cancelled_planning_never_enters_ik_search(self):
        p,adapter,_,_=self.plant();p.cancel=threading.Event();p.cancel.set();p.ik.solve=Mock()
        with self.assertRaisesRegex(RuntimeError,'cancelled'):p.solve([.35,.18,.82])
        p.ik.solve.assert_not_called();adapter.ctrl_dual_arm.assert_not_called()


if __name__ == '__main__':
    unittest.main()
