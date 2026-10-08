import os
import unittest
from unittest.mock import patch
import numpy as np
from adapter.arm_gains import arm_gains
from sim_runtime import SimulationIO
from g1_can_handoff.sim import scene


class ArmGainTests(unittest.TestCase):
    def test_selected_physical_and_simulator_defaults_match(self):
        with patch.dict(os.environ,{},clear=True):
            kp,kd=arm_gains()
            np.testing.assert_array_equal(kp,[150]*4+[40]*3)
            world=scene();io=SimulationIO(world)
            np.testing.assert_array_equal(io.arm_kp,np.tile(kp,2))
            np.testing.assert_array_equal(io.arm_kd,np.tile(kd,2))

    def test_gain_override_changes_only_shoulders_and_elbows(self):
        with patch.dict(os.environ,{'HANDOFF_ARM_KP':'100'}):
            kp,kd=arm_gains()
        np.testing.assert_array_equal(kp,[100]*4+[40]*3)
        np.testing.assert_array_equal(kd,[3]*4+[1.5]*3)
        world=scene();io=SimulationIO(world,arm_kp=80.)
        np.testing.assert_array_equal(io.arm_kp,[80]*4+[40]*3+[80]*4+[40]*3)
        q=np.asarray(io.read()['body']['q'][15:29]);io.ctrl_dual_arm(q+.01,np.zeros(14));io.sleep(.01)
        self.assertTrue(np.all(np.isfinite(world.data.qpos)))

    def test_invalid_gains_fail_before_a_writer_can_be_created(self):
        for value in ('0','-1','151','nan','inf','invalid'):
            with patch.dict(os.environ,{'HANDOFF_ARM_KP':value}):
                with self.assertRaises(ValueError):arm_gains()


if __name__=='__main__':unittest.main()
