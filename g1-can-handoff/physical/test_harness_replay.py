"""No hardware publishers: validate saved replay and its live feedback gates."""
import copy
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from g1_can_handoff.sim import scene
from harness_replay import validate_fixture, run, preflight, command


class HarnessReplayTests(unittest.TestCase):
    def fixture(self):
        return dict(period_s=.02,left_q=np.zeros((3,7)).tolist(),left_tau=np.zeros((3,7)).tolist())

    def test_invalid_saved_actions_never_pass_validation(self):
        limits=[[-1,1]]*14
        validate_fixture(self.fixture(),limits)
        for field,value in [('period_s',.04),('left_q',[[0]*7]),('left_tau',[[100]*7]*3)]:
            bad=self.fixture();bad[field]=value
            with self.assertRaises(ValueError):validate_fixture(bad,limits)
        for value in (float('nan'),float('inf'),1.1,.021):
            bad=self.fixture();bad['left_q'][1][0]=value
            with self.assertRaises(ValueError):validate_fixture(bad,limits)
        with self.assertRaises(ValueError):validate_fixture(self.fixture(),[[0,0]]*14)

    def test_harness_alone_is_not_evidence_of_an_empty_workspace(self):
        runtime=Mock()
        for request in ({},{'harness_confirmed':True},{'workspace_clear_confirmed':True}):
            with self.assertRaisesRegex(ValueError,'clear arm-sweep'):run(runtime,request)
        runtime.read.assert_not_called();runtime.arm.ctrl_dual_arm.assert_not_called()

    def test_cancel_fault_and_collision_prevent_a_command(self):
        runtime=SimpleNamespace(arm=Mock(),event=Mock())
        for error in ('Operator cancelled','body motor fault','Live target intersects'):
            plant=Mock();plant.command_lock=threading.RLock()
            if error.startswith('Live'):
                plant.check_live_geometry.side_effect=RuntimeError(error)
            else:plant.check.side_effect=RuntimeError(error)
            with self.assertRaisesRegex(RuntimeError,error):command(runtime,plant,np.zeros(7),np.zeros(7))
            runtime.arm.ctrl_dual_arm.assert_not_called()

    def test_preflight_checks_between_recorded_targets_and_restores_fk(self):
        world=scene();plant=Mock();plant.ik=world
        plant.config={'arm_limits_rad':[[-3,3]]*14}
        before=world.data.qpos.copy()
        targets=np.zeros((2,7));targets[1][0]=.016
        with patch('harness_replay.joint_path',return_value=([],{})):
            preflight(plant,targets)
        self.assertGreaterEqual(plant.check_live_geometry.call_count,7)
        checked=[call.kwargs['left_q'][0] for call in plant.check_live_geometry.call_args_list]
        self.assertIn(.008,checked)
        np.testing.assert_array_equal(world.data.qpos,before)
        plant.adapter.ctrl_dual_arm.assert_not_called()


if __name__=='__main__':unittest.main()
