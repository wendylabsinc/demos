"""Rest endpoint owns no hand output and releases only after verified arrival."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
import test_runtime_lifecycle
import test_arm_gravity


class RestArmTests(unittest.TestCase):
    def fixture(self):
        runtime=test_runtime_lifecycle.RuntimeLifecycleTests().runtime()
        plant,_,read,_=test_arm_gravity.ArmGravityTests().plant()
        target=np.array([.25,.27,-.08,.82,0,0,0])
        runtime.camera=Mock();runtime.config=Mock(return_value=plant.config)
        runtime.archive_guard=Mock(return_value={})
        runtime.make_plant=Mock(return_value=plant);runtime.plant=None
        runtime.read=Mock(side_effect=read)
        runtime.relax_owned_arms=Mock()
        plant._joint_poses=Mock(return_value=(target,[target]))
        plant._execute_poses=Mock()
        return runtime,plant,target

    def patches(self,target):
        return (patch('staging.joint_path',return_value=([target.tolist()],{})),
                patch('runtime.geometry',return_value=(Mock(center=np.array([.35,.18,.81])),None)),
                patch('runtime.calibration_start_guard'))

    def test_plan_sends_no_commands_and_does_not_release(self):
        runtime,plant,target=self.fixture()
        with self.patches(target)[0],self.patches(target)[1],self.patches(target)[2]:
            result=runtime.rest_arm(False)
        self.assertTrue(result['planned'])
        plant._execute_poses.assert_not_called()
        runtime.arm.ctrl_dual_arm.assert_not_called()
        runtime.hand.relax.assert_not_called()
        runtime.relax_owned_arms.assert_not_called()
        runtime.camera.close.assert_called_once()

    def test_failed_collision_preflight_never_executes_or_releases(self):
        runtime,plant,target=self.fixture()
        plant._joint_poses.side_effect=RuntimeError('table collision')
        with self.patches(target)[0],self.patches(target)[1],self.patches(target)[2]:
            with self.assertRaisesRegex(RuntimeError,'table collision'):runtime.rest_arm(True)
        plant._execute_poses.assert_not_called()
        runtime.relax_owned_arms.assert_not_called()
        runtime.arm.hold_arm_measured.assert_called_once()

    def test_unverified_arrival_retains_control(self):
        runtime,plant,target=self.fixture()
        with self.patches(target)[0],self.patches(target)[1],self.patches(target)[2]:
            with self.assertRaisesRegex(RuntimeError,'Resting arm pose not reached'):runtime.rest_arm(True)
        plant._execute_poses.assert_called_once()
        runtime.relax_owned_arms.assert_not_called()
        runtime.hand.relax.assert_not_called()

    def test_verified_arrival_releases_without_hand_commands(self):
        runtime,plant,target=self.fixture()
        snapshot=plant.check()
        def execute(*args):snapshot['body']['q'][15:22]=target.tolist()
        runtime.read=Mock(return_value=snapshot)
        plant._execute_poses.side_effect=execute
        with self.patches(target)[0],self.patches(target)[1],self.patches(target)[2]:
            result=runtime.rest_arm(True)
        self.assertTrue(result['control_released'])
        runtime.relax_owned_arms.assert_called_once_with('verified_rest_pose')
        runtime.hand.ctrl_dual_hand.assert_not_called()
        runtime.hand.relax.assert_not_called()

    def test_possible_held_can_or_active_task_rejects_before_camera(self):
        for field in ('grasp_started','active'):
            runtime,_,_=self.fixture();setattr(runtime,field,True)
            with self.assertRaisesRegex(RuntimeError,'possibly holding'):runtime.rest_arm(True)
            runtime.camera.acquire.assert_not_called()

if __name__=='__main__':unittest.main()
