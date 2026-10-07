"""Control endpoint cancellation and release ordering without DDS publishers."""
import os
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import json

# Runtime import creates only its event directory; no device connections.
os.environ.setdefault('HANDOFF_DATA',tempfile.mkdtemp(prefix='handoff-test-'))
from runtime import Runtime


class RuntimeLifecycleTests(unittest.TestCase):
    def runtime(self):
        runtime=Runtime.__new__(Runtime)
        runtime.cancel=threading.Event();runtime.operation_lock=threading.Lock()
        runtime.plant=Mock();runtime.arm=Mock();runtime.hand=Mock()
        runtime.status=Mock(return_value={});runtime.event=Mock()
        runtime.phase='FROZEN';runtime.active=False;runtime.grasp_started=False
        return runtime

    def test_release_cancels_but_never_relaxes_while_trial_running(self):
        runtime=self.runtime();runtime.operation_lock.acquire()
        with self.assertRaisesRegex(RuntimeError,'cancellation is pending'):runtime.release(True)
        self.assertTrue(runtime.cancel.is_set())
        runtime.plant.freeze.assert_called_once()
        runtime.hand.relax.assert_not_called()
        runtime.arm.shutdown.assert_not_called()

    def test_release_requires_supported_can_and_finishes_after_trial(self):
        runtime=self.runtime();hand=runtime.hand;arm=runtime.arm
        with self.assertRaisesRegex(RuntimeError,'support'):runtime.release(False)
        self.assertFalse(runtime.cancel.is_set())
        runtime.release(True)
        hand.relax.assert_called_once();arm.shutdown.assert_called_once_with(ramp_s=1.)
        self.assertEqual(runtime.phase,'PASSIVE')
        self.assertFalse(runtime.operation_lock.locked())

    def test_failure_holds_measured_pose_and_retains_control(self):
        runtime=self.runtime();arm=runtime.arm;hand=runtime.hand;plant=runtime.plant
        runtime.camera=Mock();runtime.archive_guard=Mock();runtime.read=Mock()
        runtime.config=Mock(return_value={});runtime.make_plant=Mock(return_value=plant)
        with patch('runtime.snapshot_guard'), patch('pickup.prepare',side_effect=RuntimeError('Target lost')):
            with self.assertRaisesRegex(RuntimeError,'Target lost'):runtime.run(True)
        plant.freeze.assert_called_once()
        arm.shutdown.assert_not_called();hand.relax.assert_not_called()
        self.assertIs(runtime.arm,arm)
        self.assertEqual(runtime.phase,'FROZEN')
        runtime.camera.close.assert_called_once()

    def test_plan_refresh_is_saved_only_after_success_without_adapter(self):
        for failure in (False,True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                runtime=self.runtime();runtime.arm=None;runtime.hand=None;runtime.plant=None
                runtime.camera=Mock();runtime.archive_guard=Mock();runtime.read=Mock()
                runtime.config=Mock(return_value={'scene_refresh_automatic':True})
                runtime.make_plant=Mock();runtime.motion_commands_sent=0
                path=Path(directory)/'calibration.json';path.write_text('{"old":true}')
                with patch('runtime.DATA',Path(directory)), patch('runtime.snapshot_guard'), patch('runtime.calibration_start_guard'), patch('pickup.prepare') as prepare:
                    if failure:
                        prepare.side_effect=RuntimeError('No collision-free path')
                        with self.assertRaisesRegex(RuntimeError,'collision-free'):runtime.run(False)
                        self.assertEqual(json.loads(path.read_text()),{'old':True})
                    else:
                        result=runtime.run(False)
                        self.assertTrue(result['planned'])
                        self.assertTrue(json.loads(path.read_text())['scene_refresh_automatic'])
                runtime.config.assert_called_once_with(refresh=True)
                runtime.camera.close.assert_called_once()
                self.assertIsNone(runtime.arm)

    def test_retry_reuses_held_controller_and_never_releases(self):
        runtime=self.runtime();arm=runtime.arm;hand=runtime.hand
        runtime.camera=Mock();runtime.read=Mock()
        runtime.archive_guard=Mock(return_value={'age_s':{'rt/arm_sdk':0},'session':'test'})
        runtime.config=Mock(return_value={'arm_limits_rad':[]})
        runtime.make_plant=Mock();plant=runtime.make_plant.return_value
        result={'lift_verified':True,'visual_rise_m':.13,'hold_verified_s':1}
        with patch('runtime.snapshot_guard'),patch('runtime.calibration_start_guard'),patch('runtime.http_json',side_effect=ConnectionError),patch('pickup.prepare'),patch('pickup.execute',return_value=result):
            self.assertTrue(runtime.run(True)['physical_lift_verified'])
        self.assertIs(runtime.arm,arm);self.assertIs(plant.adapter,arm)
        arm.shutdown.assert_not_called();hand.relax.assert_not_called()
        runtime.event.assert_any_call('ownership_reused',recorder_session='test')

    def test_retry_refuses_possible_held_can_or_active_motion(self):
        for active,grasp in ((False,True),(True,False)):
            runtime=self.runtime();runtime.active=active;runtime.grasp_started=grasp
            runtime.camera=Mock()
            with self.assertRaisesRegex(RuntimeError,'pregrasp failure hold'):runtime.run(True)
            runtime.camera.acquire.assert_not_called()
            runtime.arm.shutdown.assert_not_called()

    def test_camera_limit_tuning_preserves_hold_and_persists_without_resume(self):
        with tempfile.TemporaryDirectory() as directory,patch('runtime.DATA',Path(directory)),patch.dict('os.environ',{}):
            runtime=self.runtime();arm=runtime.arm;hand=runtime.hand
            runtime.configure_perception(400)
            self.assertEqual(json.loads((Path(directory)/'perception-config.json').read_text())['camera_max_age_ms'],400)
            self.assertEqual(os.environ['HANDOFF_CAMERA_MAX_AGE_MS'],'400.0')
            self.assertEqual(runtime.phase,'FROZEN')
            arm.shutdown.assert_not_called();hand.relax.assert_not_called()
            runtime.event.assert_called_once_with('perception_configured',camera_max_age_ms=400,automatic_resume=False)
    def test_active_or_invalid_camera_limit_does_not_mutate_runtime(self):
        with tempfile.TemporaryDirectory() as directory,patch('runtime.DATA',Path(directory)):
            runtime=self.runtime()
            for value in [None,True,401,float('nan')]:
                with self.subTest(value=value),self.assertRaises(ValueError):runtime.configure_perception(value)
            runtime.active=True
            with self.assertRaisesRegex(RuntimeError,'active trial'):runtime.configure_perception(400)
            self.assertFalse((Path(directory)/'perception-config.json').exists())
            runtime.event.assert_not_called()

    def test_demo_start_is_single_job_and_stop_remains_latched_before_worker(self):
        runtime=self.runtime();runtime.run=Mock(return_value={'physical_lift_verified':False})
        with patch('runtime.threading.Thread') as thread:
            runtime.start_demo()
            self.assertTrue(runtime.operation_lock.locked())
            with self.assertRaisesRegex(RuntimeError,'already running'):runtime.start_demo()
            runtime.stop()
            self.assertTrue(runtime.cancel.is_set())
            thread.call_args.kwargs['target']()
        runtime.run.assert_called_once_with(True,preserve_cancel=True)
        self.assertTrue(runtime.cancel.is_set())
        self.assertFalse(runtime.demo_running)
        self.assertFalse(runtime.operation_lock.locked())
    def test_cancelled_async_demo_cannot_reacquire_camera_or_arm(self):
        runtime=self.runtime();runtime.camera=Mock();runtime.cancel.set()
        with self.assertRaisesRegex(RuntimeError,'cancelled before demo preparation'):
            runtime.run(True,preserve_cancel=True)
        runtime.camera.acquire.assert_not_called()
        runtime.arm.shutdown.assert_not_called()

if __name__=='__main__':unittest.main()
