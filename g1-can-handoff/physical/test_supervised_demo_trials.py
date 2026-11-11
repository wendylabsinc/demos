import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from supervised_demo_trials import preflight, reserve, run


class SupervisedTrialTests(unittest.TestCase):
    def test_budget_must_match_verified_controller_and_yolo_release(self):
        before=dict(status=dict(phase='PASSIVE',adapter_initialized=False,demo_running=False,
            telemetry_healthy=True,calibration_present=True,camera_max_age_ms=500,yolo_confidence_strictly_above=.1),
            recorder=dict(recording=True,error=None,dropped={},missing_or_stale_required=[]),
            camera=dict(healthy=True,serial='338622073335'),camera_exposure_age_ms=490,
            yolo=dict(healthy=True,camera_max_age_ms=500,
                model_sha256='fab6252b0abc0613ef39c36ea2d75a061d80bb087f5eb9034ca33614c26301e9'))
        preflight(before,500)
        before['yolo']['camera_max_age_ms']=400
        with self.assertRaisesRegex(RuntimeError,'budgets differ'):preflight(before,500)
        before['yolo']['camera_max_age_ms']=500;before['camera_exposure_age_ms']=500.1
        with self.assertRaisesRegex(RuntimeError,'Camera exposure is stale'):preflight(before,500)
        with self.assertRaisesRegex(RuntimeError,'Unexpected configured'):preflight(before,400)

    def test_color_depth_mode_does_not_require_yolo_but_still_requires_fresh_camera(self):
        before=dict(status=dict(phase='PASSIVE',adapter_initialized=False,demo_running=False,
            telemetry_healthy=True,calibration_present=True,camera_max_age_ms=500,
            yolo_confidence_strictly_above=.1,target_identity='color_depth'),
            recorder=dict(recording=True,error=None,dropped={},missing_or_stale_required=[]),
            camera=dict(healthy=True,serial='338622073335'),camera_exposure_age_ms=100,
            yolo={'required':False})
        preflight(before,500)
        before['camera_exposure_age_ms']=501
        with self.assertRaisesRegex(RuntimeError,'Camera exposure is stale'):preflight(before,500)

    def test_no_automatic_second_start_without_operator_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'trial-01.json').write_text(json.dumps(dict(demo_running_after=False)))
            with self.assertRaisesRegex(RuntimeError,'reset confirmation'):
                reserve(root,2,False)
            self.assertFalse((root/'trial-02.json').exists())
            self.assertTrue(reserve(root,2,True).exists())

    def test_explicit_pregrasp_resume_preserves_camera_and_grasp_guards(self):
        before=dict(status=dict(phase='FROZEN',adapter_initialized=True,demo_running=False,
            telemetry_healthy=True,calibration_present=True,camera_max_age_ms=500,
            yolo_confidence_strictly_above=.1,target_identity='color_depth'),
            recorder=dict(recording=True,error=None,dropped={},missing_or_stale_required=[]),
            camera=dict(healthy=True,serial='338622073335'),camera_exposure_age_ms=100,
            yolo={'required':False},failure_hold=dict(grasp_started=False))
        preflight(before,resume_pregrasp=True)
        with self.assertRaisesRegex(RuntimeError,'Reset/release'):preflight(before)
        before['camera_exposure_age_ms']=501
        with self.assertRaisesRegex(RuntimeError,'Camera exposure is stale'):preflight(before,resume_pregrasp=True)
        before['camera_exposure_age_ms']=100
        for hold in (None,{},dict(grasp_started=True)):
            before['failure_hold']=hold
            with self.assertRaisesRegex(RuntimeError,'Reset/release'):preflight(before,resume_pregrasp=True)
        before['failure_hold']=dict(grasp_started=False);before['status']['demo_running']=True
        with self.assertRaisesRegex(RuntimeError,'already running'):preflight(before,resume_pregrasp=True)

    def test_reset_acknowledgement_cannot_skip_a_running_previous_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'trial-01.json').write_text(json.dumps(dict(demo_running_after=True)))
            with self.assertRaisesRegex(RuntimeError,'terminal trial'):
                reserve(root,2,True)

    def test_duplicate_run_never_constructs_transport_or_reissues_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);reserve(root,1,False)
            with patch('supervised_demo_trials.DEPLOYING') as marker, patch('supervised_demo_trials.Wendy') as client:
                marker.is_file.return_value=False
                with self.assertRaisesRegex(RuntimeError,'already reserved'):
                    run(root,1,False)
                client.assert_not_called()

    def test_retained_ownership_requires_release_even_if_reset_was_acknowledged(self):
        for phase,owned,running in [('FROZEN',True,False),('PASSIVE',True,False),('PASSIVE',False,True)]:
            with self.subTest(phase=phase,owned=owned,running=running):
                before=dict(status=dict(phase=phase,adapter_initialized=owned,demo_running=running),recorder={})
                with self.assertRaisesRegex(RuntimeError,'Reset/release required'):
                    preflight(before)


if __name__=='__main__':unittest.main()
