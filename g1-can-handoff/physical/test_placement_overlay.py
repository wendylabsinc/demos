"""Placement projection and visibility checks; no network or robot required."""
import unittest
import json

import numpy as np
from placement_overlay import project_area, project_status, PlacementGuide
from unittest.mock import patch, Mock


class PickupAreaTests(unittest.TestCase):
    def snapshot(self):
        return dict(demo_running=False,phase='PASSIVE',body_age_s=.01,
            body=dict(time=100.,imu_rpy=[0,0,0],q=[0]*29),
            bounds={'0':[.27,.45],'1':[.09,.29]},camera_serial='test',
            camera_health=dict(healthy=True,frame_age_s=.03),
            intrinsics=dict(fx=100.,fy=100.,ppx=0.,ppy=0.,width=640,height=480,coeffs=[0]*5),
            config=dict(calibration_method='operator_accepted_factory_table',operator_accepted=True,
                stationary_base_confirmed=True,camera_serial='test',reference_imu_rpy=[0,0,0],
                reference_base_q=[0]*15,table_height_m=1.,
                camera_to_world=np.eye(4).tolist(),pelvis_to_world=np.eye(4).tolist()))

    def test_projects_all_four_workspace_corners_on_table(self):
        result=project_area(self.snapshot())
        np.testing.assert_allclose(result['polygon'],[[27,9],[45,9],[45,29],[27,29]])
        self.assertEqual(result['bounds_m'],{'0':[.27,.45],'1':[.09,.29]})

    def test_uses_camera_translation_and_current_attitude(self):
        snapshot=self.snapshot();snapshot['config']['camera_to_world'][0][3]=.1
        np.testing.assert_allclose(project_area(snapshot)['polygon'],[[17,9],[35,9],[35,29],[17,29]])
        before=project_area(snapshot)['polygon'];snapshot['body']['imu_rpy'][2]=.01
        self.assertFalse(np.allclose(project_area(snapshot)['polygon'],before))

    def test_rejects_stale_mismatched_and_unsupported_current_projection(self):
        changes=[{'body_age_s':.201},{'camera_serial':'other'},
                 {'camera_health':dict(healthy=False,frame_age_s=.03)}]
        for fields in changes:
            snapshot=self.snapshot();snapshot.update(fields)
            with self.subTest(fields=fields),self.assertRaises(ValueError):project_area(snapshot)
        snapshot=self.snapshot();snapshot['intrinsics']['coeffs'][0]=.01
        with self.assertRaisesRegex(ValueError,'distortion'):project_area(snapshot)
        snapshot=self.snapshot();snapshot['config']['operator_accepted']=False
        with self.assertRaisesRegex(ValueError,'accepted'):project_area(snapshot)

    def test_running_demo_keeps_current_projection(self):
        snapshot=self.snapshot();snapshot['demo_running']=True
        self.assertTrue(project_status(snapshot)['valid'])
        self.assertEqual(len(project_status(snapshot)['polygon']),4)

    def test_waist_mismatch_keeps_explicit_reference_even_on_first_load(self):
        snapshot=self.snapshot();snapshot['body']['q'][12]=.01
        result=project_status(snapshot)
        self.assertFalse(result['valid']);self.assertTrue(result['reference_only'])
        self.assertIn('Waist',result['error'])
        np.testing.assert_allclose(result['polygon'],[[27,9],[45,9],[45,29],[27,29]])

    def test_stale_feedback_retains_reference_but_never_current_green_status(self):
        snapshot=self.snapshot();snapshot['body_age_s']=.8
        result=project_status(snapshot)
        self.assertFalse(result['valid']);self.assertTrue(result['reference_only'])
        self.assertIn('stale',result['error'])

    def test_reference_still_rejects_wrong_camera_and_distortion(self):
        for field in ('camera','distortion'):
            snapshot=self.snapshot()
            if field=='camera':snapshot['camera_serial']='other'
            else:snapshot['intrinsics']['coeffs'][0]=.1
            with self.subTest(field=field),self.assertRaises(ValueError):project_status(snapshot)

    def test_expired_cache_retains_polygon_as_reference(self):
        guide=PlacementGuide(lambda:None)
        guide.value=project_status(self.snapshot());guide.updated=10
        with patch('placement_overlay.time.monotonic',return_value=14):result=guide.read()
        self.assertFalse(result['valid']);self.assertTrue(result['reference_only'])
        self.assertEqual(len(result['polygon']),4)
        self.assertIn('expired',result['error'])

    def test_transport_failure_preserves_only_reference_and_identity_failure_clears_it(self):
        for transport_failure in (True,False):
            client=Mock();guide=PlacementGuide(lambda:client)
            guide.value=project_status(self.snapshot())
            def poll(*args,**kwargs):
                guide.stop.set()
                if transport_failure:raise RuntimeError('transport unavailable')
                snapshot=self.snapshot();snapshot['camera_serial']='wrong'
                return {'structuredContent':{'exit_code':0,'truncated':False,'stdout':json.dumps(snapshot)}}
            client.tool.side_effect=poll;guide.update();result=guide.read()
            with self.subTest(transport_failure=transport_failure):
                self.assertFalse(result['valid'])
                self.assertEqual(bool(result.get('polygon')),transport_failure)
            client.process.terminate.assert_called_once()


if __name__=='__main__':unittest.main()
