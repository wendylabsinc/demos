"""Actual captured geometry acceptance; network and command IO stay disabled."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from offline.pickup_area_replay import run
from test_placement_overlay import PickupAreaTests
from placement_overlay import SCRIPT, project_area

class PickupAreaAcceptanceTests(unittest.TestCase):
    def test_original_three_captures_reproduce_area_rejection(self):
        for index in range(3):
            with self.subTest(index=index):
                result=run(index,historical_bounds=True)
                self.assertFalse(result['passed'])
                self.assertIn('outside the left-hand pickup area',result['error'])

    def test_two_reachable_captures_pass_complete_real_clock_preflight(self):
        for index in (1,2):
            with self.subTest(index=index):
                result=run(index)
                self.assertTrue(result['passed'],result.get('error'))
                self.assertFalse(result['actuation']);self.assertFalse(result['physical_lift_verified'])
                self.assertGreater(result['clearance']['samples'],100)
                self.assertGreaterEqual(result['clearance']['min_table_distance_m'],.01)
                self.assertGreater(result['clearance']['min_elbow_torso_distance_m'],0.)
                self.assertLess(result['elapsed_s'],30.)
                event=next(e for e in result['events'] if e['phase']=='path_preflight_passed')
                self.assertEqual(event['waypoints_checked'],4);self.assertEqual(event['hand_paths_checked'],2)
                self.assertEqual(event['motion_commands_sent'],0)

    def test_far_capture_remains_outside_bounded_admission(self):
        result=run(0)
        self.assertFalse(result['passed'])
        self.assertIn('forward=50.2 cm',result['error'])
        self.assertIn('requires forward=25.5–45 cm and robot-left=6.5–29 cm',result['error'])

    def test_controls_read_and_project_actual_deployed_task_bounds(self):
        snapshot=PickupAreaTests().snapshot()
        metadata=dict(source_metadata=dict(timestamp_valid=True,depth_time=99.97,color_time=99.97,camera_serial='test'),
                      calibration=dict(intrinsics=snapshot['intrinsics']))
        raw=io.BytesIO()
        with zipfile.ZipFile(raw,'w') as archive:archive.writestr('metadata.json',json.dumps(metadata))
        responses={'/status':json.dumps(dict(demo_running=False,phase='PASSIVE',measured=dict(body=snapshot['body']))).encode(),
                   '/health':json.dumps(snapshot['camera_health']).encode(),'/frame.tracking-rgbd':raw.getvalue()}
        source=(Path(__file__).parent/'g1_can_handoff/task.py').read_text()
        def read(path,*args,**kwargs):return json.dumps(snapshot['config']) if str(path)=='/data/calibration.json' else source
        def response(url,**kwargs):return io.BytesIO(responses['/'+url.split('/')[-1]])
        output=io.StringIO()
        with patch('pathlib.Path.read_text',read),patch('urllib.request.urlopen',response),patch('time.monotonic',return_value=100.),contextlib.redirect_stdout(output):
            exec(compile(SCRIPT,'placement_snapshot','exec'),{})
        projected=project_area(json.loads(output.getvalue()))
        self.assertEqual(projected['bounds_m'],{'0':[.255,.45],'1':[.065,.29]})
        self.assertIn('25.5–45 cm forward',projected['label'])
        self.assertIn('6.5–29 cm robot-left',projected['label'])
        np.testing.assert_allclose(projected['polygon'],[[25.5,6.5],[45.,6.5],[45.,29.],[25.5,29.]])

if __name__=='__main__':unittest.main()
