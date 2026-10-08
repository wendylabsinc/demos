"""Measured failures replay through current geometry and command gates offline."""
import gzip
import io
import json
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from camera_input import Camera,geometry
from physical_core import PhysicalPlant
from pose_alignment import corrected_config
from runtime import Runtime
from yolo_client import SCHEMA,identity,validate_result

ROOT=Path(__file__).parent/'offline'

def trials():return json.loads(gzip.decompress((ROOT/'trial-events.json.gz').read_bytes()))

def events(attempt,phase):return [e for e in trials()[attempt]['events'] if e['phase']==phase]

def recorded_frame(stem):
    directory=ROOT/'evidence/rgbd'
    with np.load(directory/'rays.npz',allow_pickle=False) as data:rays=data['rays'].copy()
    timing=(directory/(stem+'.timing.npz')).read_bytes()
    compact=(directory/(stem+'.source.zip')).read_bytes()
    with np.load(io.BytesIO(timing),allow_pickle=False) as data:meta=json.loads(str(data['metadata_json']))
    camera=Camera();camera.rays=rays;camera.intrinsics=meta['intrinsics'];camera.serial=meta['camera_serial']
    def request(path,data=None):
        if path=='/lease/renew':return b'{}'
        if path=='/health':return json.dumps(dict(frame_id=meta['frame_id'],serial=camera.serial)).encode()
        if 'policy-rgbd' in path:return timing
        if 'recorder-rgbd' in path:return compact
        raise AssertionError('Unexpected offline endpoint: '+path)
    camera.request=request
    try:
        with patch('camera_input.time.monotonic',return_value=meta['depth_time']+.25):
            return camera.frame(materialize=False)
    finally:camera.close()

class RecordedFailureTests(unittest.TestCase):
    def test_all_ten_trial_traces_retain_primary_error_and_never_claim_lift(self):
        summaries=[json.loads(line) for line in (ROOT/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual(len(summaries),10)
        for row in summaries:
            with self.subTest(attempt=row['evidence']):
                self.assertFalse(row['physical_lift_verified'])
                attempt=Path(row['evidence']).stem
                failure=events(attempt,'primary_failure')[0]['error']
                if attempt=='20261008T001246Z':
                    # Historical status overwrote the primary fault on cancel.
                    self.assertIn('478.3 ms',failure)
                    self.assertEqual(row['error'],'RuntimeError: Operator cancelled trial')
                else:self.assertEqual(failure,row['error'])
                if row['phase']=='BLOCKED':self.assertEqual(row['motion_commands_sent'],0)
                self.assertFalse(events(attempt,'physical_pickup_verified'))

    def test_captured_jpeg_depth_rays_yolo_identity_and_cylinder_reproduce_planning(self):
        attempt='20261008T004012Z';frame=recorded_frame('camera-44504-planning')
        config=events(attempt,'plan_context')[0]['calibration']
        result=next(e for e in events(attempt,'perception_result') if e['frame_id']==44504)
        pose=next(e['pose'] for e in events(attempt,'exposure_pose') if e['frame_timestamp']==frame['timestamp'])
        frozen=dict(schema=SCHEMA,source=result['source'],model_sha256=result['model_sha256'],
            backend='pytorch_cuda',truncated=False,detections=result['detections'])
        self.assertEqual(identity(frame),frozen['source'])
        detector=SimpleNamespace(detect=lambda exact:validate_result(exact,frozen))
        with patch('time.monotonic',return_value=frame['timestamp']+.25):
            observation,vision=geometry(frame,corrected_config(config,pose),detector=detector)
        np.testing.assert_allclose(observation.center[:2],[.386271602795,.118108939549],atol=2e-6)
        self.assertLess(vision['fit_error_m'],.006)

    def test_153_measured_table_frames_reproduce_uncorrected_drift_and_corrected_stability(self):
        old=[];aligned=[]
        with np.load(ROOT/'evidence/attitude-replay.npz',allow_pickle=False) as data:
            meta=json.loads(str(data['metadata_json']));self.assertEqual(len(meta['frames']),153)
            for i,row in enumerate(meta['frames']):
                points=data['points_'+str(i)];config=meta['config']
                for target,transform in ((old,config),(aligned,corrected_config(config,row['pose']))):
                    matrix=np.array(transform['camera_to_world']);z=(points@matrix[:3,:3].T+matrix[:3,3])[:,2]
                    target.append(float(np.median(z)-.762))
        self.assertGreater(max(old),.005)
        self.assertLess(max(abs(x) for x in aligned),.0015)

    def test_recorded_411ms_gap_blocks_next_arm_and_finger_tick(self):
        attempt='20261008T004101Z';failure=events(attempt,'primary_failure')[0]
        age_ms=float(re.search(r'\(([0-9.]+) ms\)',failure['error'])[1])
        frame=recorded_frame('camera-45594-motion-tracking')
        runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock();runtime.tracking_error=None
        runtime.tracking_observation=SimpleNamespace(timestamp=frame['timestamp'])
        plant=PhysicalPlant.__new__(PhysicalPlant);plant.cancelled=False;plant.cancel=threading.Event()
        plant.perception_guard=runtime.tracking_guard;plant.command_lock=threading.RLock()
        plant.clock=time
        plant.command_arm=Mock();plant.sender=Mock();plant.read=Mock()
        with patch('runtime.time.monotonic',return_value=frame['timestamp']+.322938):runtime.tracking_guard()
        with patch('runtime.time.monotonic',return_value=frame['timestamp']+age_ms/1000):
            with self.assertRaisesRegex(RuntimeError,'411.8 ms'):
                plant._execute_poses(np.zeros(7),[np.zeros(7)])
        plant.command_arm.assert_not_called();plant.sender.ctrl_dual_hand.assert_not_called()
        plant.read.assert_not_called()
        # This is a guard regression: the original latency is still an open bug.
        next_result=events(attempt,'perception_result')[-1]
        self.assertGreater(next_result['monotonic_ns'],failure['monotonic_ns'])

    def test_cold_ownership_shift_is_reproduced_and_rejected_without_moving_baseline(self):
        observations=events('20261008T004012Z','perception_result')
        initial,last=observations[0]['geometry'],observations[-1]['geometry']
        baseline=np.r_[initial['center_xy_m'],.823];shift=np.linalg.norm(np.array(last['center_xy_m'])-baseline[:2])
        self.assertGreater(shift,.010);self.assertLess(shift,.011)
        runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock();runtime.tracking_error=None
        runtime.tracking_baseline=SimpleNamespace(center=baseline);runtime.tracking_observation=None
        observed=SimpleNamespace(center=np.r_[last['center_xy_m'],.823],timestamp=100.)
        with patch('runtime.time.monotonic',return_value=100.25):
            with self.assertRaisesRegex(RuntimeError,'outside the checked pickup path'):
                runtime.publish_tracking_observation(observed,last,True)
        self.assertIsNone(runtime.tracking_observation)
        np.testing.assert_array_equal(runtime.tracking_baseline.center,baseline)

if __name__=='__main__':unittest.main()
