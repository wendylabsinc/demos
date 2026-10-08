"""The optimized fetch retains real exposure, calibration and frame identity."""
import io
import json
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
from test_camera_pipeline import FakeRGBDHTTP


class CompactTrackingTests(unittest.TestCase):
    def test_all_world_points_match_previous_arithmetic_exactly(self):
        from camera_input import world_points
        from scipy.spatial.transform import Rotation
        random=np.random.default_rng(42)
        for dtype in (np.float32,np.float64):
            depth=random.uniform(.1,2.5,(120,160)).astype(dtype)
            rays=random.uniform(-1,1,(120,160,3)).astype(dtype)
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',[.1,.6,-.03]).as_matrix();t[:3,3]=[.05,-.04,.91]
            expected=(rays*depth[...,None])@t[:3,:3].T+t[:3,3]
            np.testing.assert_array_equal(world_points({'depth_m':depth,'rays':rays},
                {'camera_to_world':t.tolist()}),expected)

    def fixture(self):
        http=FakeRGBDHTTP();http.meta['stream_id']='original-stream'
        camera=http.camera();camera.tracking_compact=True
        paths=[]
        def request(path,data=None):
            paths.append(path)
            if path=='/lease/renew':return b'{}'
            if not path.startswith('/frame.tracking-rgbd'):raise AssertionError(path)
            raw=io.BytesIO()
            with zipfile.ZipFile(raw,'w') as archive:
                archive.writestr('metadata.json',json.dumps({**http.compact,'source_metadata':http.meta}))
                archive.writestr('depth.png',http.png);archive.writestr('color.jpg',http.jpg)
            return raw.getvalue()
        camera.request=request
        self.addCleanup(camera.close)
        return http,camera,paths

    def read(self,camera):
        with patch('camera_input.time.monotonic',return_value=100.):
            return camera.frame(materialize=False)

    def test_single_bundle_preserves_decoded_pixels_and_original_exposure(self):
        http,camera,paths=self.fixture();frame=self.read(camera)
        self.assertEqual(paths,['/lease/renew','/frame.tracking-rgbd'])
        self.assertEqual(frame['timestamp'],99.95)
        np.testing.assert_array_equal(frame['depth_m'],http.depth.astype(np.float32))
        self.assertEqual(frame['timing_source'],'compact_source_metadata')
        with np.load(io.BytesIO(frame['source_timing_npz']),allow_pickle=False) as archive:
            self.assertEqual(json.loads(str(archive['metadata_json'])),http.meta)

    def test_next_request_demands_a_newer_frame_in_the_same_stream(self):
        http,camera,paths=self.fixture();self.read(camera)
        with self.assertRaisesRegex(RuntimeError,'identity did not advance'):self.read(camera)
        self.assertIn('after_frame_id=12&stream_id=original-stream',paths[-1])
        http.meta.update(frame_id=13,stream_id='restarted');http.compact['source_frame_id']=13
        with self.assertRaisesRegex(RuntimeError,'identity did not advance'):self.read(camera)

    def test_stale_future_invalid_and_skewed_frames_still_fail(self):
        for changes in [dict(depth_time=99.599),dict(depth_time=100.001),dict(timestamp_valid=False),
                        dict(depth_time=float('nan')),dict(color_time=99.97)]:
            with self.subTest(changes=changes):
                http,camera,_=self.fixture();http.meta.update(changes)
                with self.assertRaisesRegex(RuntimeError,'exposure|skew'):self.read(camera)

    def test_mismatched_identity_scale_and_calibration_still_fail(self):
        for field in ['source_frame_id','captured_at_unix_ns']:
            http,camera,_=self.fixture();http.compact[field]+=1
            with self.assertRaisesRegex(RuntimeError,'identity mismatch'):self.read(camera)
        http,camera,_=self.fixture();http.compact['calibration']['depth_scale_m']=.002
        with self.assertRaisesRegex(RuntimeError,'scale'):self.read(camera)
        http,camera,_=self.fixture();http.meta['camera_serial']='changed'
        with self.assertRaisesRegex(RuntimeError,'calibration changed'):self.read(camera)
        self.assertIsNone(camera.rays)


if __name__=='__main__':unittest.main()
