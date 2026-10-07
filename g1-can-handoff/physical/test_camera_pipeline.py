"""Exact-frame decoder and worker-lifetime regressions for the hot path."""
import io
import json
import time
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from PIL import Image

from camera_input import Camera,decode_color,decode_depth


class CameraPipelineTests(unittest.TestCase):
    def camera(self):
        camera=Camera()
        rgb=np.random.default_rng(4).integers(0,256,(48,64,3),dtype=np.uint8)
        depth=np.arange(48*64,dtype=np.uint16).reshape(48,64)
        png=io.BytesIO();Image.fromarray(depth).save(png,format='PNG')
        jpg=io.BytesIO();Image.fromarray(rgb).save(jpg,format='JPEG')
        intrinsics={'width':64,'height':48}
        camera.rays=np.ones((48,64,3));camera.intrinsics=intrinsics;camera.serial='paired'
        self.png=png.getvalue();self.jpg=jpg.getvalue();self.depth=depth
        compactmeta={'source_frame_id':12,'captured_at_unix_ns':33,
                     'calibration':{'serial':'paired','intrinsics':intrinsics,'depth_scale_m':.001}}
        def request(path,data=None):
            if path=='/health':return json.dumps({'frame_id':12,'serial':'paired'}).encode()
            if path=='/lease/renew':return b'{}'
            if 'policy-rgbd' in path:
                timestamp=time.monotonic()
                metadata=dict(frame_id=12,captured_at_unix_ns=33,intrinsics=intrinsics,
                              camera_serial='paired',depth_time=timestamp,color_time=timestamp,timestamp_valid=True)
                result=io.BytesIO();np.savez(result,metadata_json=json.dumps(metadata));return result.getvalue()
            if 'recorder-rgbd' in path:
                result=io.BytesIO()
                with zipfile.ZipFile(result,'w') as z:
                    z.writestr('metadata.json',json.dumps(compactmeta));z.writestr('depth.png',self.png);z.writestr('color.jpg',self.jpg)
                return result.getvalue()
            self.fail('Unexpected camera endpoint: '+path)
        camera.request=request
        return camera,compactmeta

    def test_paired_decode_matches_original_pixels_and_reuses_workers(self):
        camera,_=self.camera()
        try:
            first=camera.frame(materialize=False);pool=camera.frame_executor
            second=camera.frame(materialize=False)
            self.assertIs(pool,camera.frame_executor)
            with Image.open(io.BytesIO(self.jpg)) as image:
                expected=np.asarray(image.convert('RGB'))[...,::-1].copy()
            np.testing.assert_array_equal(first['color_bgr'],expected)
            np.testing.assert_array_equal(first['depth_m'],self.depth.astype(np.float32)*.001)
            np.testing.assert_array_equal(first['color_bgr'],second['color_bgr'])
            self.assertEqual(first['meta']['frame_id'],12)
            self.assertIn('decode',first['timings_ms'])
        finally:camera.close()
        self.assertIsNone(camera.frame_executor)
        self.assertTrue(pool._shutdown)

    def test_parallel_decode_retains_exact_frame_rejection(self):
        camera,metadata=self.camera();metadata['source_frame_id']=13
        try:
            with self.assertRaisesRegex(RuntimeError,'exact-frame identity mismatch'):
                camera.frame(materialize=False)
        finally:camera.close()

    def test_both_decoders_run_before_either_is_joined(self):
        import threading
        camera,_=self.camera();started={kind:threading.Event() for kind in ('depth','color')}
        def decode(kind,original,*args):
            started[kind].set()
            if not started['color' if kind=='depth' else 'depth'].wait(1):
                raise AssertionError('Exact paired decoders ran serially')
            return original(*args)
        try:
            with patch('camera_input.decode_depth',side_effect=lambda *a:decode('depth',decode_depth,*a)),\
                 patch('camera_input.decode_color',side_effect=lambda *a:decode('color',decode_color,*a)):
                camera.frame(materialize=False)
        finally:camera.close()


if __name__=='__main__':unittest.main()
