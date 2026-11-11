"""Exact-frame decoder and worker-lifetime regressions for the hot path."""
import io
import json
import time
import unittest
import zipfile
from unittest.mock import patch
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from camera_input import Camera,decode_color,decode_depth,world_points


class FakeRGBDHTTP:
    """Immutable encoded frame bundles served through the real HTTP client."""
    def __init__(self,color=None,depth=None,rays=None,serial='paired'):
        self.color=np.random.default_rng(4).integers(0,256,(48,64,3),dtype=np.uint8) if color is None else color
        self.depth=np.ones(self.color.shape[:2],np.float32) if depth is None else depth
        self.rays=np.ones((*self.depth.shape,3)) if rays is None else rays
        self.serial=serial;self.scale=.001;self.paths=[];self.yolo=None
        self.intrinsics=dict(width=self.depth.shape[1],height=self.depth.shape[0])
        self.advance(12,99.95)

    def advance(self,fid,timestamp,color=None):
        if color is not None:self.color=color
        self.fid=fid
        self.meta=dict(frame_id=fid,captured_at_unix_ns=100000+fid,
            intrinsics=dict(self.intrinsics),camera_serial=self.serial,
            depth_time=timestamp,color_time=timestamp,timestamp_valid=True,depth_scale_m=self.scale)
        self.compact=dict(source_frame_id=fid,captured_at_unix_ns=100000+fid,
            calibration=dict(serial=self.serial,intrinsics=dict(self.intrinsics),depth_scale_m=self.scale))
        png=io.BytesIO();Image.fromarray(np.rint(self.depth/self.scale).astype(np.uint16)).save(png,format='PNG')
        jpg=io.BytesIO();Image.fromarray(self.color[...,::-1]).save(jpg,format='JPEG',quality=95)
        self.png=png.getvalue();self.jpg=jpg.getvalue()

    def camera(self):
        camera=Camera();camera.rays=self.rays.copy();camera.intrinsics=dict(self.intrinsics);camera.serial=self.serial
        return camera

    def urlopen(self,request,timeout=None):
        url=urlsplit(request.full_url);path=url.path;self.paths.append(path)
        if url.port==8112:
            if self.yolo is None:raise AssertionError('Unexpected YOLO request')
            payload=self.yolo(request)
        elif path=='/lease/renew':payload=b'{}'
        elif path=='/health':payload=json.dumps({'frame_id':self.fid,'serial':self.serial}).encode()
        elif path=='/frame.policy-rgbd':
            self.check_fid(url.query)
            bundle=io.BytesIO();np.savez(bundle,metadata_json=json.dumps(self.meta));payload=bundle.getvalue()
        elif path=='/frame.recorder-rgbd':
            self.check_fid(url.query)
            bundle=io.BytesIO()
            with zipfile.ZipFile(bundle,'w') as archive:
                archive.writestr('metadata.json',json.dumps(self.compact))
                archive.writestr('depth.png',self.png);archive.writestr('color.jpg',self.jpg)
            payload=bundle.getvalue()
        else:raise AssertionError('Unexpected HTTP endpoint: '+request.full_url)
        return io.BytesIO(payload)

    def check_fid(self,query):
        if query!='frame_id='+str(self.fid):raise AssertionError('Request did not pin exact frame ID')


class WorldTransformTests(unittest.TestCase):
    def test_parallel_transform_retains_every_pixel_and_original_arithmetic(self):
        rng=np.random.default_rng(219)
        for height,width in [(480,640),(7,11),(1,3),(0,3)]:
            for ray_dtype in [np.float32,np.float64]:
                rays=rng.normal(size=(height,width,3)).astype(ray_dtype)
                depth=rng.uniform(0,2,(height,width)).astype(np.float32)
                if height:
                    depth[0,0]=np.nan;depth[-1,-1]=0
                t=np.eye(4);t[:3,:3]=rng.normal(size=(3,3));t[:3,3]=rng.normal(size=3)
                points=rays*depth[...,None]
                expected=np.empty(points.shape,dtype=np.result_type(points,t))
                np.matmul(points,t[:3,:3].T,out=expected);expected+=t[:3,3]
                frame=dict(rays=rays,depth_m=depth)
                saved_rays=rays.copy();saved_depth=depth.copy()
                for workers in (1,2,4):
                    actual=world_points(frame,{'camera_to_world':t},workers=workers)
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(actual.dtype,expected.dtype)
                np.testing.assert_array_equal(rays,saved_rays)
                np.testing.assert_array_equal(depth,saved_depth)

    def test_worker_configuration_is_bounded_and_serial_is_available(self):
        frame=dict(rays=np.ones((5,3,3)),depth_m=np.ones((5,3)))
        config={'camera_to_world':np.eye(4)}
        with patch.dict('os.environ',{'HANDOFF_GEOMETRY_WORKERS':'1'}):
            with patch('camera_input._world_executor.submit',side_effect=AssertionError('Unexpected worker')):
                np.testing.assert_array_equal(world_points(frame,config),frame['rays'])
        for invalid in ['0','3','8','invalid','4.0']:
            with patch.dict('os.environ',{'HANDOFF_GEOMETRY_WORKERS':invalid}):
                with self.assertRaises(ValueError):world_points(frame,config)

    def test_parallel_transform_propagates_bad_input_instead_of_partial_results(self):
        frame=dict(rays=np.ones((5,3,3)),depth_m=np.ones((5,4)))
        with self.assertRaises(ValueError):
            world_points(frame,{'camera_to_world':np.eye(4)},workers=4)


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
                              camera_serial='paired',depth_time=timestamp,color_time=timestamp,timestamp_valid=True,depth_scale_m=.001)
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

    def assert_bundle_rejected(self,http,pattern):
        camera=http.camera()
        try:
            with patch('urllib.request.urlopen',side_effect=http.urlopen),patch('camera_input.time.monotonic',return_value=100.):
                with self.assertRaisesRegex(RuntimeError,pattern):camera.frame(materialize=False)
        finally:camera.close()
        return camera

    def test_camera_http_rejects_stale_future_and_invalid_exposure(self):
        for timestamp,valid in ((99.599,True),(100.001,True),(99.95,False),(float('nan'),True)):
            with self.subTest(timestamp=timestamp,valid=valid):
                http=FakeRGBDHTTP();http.meta.update(depth_time=timestamp,color_time=timestamp,timestamp_valid=valid)
                self.assert_bundle_rejected(http,'exposure timestamp')

    def test_camera_http_accepts_configured_freshness_boundary_without_retimestamping(self):
        http=FakeRGBDHTTP();http.meta.update(depth_time=99.6001,color_time=99.6001)
        camera=http.camera()
        try:
            with patch('urllib.request.urlopen',side_effect=http.urlopen),patch('camera_input.time.monotonic',return_value=100.):
                frame=camera.frame(materialize=False)
            self.assertEqual(frame['timestamp'],99.6001)
            self.assertEqual(frame['meta']['depth_time'],99.6001)
            self.assertNotIn('raw',frame)
        finally:camera.close()

    def test_camera_http_rejects_positive_and_negative_rgb_depth_skew(self):
        for skew in (-.0121,.0121,float('inf'),float('nan')):
            with self.subTest(skew=skew):
                http=FakeRGBDHTTP();http.meta['color_time']+=skew
                self.assert_bundle_rejected(http,'skew|timestamp')

    def test_camera_http_rejects_policy_frame_and_capture_time_mismatch(self):
        for field in ('frame_id','captured_at_unix_ns'):
            with self.subTest(field=field):
                http=FakeRGBDHTTP();http.meta[field]+=1
                self.assert_bundle_rejected(http,'exact-frame identity')

    def test_camera_http_invalidates_ray_cache_on_timing_calibration_change(self):
        for field,value in (('camera_serial','other'),('intrinsics',{'width':65,'height':48})):
            with self.subTest(field=field):
                http=FakeRGBDHTTP();http.meta[field]=value
                camera=self.assert_bundle_rejected(http,'calibration changed')
                self.assertIsNone(camera.rays)

    def test_camera_http_rejects_compact_intrinsics_and_dimensions_mismatch(self):
        http=FakeRGBDHTTP();http.compact['calibration']['intrinsics']['width']+=1
        self.assert_bundle_rejected(http,'intrinsics disagree')
        http=FakeRGBDHTTP();smaller=io.BytesIO();Image.fromarray(np.ones((47,64),np.uint16)).save(smaller,format='PNG')
        http.png=smaller.getvalue()
        self.assert_bundle_rejected(http,'dimensions disagree')

    def test_camera_http_rejects_compact_calibration_serial_mismatch(self):
        http=FakeRGBDHTTP();http.compact['calibration']['serial']='other-camera'
        self.assert_bundle_rejected(http,'identity|serial|calibration')

    def test_camera_http_rejects_compact_depth_scale_mismatch(self):
        for scale in (.002,0.,-.001,float('nan')):
            with self.subTest(scale=scale):
                http=FakeRGBDHTTP();http.compact['calibration']['depth_scale_m']=scale
                self.assert_bundle_rejected(http,'scale|calibration')


if __name__=='__main__':unittest.main()
