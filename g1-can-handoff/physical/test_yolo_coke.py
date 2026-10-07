"""Semantic target, RGB-D binding, and existing geometry failure regressions."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).parent/'yolo'))

from camera_input import geometry
from yolo_client import MODEL_SHA256, SCHEMA, identity, validate_result, YoloClient


class Detector:
    def __init__(self, boxes=None):
        self.boxes=boxes or [dict(xyxy=[60,25,100,85],confidence=.95,class_id=0,class_name='Coca-Cola')]
    def detect(self, frame):
        result=dict(schema=SCHEMA,source=identity(frame),model_sha256=MODEL_SHA256,
                    backend='pytorch_cuda',truncated=False,detections=self.boxes)
        return validate_result(frame,result)


class FrameFixture:
    def frame(self):
        depth=np.ones((120,160),dtype=np.float32)
        color=np.zeros((120,160,3),dtype=np.uint8)
        rays=np.zeros((120,160,3));rays[:,:,2]=1
        ys,xs=np.mgrid[25:85,60:100]
        angle=np.linspace(2.,4.2,40)[None,:]
        # White/grey label: semantic box must work without any red pixels.
        color[ys,xs]=180;depth[ys,xs]=.95
        rays[ys,xs,0]=(.35+.033*np.cos(angle))/.95
        rays[ys,xs,1]=(.18+.033*np.sin(angle))/.95
        return dict(color_bgr=color,depth_m=depth,rays=rays,timestamp=time.monotonic(),
                    meta=dict(camera_serial='nx2-test',frame_id=123))
    def config(self):
        return dict(camera_serial='nx2-test',camera_to_world=np.eye(4).tolist(),
                    can_radius_m=.033,can_height_m=.122,table_height_m=.762)


class CokeGeometryTests(FrameFixture, unittest.TestCase):
    def test_lid_interior_does_not_enter_cylinder_sidewall_fit(self):
        frame=self.frame();config=self.config()
        # Background is the measured table, with an upright cylinder below
        # a broad flat lid. Grey pixels throughout prevent a color shortcut.
        frame['rays'][:,:,0]=.35
        frame['rays'][:,:,1]=.18
        frame['rays'][:,:,2]=.762
        ys,xs=np.mgrid[25:85,60:100]
        angle=np.linspace(2.,4.2,40)[None,:]
        frame['rays'][ys,xs,0]=(.35+.033*np.cos(angle))/.95
        frame['rays'][ys,xs,1]=(.18+.033*np.sin(angle))/.95
        frame['rays'][ys,xs,2]=np.linspace(.884,.782,60)[:,None]/.95
        ys,xs=np.mgrid[25:53,60:100]
        frame['rays'][ys,xs,0]=(.35+np.linspace(-.02,.02,40)[None,:])/.95
        frame['rays'][ys,xs,1]=(.18+np.linspace(-.02,.02,28)[:,None])/.95
        frame['rays'][ys,xs,2]=.884/.95
        frame['rays'][53:85,60:100,2]=np.linspace(.874,.782,32)[:,None]/.95
        _,vision=geometry(frame,config,detector=Detector())
        np.testing.assert_allclose(vision['center_xy_m'],[.35,.18],atol=1e-6)
        self.assertLess(vision['fit_error_m'],.006)
        # The same cap exclusion must continue after lift, where table-relative
        # height is no longer the can's top.
        frame['timestamp']=time.monotonic()
        frame['rays'][25:85,60:100,2]+=.15/.95
        _,lifted=geometry(frame,config,require_table=False,
                          expected_center_xy=[.35,.18],detector=Detector())
        self.assertLess(lifted['fit_error_m'],.006)
        self.assertGreater(lifted['top_quantile_z_m']-vision['top_quantile_z_m'],.13)

    def test_nonred_coke_label_fits_measured_cylinder(self):
        frame=self.frame()
        _,vision=geometry(frame,self.config(),require_table=False,detector=Detector())
        np.testing.assert_allclose(vision['center_xy_m'],[.35,.18],atol=1e-6)
        self.assertEqual(vision['detector'],'coke_yolo')
    def test_red_object_without_semantic_detection_fails(self):
        frame=self.frame();frame['color_bgr'][25:85,60:100]=[0,0,255]
        detector=Detector();detector.boxes=[]
        with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):
            geometry(frame,self.config(),require_table=False,detector=detector)
    def test_multiple_cokes_cannot_choose_highest_confidence(self):
        detector=Detector();detector.boxes.append(dict(detector.boxes[0],xyxy=[110,25,150,85],confidence=.8))
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):
            geometry(self.frame(),self.config(),require_table=False,detector=detector)
    def test_low_confidence_cannot_enable_grasp(self):
        detector=Detector();detector.boxes[0]['confidence']=.30
        with self.assertRaisesRegex(RuntimeError,'absent'):
            geometry(self.frame(),self.config(),require_table=False,detector=detector)
    def test_confidence_just_above_30_percent_is_eligible(self):
        detector=Detector();detector.boxes[0]['confidence']=.3001
        _,vision=geometry(self.frame(),self.config(),require_table=False,detector=detector)
        self.assertEqual(vision['box']['confidence'],.3001)

    def test_threshold_cannot_be_lowered_below_user_required_30_percent(self):
        with patch.dict('os.environ',{'HANDOFF_YOLO_MIN_CONFIDENCE':'0.29'}):
            with self.assertRaisesRegex(ValueError,'0.30'):
                geometry(self.frame(),self.config(),require_table=False,detector=Detector())

    def test_clipped_box_rejected_even_with_tracking_roi(self):
        detector=Detector();detector.boxes[0]['xyxy']=[60,0,100,85]
        with self.assertRaisesRegex(RuntimeError,'clipped'):
            geometry(self.frame(),self.config(),require_table=False,expected_center_xy=[.35,.18],detector=detector)
    def test_off_target_detection_cannot_replace_lifted_coke(self):
        with self.assertRaisesRegex(RuntimeError,'absent'):
            geometry(self.frame(),self.config(),require_table=False,expected_center_xy=[.7,.18],detector=Detector())
    def test_tracking_cannot_skip_initial_ambiguity(self):
        with self.assertRaisesRegex(ValueError,'only allowed after lifting'):
            geometry(self.frame(),self.config(),expected_center_xy=[.35,.18],detector=Detector())
    def test_no_depth_cannot_enable_geometry(self):
        frame=self.frame();frame['depth_m'][:]=np.nan
        with self.assertRaisesRegex(RuntimeError,'absent'):
            geometry(frame,self.config(),require_table=False,detector=Detector())
    def test_table_mismatch_still_blocks_semantic_target(self):
        config=self.config();config['table_height_m']=.9
        with self.assertRaisesRegex(RuntimeError,'table height'):
            geometry(self.frame(),config,detector=Detector())
    def test_cylinder_fit_still_blocks_noncan_geometry(self):
        frame=self.frame();frame['rays'][25:85,60:100,:2]=[.35/.95,.18/.95]
        # A flat small patch cannot fit the known radius with 6 mm residual.
        frame['rays'][25:85,60:100,0]=np.linspace(.2,.5,40)[None,:]/.95
        with self.assertRaisesRegex(RuntimeError,'cylinder fit'):
            geometry(frame,self.config(),require_table=False,detector=Detector())
    def test_physical_source_cannot_use_simulation_red_mode(self):
        with self.assertRaisesRegex(ValueError,'MuJoCo'):
            geometry(self.frame(),self.config(),synthetic_red=True)
    def test_physical_default_calls_yolo_and_does_not_fallback(self):
        with patch.object(YoloClient,'detect',side_effect=RuntimeError('offline')):
            with self.assertRaisesRegex(RuntimeError,'offline'):
                geometry(self.frame(),self.config(),require_table=False)


class ExactFrameTests(FrameFixture, unittest.TestCase):
    def test_rejected_response_identity_and_backend(self):
        frame=self.frame();good=Detector().detect(frame)
        for key,value in [('frame_id',124),('camera_serial','other'),('image_sha256','0'*64),('timestamp',frame['timestamp']-.1),('shape',[1,1,3])]:
            bad=copy.deepcopy(good);bad['source'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'identity'):
                validate_result(frame,bad)
        for key,value in [('model_sha256','0'*64),('backend','cpu'),('truncated',True),('schema','old')]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):validate_result(frame,bad)
    def test_stale_or_future_frame_rejected(self):
        for age in (.401,-.1):
            frame=self.frame();frame['timestamp']=time.monotonic()-age
            with self.assertRaisesRegex(RuntimeError,'stale'):Detector().detect(frame)
    def test_invalid_box_and_wrong_class_rejected(self):
        frame=self.frame();good=Detector().detect(frame)
        for key,value in [('xyxy',[-1,25,100,85]),('xyxy',[60,25,60,85]),('confidence',float('nan')),('class_id',1),('class_name','red')]:
            bad=copy.deepcopy(good);bad['detections'][0][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):validate_result(frame,bad)


class ServiceContractTests(FrameFixture, unittest.TestCase):
    def test_request_decoding_checks_pixels_dimensions_and_age(self):
        path=Path(__file__).parent/'yolo/service.py'
        spec=importlib.util.spec_from_file_location('coke_yolo_service',path)
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        frame=self.frame();raw=frame['color_bgr'].tobytes();source=identity(frame)
        np.testing.assert_array_equal(service.decode_frame(source,raw),frame['color_bgr'])
        for key,value in [('image_sha256','0'*64),('shape',[120,160,4]),('timestamp',time.monotonic()-1)]:
            bad=dict(source);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):service.decode_frame(bad,raw)
    def test_http_roundtrip_preserves_exact_bgr_and_identity(self):
        import threading
        from http.server import HTTPServer
        path=Path(__file__).parent/'yolo/service.py'
        spec=importlib.util.spec_from_file_location('coke_yolo_http',path)
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        frame=self.frame();frame['color_bgr'][35,70]=[11,22,199]
        worker=service.Detector.__new__(service.Detector)
        seen=[]
        def predict(image):
            seen.append(image.copy())
            return Detector().boxes,False
        worker.predict=predict
        server=HTTPServer(('127.0.0.1',0),service.Handler)
        server.detector=worker
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            client=YoloClient();client.url='http://127.0.0.1:'+str(server.server_port)
            result=client.detect(frame)
            np.testing.assert_array_equal(seen[0],frame['color_bgr'])
            self.assertEqual(result['source'],identity(frame))
        finally:
            server.shutdown();server.server_close();thread.join(2)

    def test_pinned_artifact_matches_manifest_and_bytes(self):
        import json
        root=Path(__file__).parent/'yolo/model'
        manifest=json.loads((root/'SOURCE.json').read_text())
        self.assertEqual(hashlib.sha256((root/'coke-can.pt').read_bytes()).hexdigest(),MODEL_SHA256)
        self.assertEqual(manifest['sha256'],MODEL_SHA256)


if __name__=='__main__':unittest.main()
