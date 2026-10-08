"""Initial semantic identity, conservative tracking and rejection regressions."""
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from target_tracker import TargetTracker
from test_yolo_coke import FrameFixture
from yolo_client import MODEL_SHA256,SCHEMA,identity,validate_result


class CountingDetector:
    def __init__(self):
        self.calls=0
        self.boxes=[dict(xyxy=[60,25,100,85],confidence=.95,class_id=0,class_name='Coca-Cola')]
    def detect(self,frame):
        self.calls+=1
        return validate_result(frame,dict(schema=SCHEMA,source=identity(frame),model_sha256=MODEL_SHA256,
            backend='pytorch_cuda',truncated=False,detections=self.boxes))


class TrackerTests(FrameFixture,unittest.TestCase):
    def setUp(self):
        self.detector=CountingDetector();self.tracker=TargetTracker(self.detector)
        self.fid=122
    def next_frame(self):
        frame=self.frame();self.fid+=1;frame['meta']['frame_id']=self.fid
        return frame
    def select(self,frame):
        world=frame['rays']*frame['depth_m'][...,None]
        return self.tracker.select(frame,self.config(),world,require_table=False)
    def accept(self,frame,vision):
        self.tracker.accept(SimpleNamespace(center=np.array([.35,.18,.95])),vision,frame)
    def seed(self):
        frame=self.next_frame();mask,vision=self.select(frame);self.accept(frame,vision)
        return frame

    def test_initial_semantic_identity_and_pending_commit(self):
        frame=self.next_frame();mask,vision=self.select(frame)
        self.assertEqual(self.detector.calls,1);self.assertGreater(mask.sum(),40)
        self.assertIsNone(self.tracker.state)
        self.accept(frame,vision);self.assertIsNotNone(self.tracker.state)
        self.assertEqual(vision['tracking']['reason'],'initial_identity')

    def test_visual_depth_frames_do_not_requery_yolo(self):
        self.seed()
        for _ in range(4):
            frame=self.next_frame();mask,vision=self.select(frame)
            self.assertEqual(vision['tracking']['mode'],'visual_depth')
            self.assertGreater(mask.sum(),40);self.accept(frame,vision)
        self.assertEqual(self.detector.calls,1)

    def test_periodic_identity_recheck_uses_current_frame(self):
        self.seed();self.tracker.state['semantic_timestamp']-=1.01
        frame=self.next_frame();_,vision=self.select(frame)
        self.assertEqual(self.detector.calls,2)
        self.assertEqual(vision['tracking']['reason'],'periodic_recheck')
        self.assertEqual(vision['tracking']['semantic_frame_id'],frame['meta']['frame_id'])

    def test_phase_tracking_does_not_reclassify_a_confident_track_after_time_passes(self):
        self.tracker=TargetTracker(self.detector,recheck_interval_s=None)
        self.seed();self.tracker.state['semantic_timestamp']-=90
        # The latest failure was a periodic empty YOLO result despite a
        # confident visual/depth track. It must not be queried on elapsed time.
        self.detector.boxes=[]
        for _ in range(5):
            frame=self.next_frame();_,vision=self.select(frame)
            self.assertEqual(vision['tracking']['mode'],'visual_depth')
            self.accept(frame,vision)
        self.assertEqual(self.detector.calls,1)

    def test_phase_tracking_still_reacquires_uncertain_identity(self):
        self.tracker=TargetTracker(self.detector,recheck_interval_s=None)
        self.seed();self.detector.boxes=[]
        frame=self.next_frame();frame['color_bgr'][25:85,60:100]=[0,255,0]
        with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):self.select(frame)
        self.assertEqual(self.detector.calls,2)

    def test_visual_uncertainty_forces_same_frame_semantic_confirmation(self):
        self.seed();frame=self.next_frame();frame['color_bgr'][25:85,60:100]=[0,255,0]
        _,vision=self.select(frame)
        self.assertEqual(self.detector.calls,2)
        self.assertEqual(vision['tracking']['mode'],'yolo')
        self.assertEqual(vision['tracking']['semantic_frame_id'],frame['meta']['frame_id'])

    def test_uncertain_empty_or_ambiguous_identity_never_returns_old_mask(self):
        for boxes in ([],[dict(xyxy=[60,25,100,85],confidence=.95,class_id=0,class_name='Coca-Cola')]*2):
            self.setUp();self.seed();self.detector.boxes=boxes
            frame=self.next_frame();frame['color_bgr'][25:85,60:100]=[0,255,0]
            with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):self.select(frame)
            self.assertIsNone(self.tracker.pending)

    def test_stale_future_or_repeated_frame_rejected(self):
        seeded=self.seed()
        for age in (.401,-.1):
            frame=self.next_frame();frame['timestamp']=time.monotonic()-age
            with self.assertRaisesRegex(RuntimeError,'stale'):self.select(frame)
        with self.assertRaisesRegex(RuntimeError,'distinct'):self.select(seeded)
        self.assertEqual(self.detector.calls,1)

    def test_clipped_tracking_forces_semantic_rejection(self):
        self.seed();frame=self.next_frame()
        frame['color_bgr'][25:85,0:40]=180;frame['depth_m'][25:85,0:40]=.95
        frame['rays'][25:85,0:40]=frame['rays'][25:85,60:100]
        frame['color_bgr'][25:85,60:100]=0
        self.detector.boxes=[dict(xyxy=[0,25,40,85],confidence=.95,class_id=0,class_name='Coca-Cola')]
        with self.assertRaisesRegex(RuntimeError,'clipped'):self.select(frame)
        self.assertEqual(self.detector.calls,2)

    def test_acceptance_rejects_unchecked_frame_and_large_center_shift(self):
        self.seed();frame=self.next_frame()
        with self.assertRaisesRegex(RuntimeError,'pending'):self.accept(frame,{})
        _,vision=self.select(frame)
        with self.assertRaisesRegex(RuntimeError,'40mm'):
            self.tracker.accept(SimpleNamespace(center=np.array([.4,.18,.95])),vision,frame)

    def test_bad_tracked_cylinder_reacquires_same_frame_without_old_mask_fallback(self):
        from camera_input import geometry
        self.seed()
        for missing in (False,True):
            frame=self.next_frame()
            good=self.tracker.select
            def contaminated(*args,**kwargs):
                _,vision=good(*args,**kwargs)
                return np.ones(frame['depth_m'].shape,dtype=bool),vision
            # Spread background points so an all-image tracker proposal is
            # observably not a cylinder. The semantic box still encloses it.
            yy,xx=np.indices(frame['depth_m'].shape)
            outside=frame['color_bgr'][:,:,0]==0
            frame['rays'][outside,0]=(.1+xx*.004)[outside]
            frame['rays'][outside,1]=(.05+yy*.004)[outside]
            prior=self.tracker.state
            if missing:self.detector.boxes=[]
            with patch.object(self.tracker,'select',side_effect=contaminated):
                if missing:
                    with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):
                        geometry(frame,self.config(),require_table=False,tracker=self.tracker)
                    self.assertIsNone(self.tracker.pending)
                else:
                    observation,vision=geometry(frame,self.config(),require_table=False,tracker=self.tracker)
                    self.assertEqual(vision['tracking']['reason'],'cylinder_fit_uncertain')
                    self.assertEqual(vision['tracking']['semantic_frame_id'],frame['meta']['frame_id'])
                    np.testing.assert_allclose(observation.center[:2],[.35,.18],atol=1e-6)
            self.assertIs(self.tracker.state,prior)


if __name__=='__main__':unittest.main()
