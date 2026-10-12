"""Measured per-exposure immutable-anchor flow, recovery and unchanged fit bounds."""
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scene_reference import TableSceneReference


class SceneFlowTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict('os.environ', dict(HANDOFF_SCENE_REFERENCE_REGION='background',
                             HANDOFF_SCENE_FIT_RESIDUAL_MM='12', HANDOFF_SCENE_TRACKING='flow',
                             HANDOFF_VISUAL_THREADS='1'))
        self.env.start(); self.addCleanup(self.env.stop)
        rng = np.random.default_rng(615)
        noise = rng.integers(0, 256, (480, 640), dtype=np.uint8)
        self.gray = cv2.GaussianBlur(noise, (5, 5), .9)
        xx, yy = np.meshgrid(np.linspace(-.6, .6, 640), np.linspace(-.4, .4, 480))
        self.rays = np.stack([xx, yy, np.ones_like(xx)], axis=2)
        self.config = dict(camera_to_world=np.eye(4).tolist(), pelvis_to_world=np.eye(4).tolist(),
                           table_height_m=1.)
        self.frame = self.make_frame(self.gray, 10, 99.9)
        self.ref = TableSceneReference()
        with patch('time.monotonic', return_value=100.):
            self.ref.capture(self.frame, self.config, [3., 3., 1.])

    def make_frame(self, gray, fid, timestamp):
        return dict(color_bgr=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), rays=self.rays,
                    depth_m=np.ones_like(gray, dtype=float), timestamp=timestamp,
                    meta=dict(frame_id=fid, camera_serial='camera', stream_id='stream',
                              intrinsics=dict(fx=500., fy=500.)))

    def test_direct_anchor_flow_measures_new_translation_and_never_accumulates(self):
        original = copy.deepcopy(self.ref.anchor['world'])
        saved_gray = self.ref.anchor['gray'].copy()
        for shift, fid in [(2, 11), (-2, 12), (1, 13)]:
            image = cv2.warpAffine(self.gray, np.float32([[1, 0, shift], [0, 1, 0]]), (640, 480))
            frame = self.make_frame(image, fid, 99.95)
            with patch('time.monotonic', return_value=100.):
                corrected, diagnostic = self.ref.correct(frame, self.config)
            transform = np.asarray(diagnostic['transform'])
            self.assertEqual(diagnostic['metrics']['tracking_method'], 'immutable_anchor_flow')
            self.assertGreaterEqual(diagnostic['metrics']['inliers'], 30)
            np.testing.assert_allclose(transform[:3, 3], [-shift * 1.2 / 639, 0, 0], atol=.0003)
            np.testing.assert_allclose(corrected['camera_to_world'], corrected['pelvis_to_world'], atol=1e-12)
            self.assertEqual(diagnostic['reference_frame_id'], 10)
        np.testing.assert_array_equal(self.ref.anchor['gray'], saved_gray)
        self.assertFalse(self.ref.anchor['gray'].flags.writeable)
        self.assertFalse(self.ref.anchor['pixels'].flags.writeable)
        for pixel, value in original.items(): np.testing.assert_array_equal(self.ref.anchor['world'][pixel], value)

    def test_lost_flow_recovers_only_from_same_fresh_exposure_descriptors(self):
        frame = self.make_frame(self.gray, 11, 99.95)
        with patch('time.monotonic', return_value=100.), patch.object(
                self.ref, '_flow_correspondences', side_effect=RuntimeError('flow unavailable')):
            corrected, diagnostic = self.ref.correct(frame, self.config)
        self.assertEqual(diagnostic['metrics']['tracking_method'], 'sift')
        self.assertEqual(diagnostic['metrics']['flow_recovery_reason'], 'flow unavailable')
        self.assertEqual(diagnostic['frame_id'], 11)
        self.assertEqual(diagnostic['timestamp'], 99.95)
        np.testing.assert_allclose(corrected['camera_to_world'], np.eye(4), atol=1e-12)

    def test_no_background_does_not_reuse_last_transform(self):
        blank = self.make_frame(np.zeros_like(self.gray), 11, 99.95)
        with patch('time.monotonic', return_value=100.):
            with self.assertRaises((RuntimeError, ValueError)): self.ref.correct(blank, self.config)
        self.assertEqual(self.ref.anchor['metadata']['frame_id'], 10)

    def test_changed_identity_nonadvancing_or_stale_frames_fail_before_tracking(self):
        for key, value in [('camera_serial', 'other'), ('stream_id', 'other'), ('intrinsics', {})]:
            frame = self.make_frame(self.gray, 11, 99.95); frame['meta'][key] = value
            with self.subTest(key=key), patch.object(self.ref, '_flow_correspondences') as flow:
                with self.assertRaisesRegex(RuntimeError, 'changed'): self.ref.correct(frame, self.config)
                flow.assert_not_called()
        for fid, timestamp in [(10, 99.95), (11, 99.8), (11, 99.49), (11, 100.1)]:
            frame = self.make_frame(self.gray, fid, timestamp)
            with self.subTest(fid=fid, timestamp=timestamp), patch('time.monotonic', return_value=100.), patch.object(self.ref, '_flow_correspondences') as flow:
                with self.assertRaises((RuntimeError, ValueError)): self.ref.correct(frame, self.config)
                flow.assert_not_called()

    def test_flow_filters_hand_region_target_invalid_depth_and_reverse_mismatch(self):
        # Force correspondences to otherwise tempting excluded points. None
        # can contribute to the unchanged 30-landmark rigid fit.
        count = len(self.ref.anchor['pixels'])
        pixels = np.tile(np.float32([[[100, 400]]]), (count, 1, 1))
        back = self.ref.anchor['pixels'].copy()
        statuses = np.ones((count, 1), np.uint8); errors = np.zeros((count, 1), np.float32)
        for kind in ['hand', 'target', 'depth', 'reverse']:
            frame = self.make_frame(self.gray, 11, 99.95)
            points = pixels.copy(); returned = back.copy()
            if kind != 'hand': points[:] = [400, 200]
            if kind == 'target': self.ref.center = self.rays[200, 400, :2].copy()
            else: self.ref.center = np.array([3., 3.])
            if kind == 'depth': frame['depth_m'][200, 400] = np.nan
            if kind == 'reverse': returned += 2
            with self.subTest(kind=kind), patch.object(cv2, 'calcOpticalFlowPyrLK', side_effect=[
                    (points, statuses, errors), (returned, statuses, errors)]):
                source, reference = self.ref._flow_correspondences(frame, self.config)
                self.assertEqual(len(source), 0); self.assertEqual(len(reference), 0)

    def test_original_sift_rollback_and_invalid_mode(self):
        with patch.dict('os.environ', {'HANDOFF_SCENE_TRACKING': 'sift'}):
            self.assertEqual(TableSceneReference().tracking, 'sift')
        with patch.dict('os.environ', {'HANDOFF_SCENE_TRACKING': 'unchecked'}):
            with self.assertRaisesRegex(ValueError, 'sift or flow'): TableSceneReference()
