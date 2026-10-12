"""Broad independent scene support and finite forgiving fit allowance."""
import unittest
from unittest.mock import patch
import numpy as np
from scene_reference import estimate_scene_reference, scene_fit_residual_m, TableSceneReference

class BackgroundReferenceTests(unittest.TestCase):
    def test_forgiving_fit_admits_measured_scale_noise_with_original_pose_bounds(self):
        rng=np.random.default_rng(801)
        reference=np.c_[rng.uniform(.1,.8,160),rng.uniform(-.4,.4,160),rng.uniform(.76,1.1,160)]
        source=reference+rng.uniform(-.006,.006,reference.shape)
        with patch.dict('os.environ',{'HANDOFF_SCENE_FIT_RESIDUAL_MM':'12'}):
            transform,metrics=estimate_scene_reference(source,reference)
        self.assertGreaterEqual(metrics['inliers'],96)
        self.assertLessEqual(metrics['median_residual_mm'],6)
        self.assertLessEqual(metrics['p95_residual_mm'],12)
        self.assertLess(np.linalg.norm(transform[:3,3]),.02)
        with patch.dict('os.environ',{'HANDOFF_SCENE_FIT_RESIDUAL_MM':'6'}):
            with self.assertRaises(ValueError):estimate_scene_reference(source,reference)

    def test_invalid_unbounded_fit_allowances_fail(self):
        for value in ['0','-1','12.1','nan','inf','bad']:
            with self.subTest(value=value),patch.dict('os.environ',{'HANDOFF_SCENE_FIT_RESIDUAL_MM':value}):
                with self.assertRaises(ValueError):scene_fit_residual_m()

    def test_background_mask_cannot_select_target_hand_or_can_world_region(self):
        from types import SimpleNamespace
        ref=TableSceneReference.__new__(TableSceneReference);ref.region='background';ref.center=np.array([.3,.1])
        frame=dict(timestamp=99.9,depth_m=np.ones((480,640)),rays=np.zeros((480,640,3)),color_bgr=np.zeros((480,640,3),np.uint8))
        # Background objects above the table must remain eligible.
        points=[SimpleNamespace(pt=(float(300+i),float(100+i))) for i in range(32)]
        for i in range(32):frame['rays'][100+i,300+i]=[.6,-.2,1.1]
        frame['rays'][100,300]=[.3,.1,.823]
        def detect(image,mask):
            self.assertTrue(np.all(mask[255:,:260]==0));self.assertEqual(mask[100,300],255)
            return points,np.zeros((32,128),np.float32)
        ref.detector=SimpleNamespace(detectAndCompute=detect)
        with patch('time.monotonic',return_value=100.):
            world,keys,descriptors=ref._features(frame,dict(table_height_m=.762,camera_to_world=np.eye(4).tolist()))
        self.assertEqual(len(keys),31);self.assertNotIn((100,300),world)
        self.assertTrue(all(p[2]>.762+.01 for p in world.values()))
