"""SIFT parallelism preserves scene features; worker avoids control CPUs."""
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from scene_reference import TableSceneReference

class VisualParallelTests(unittest.TestCase):
    def test_requested_two_threads_reaches_actual_scene_detector(self):
        with patch.dict('os.environ',{'HANDOFF_SCENE_REFERENCE_REGION':'background','HANDOFF_VISUAL_THREADS':'2'}):
            with patch('cv2.setNumThreads',wraps=cv2.setNumThreads) as configure:
                TableSceneReference()
                configure.assert_called_once_with(2)
        cv2.setNumThreads(1)

    def test_parallel_features_are_bitwise_identical_for_same_frame(self):
        rng=np.random.default_rng(1243);y,x=np.mgrid[:480,:640]
        frame=dict(timestamp=99.9,color_bgr=rng.integers(0,256,(480,640,3),dtype=np.uint8),depth_m=np.ones((480,640)),rays=np.stack([.5+x/1000,-.3+y/1000,np.ones_like(x)],axis=2))
        results=[]
        for count in ['1','2']:
            with patch.dict('os.environ',{'HANDOFF_SCENE_REFERENCE_REGION':'background','HANDOFF_VISUAL_THREADS':count}),patch('time.monotonic',return_value=100.):
                ref=TableSceneReference();ref.center=np.array([.3,.1]);w,k,d=ref._features(frame,dict(camera_to_world=np.eye(4).tolist(),table_height_m=.762));results.append((w,np.array([[*a.pt,a.size,a.angle,a.response,a.octave] for a in k]),d))
        self.assertEqual(list(results[0][0]),list(results[1][0]))
        np.testing.assert_array_equal(results[0][1],results[1][1]);np.testing.assert_array_equal(results[0][2],results[1][2])
        for key in results[0][0]:np.testing.assert_array_equal(results[0][0][key],results[1][0][key])
        cv2.setNumThreads(1)

class VisualWorkerAffinityTests(unittest.TestCase):
    def test_parallel_worker_reserves_motor_and_feedback_cpus(self):
        import os
        from perception_worker import configure_visual_worker
        with patch.dict(os.environ,{'HANDOFF_VISUAL_THREADS':'2','G1_MOTOR_CPU':'auto','G1_LOWSTATE_CPU':'auto'}),patch('os.sched_getaffinity',return_value=set(range(8)),create=True),patch('os.sched_setaffinity',create=True) as pin:
            result=configure_visual_worker()
            pin.assert_called_once_with(0,{4,5})
            self.assertTrue(result['active']);self.assertEqual(result['reserved_control_cpus'],[6,7])
            self.assertEqual(result['threads'],2)

    def test_small_cpu_pool_falls_back_without_touching_control_affinity(self):
        import os
        from perception_worker import configure_visual_worker
        with patch.dict(os.environ,{'HANDOFF_VISUAL_THREADS':'2','G1_MOTOR_CPU':'auto','G1_LOWSTATE_CPU':'auto'}),patch('os.sched_getaffinity',return_value={0,1},create=True),patch('os.sched_setaffinity',create=True) as pin:
            result=configure_visual_worker()
            pin.assert_not_called();self.assertEqual(result['threads'],1)
            self.assertEqual(os.environ['HANDOFF_VISUAL_THREADS'],'1')
            self.assertFalse(result['active']);self.assertTrue(result['error'])

    def test_custom_control_cpu_selections_are_excluded(self):
        import os
        from perception_worker import configure_visual_worker
        with patch.dict(os.environ,{'HANDOFF_VISUAL_THREADS':'2','G1_MOTOR_CPU':'2','G1_LOWSTATE_CPU':'5'}),patch('os.sched_getaffinity',return_value=set(range(8)),create=True),patch('os.sched_setaffinity',create=True) as pin:
            result=configure_visual_worker();pin.assert_called_once_with(0,{6,7})
            self.assertEqual(result['reserved_control_cpus'],[2,5])
