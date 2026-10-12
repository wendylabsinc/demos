"""Exact-frame publication and post-grasp scene pose guards; no hardware."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import numpy as np
from runtime import Runtime

class ScenePublicationTests(unittest.TestCase):
    def setUp(self):
        self.runtime=Runtime.__new__(Runtime)
        r=self.runtime;r.lock=threading.RLock();r.camera=SimpleNamespace(serial='camera');r.scene_pose_data=None
        r.tracking_error=None;r.tracking_baseline=None;r.post_grasp_perception_disabled=False
        self.scene=dict(source='independent_static_table_landmarks',transform=np.eye(4).tolist(),timestamp=99.9,frame_id=11,camera_serial='camera',stream_id='stream')
        self.override=patch.dict('os.environ',{'HANDOFF_SCENE_REFERENCE':'1','HANDOFF_CAMERA_MAX_AGE_MS':'500'})
        self.override.start();self.addCleanup(self.override.stop)

    def publish(self,scene):
        with patch('runtime.time.monotonic',return_value=100.):
            self.runtime.publish_tracking_observation(SimpleNamespace(timestamp=scene['timestamp']),dict(frame_id=scene['frame_id'],scene_pose=scene),True)

    def test_planning_and_postgrasp_pose_only_publish_without_can_position(self):
        self.publish(self.scene);self.runtime.scene_pose()
        self.runtime.post_grasp_perception_disabled=True
        self.scene['timestamp']=99.95;self.scene['frame_id']=12;self.publish(self.scene)
        self.assertEqual(self.runtime.scene_pose()[1],99.95)
        self.runtime.tracking_error='Scene registration lost'
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.runtime.scene_pose()

    def test_wrong_identity_stale_and_nonadvancing_pose_never_publish(self):
        for key,value in [('source','target'),('camera_serial','other'),('stream_id',None),('timestamp',99.49)]:
            with self.subTest(key=key):
                bad=dict(self.scene);bad[key]=value
                with self.assertRaisesRegex(RuntimeError,'identity/timestamp'):self.publish(bad)
                self.assertIsNone(self.runtime.scene_pose_data)
        with patch('runtime.time.monotonic',return_value=100.):
            with self.assertRaisesRegex(RuntimeError,'identity/timestamp'):
                self.runtime.publish_tracking_observation(SimpleNamespace(timestamp=99.9),dict(frame_id=12,scene_pose=self.scene),True)
        self.publish(self.scene)
        with self.assertRaisesRegex(RuntimeError,'did not advance'):self.publish(self.scene)

    def test_postgrasp_keeps_only_scene_pose_worker_when_configured(self):
        r=self.runtime;r.tracking_stop=threading.Event();r.event=Mock();plant=SimpleNamespace(perception_guard=Mock())
        r.finish_grasp_perception(plant)
        self.assertTrue(r.post_grasp_perception_disabled)
        self.assertFalse(r.tracking_stop.is_set());self.assertIsNone(plant.perception_guard)
