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


class PlanningHandoffTests(unittest.TestCase):
    def test_recorded_handoff_selects_fresh_tracking_sample_instead_of_expired_first(self):
        r=Runtime.__new__(Runtime);r.lock=threading.RLock();r.tracking_error=None
        old=SimpleNamespace(timestamp=8558.93770647049,center=np.array([.31,.15,.823]))
        new=SimpleNamespace(timestamp=8559.20454788208,center=np.array([.311,.15,.823]))
        r.tracking_observation=new;r.tracking_vision={'frame_id':85184}
        with patch.dict('os.environ',{'HANDOFF_CAMERA_MAX_AGE_MS':'500'}),patch('runtime.time.monotonic',return_value=8559.505758736):
            self.assertGreater(8559.505758736-old.timestamp,.5)
            selected,vision=r.select_planning_observation()
            self.assertLess(8559.505758736-selected.timestamp,.5)
            self.assertEqual(vision['frame_id'],85184)
            self.assertEqual(selected.timestamp,new.timestamp)
        new.center[0]=.4
        self.assertEqual(r.tracking_baseline.center[0],.311)
        self.assertFalse(r.tracking_baseline.center.flags.writeable)

    def test_missing_stale_future_or_faulted_planning_samples_never_rebase(self):
        r=Runtime.__new__(Runtime);r.lock=threading.RLock();r.tracking_error=None
        baseline=object();r.tracking_baseline=baseline;r.tracking_vision={}
        with patch.dict('os.environ',{'HANDOFF_CAMERA_MAX_AGE_MS':'500'}),patch('runtime.time.monotonic',return_value=100.):
            for value in [None,SimpleNamespace(timestamp=99.49),SimpleNamespace(timestamp=100.01)]:
                r.tracking_observation=value
                with self.assertRaisesRegex(RuntimeError,'No fresh'):r.select_planning_observation()
                self.assertIs(r.tracking_baseline,baseline)
            r.tracking_error='static scene lost'
            with self.assertRaisesRegex(RuntimeError,'uncertain'):r.select_planning_observation()
            self.assertIs(r.tracking_baseline,baseline)
