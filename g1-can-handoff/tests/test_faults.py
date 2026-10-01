import time
import unittest
import numpy as np
from g1_can_handoff.perception import Observation,detect
from g1_can_handoff.task import plan,accept

class NeverMove:
    def __init__(self):self.calls=0
    def solve(self,*args):self.calls+=1;raise AssertionError("Invalid observations must be rejected before IK")
    def hand(self,*args):raise AssertionError("Unconfirmed handoff must not open the hand")

class FaultTests(unittest.TestCase):
    def test_expired_detection_cannot_plan_motion(self):
        p=NeverMove()
        with self.assertRaisesRegex(ValueError,"expired"):
            plan(p,Observation(np.array([.35,.18,.811]),.75,time.monotonic()-2,0,500))
        self.assertEqual(p.calls,0)

    def test_nonfinite_pose_cannot_plan_motion(self):
        with self.assertRaisesRegex(ValueError,"Invalid"):
            plan(NeverMove(),Observation(np.array([np.nan,.18,.811]),.75,time.monotonic(),0,500))

    def test_unreachable_area_cannot_plan_motion(self):
        with self.assertRaisesRegex(ValueError,"outside"):
            plan(NeverMove(),Observation(np.array([1.2,.18,.811]),.75,time.monotonic(),0,500))

    def test_release_requires_recipient_support(self):
        with self.assertRaisesRegex(ValueError,"support"):accept(NeverMove())

    def test_blank_camera_image_does_not_detect_a_can(self):
        with self.assertRaisesRegex(ValueError,"visible"):
            detect(np.zeros((100,100,3)),np.ones((100,100)),np.array([[100,0,50],[0,100,50],[0,0,1]]),np.eye(4))

    def test_stale_camera_capture_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"stale"):
            detect(np.zeros((100,100,3)),np.ones((100,100)),np.eye(3),np.eye(4),time.monotonic()-1)

    def test_invalid_calibration_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"calibration"):
            detect(np.zeros((100,100,3)),np.ones((100,100)),np.zeros((3,3)),np.eye(4))

if __name__=="__main__":unittest.main()
