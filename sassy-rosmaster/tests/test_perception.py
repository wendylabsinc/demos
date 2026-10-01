"""Real rendered pixels through the bundled neural model and measured depth."""
from dataclasses import replace
import math
import unittest

import numpy as np

from sassy_rosmaster.perception import CameraPerception, MuJoCoCameraSource, PersonDetector, ground_people
from sassy_rosmaster.simulation import Simulation


class RenderedVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PersonDetector()

    def setUp(self):
        self.simulation = Simulation(seed=7, population=0)
        self.camera = MuJoCoCameraSource(self.simulation)

    def tearDown(self):
        self.camera.close()

    def test_person_requires_actual_visible_pixels_and_registered_depth(self):
        empty = self.camera()
        self.assertEqual(self.detector.detect(empty.rgb), [])
        self.assertTrue(self.simulation.add_person(ahead=True))
        frame = self.camera()
        frame.validate()
        detections = self.detector.detect(frame.rgb)
        people = ground_people(frame, detections)
        self.assertEqual(len(people), 1, "Visible human meshes must reach actual camera inference")
        person = people[0]
        self.assertGreaterEqual(person["confidence"], self.detector.threshold)
        truth = self.simulation.scene()["people"][0]
        self.assertLess(math.hypot(person["x"]-truth["x"], person["y"]-truth["y"]), .4)
        self.assertLess(abs(person["bearing"]), .1)
        self.assertEqual(self.simulation.read().people, [], "Backend must not supply hidden person labels")
        missing_depth = replace(frame, depth=np.zeros_like(frame.depth))
        self.assertEqual(ground_people(missing_depth, detections), [])
        self.simulation.clear_people()
        self.assertEqual(self.detector.detect(self.camera().rgb), [])

    def test_person_behind_camera_is_not_detected_from_world_metadata(self):
        self.assertTrue(self.simulation.add_person(ahead=True))
        # Change the physical camera view, keeping the actor in the simulator.
        self.simulation.data.qpos[2] += math.pi
        import mujoco
        mujoco.mj_forward(self.simulation.model, self.simulation.data)
        frame = self.camera()
        self.assertEqual(len(self.simulation.scene()["people"]), 1)
        self.assertEqual(self.detector.detect(frame.rgb), [])


class TrackingTests(unittest.TestCase):
    def test_brief_missed_box_preserves_identity_without_inventing_detection(self):
        worker = CameraPerception(lambda: None)
        first = [dict(x=1., y=0., captured_at=10.)]
        worker._track(first, 10.)
        identity = first[0]['id']
        absent = []
        worker._track(absent, 10.2)
        self.assertEqual(absent, [])
        returned = [dict(x=1.1, y=0., captured_at=10.4)]
        worker._track(returned, 10.4)
        self.assertEqual(returned[0]['id'], identity)
        worker._track([], 13.)
        new = [dict(x=1.1, y=0., captured_at=13.1)]
        worker._track(new, 13.1)
        self.assertNotEqual(new[0]['id'], identity)


if __name__ == "__main__":
    unittest.main()
