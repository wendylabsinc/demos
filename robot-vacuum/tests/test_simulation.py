"""Mission and geometry checks against actual MuJoCo motion."""

from __future__ import annotations

import math
import unittest

import numpy as np

from robot_vacuum.planner import CoveragePlanner
from robot_vacuum.simulation import Simulator


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulator(seed=7)

    def test_route_enters_every_room_without_crossing_obstacles(self):
        planner = self.sim.planner
        current = tuple(self.sim.home["dock"][:2])
        for room in self.sim.home["rooms"]:
            x0, y0, x1, y1 = room["bounds"]
            cells = [cell for cell in planner.cells if x0 < planner.point(cell)[0] < x1 and y0 < planner.point(cell)[1] < y1]
            self.assertTrue(cells, room["name"])
            goal = planner.point(cells[len(cells) // 2])
            for target in planner.route(current, goal):
                samples = np.linspace(current, target, max(2, math.ceil(math.dist(current, target) / 0.02)))
                self.assertTrue(np.all(planner.points_clear(samples, planner.radius)), room["name"])
                current = target
            self.assertLess(math.dist(current, goal), 0.001)

    def test_dirt_only_on_reachable_unobstructed_floor(self):
        sim = self.sim
        self.assertEqual(len(sim.snapshot()["dust"]), 500)
        for particle in sim.snapshot()["dust"]:
            point = (particle["x"], particle["y"])
            self.assertTrue(sim.planner.is_clear(point, 0))
            center = sim.planner.point(sim.planner.nearest(point))
            self.assertLessEqual(math.dist(point, center), sim.planner.brush_radius - 0.034)

    def test_disconnected_space_excluded_from_cleaning(self):
        home = {
            "width": 4.0, "height": 3.0, "dock": [0.5, 0.5, 0.0],
            "robot": {"radius": 0.18, "brush_radius": 0.24, "max_speed": 0.65},
            "walls": [{"center": [2, 1.5], "size": [0.14, 3]}],
            "furniture": [], "rooms": [{"name": "Test", "bounds": [0, 0, 4, 3]}],
        }
        planner = CoveragePlanner(home)
        self.assertTrue(np.all(planner.positions[:, 0] < 2))
        self.assertTrue(np.all(planner.cleanable_points()[:, 0] < 2))

    def test_swept_brush_checks_entire_segment(self):
        points = np.asarray([[0.5, 0.2], [0.5, 0.3], [1.1, 0]])
        distances = Simulator._swept_distance_squared(points, (0, 0), (1, 0))
        np.testing.assert_allclose(distances, [0.04, 0.09, 0.01])


class SimulationTests(unittest.TestCase):
    def test_initial_pause_and_pause_resume_stop_motion(self):
        sim = Simulator()
        initial = sim.snapshot()
        self.assertEqual(initial["status"], "paused")
        sim.step(1)
        self.assertEqual(sim.snapshot()["robot"], initial["robot"])
        self.assertEqual(sim.snapshot()["time"], 0)
        sim.start()
        for _ in range(30):
            sim.step(0.05)
        self.assertGreater(sim.distance, 0.05)
        sim.pause()
        paused = sim.snapshot()
        sim.step(1)
        self.assertEqual(sim.snapshot()["robot"], paused["robot"])
        self.assertEqual(sim.snapshot()["time"], paused["time"])
        sim.start()
        self.assertEqual(sim.snapshot()["status"], "cleaning")
        sim.reset()
        self.assertEqual(sim.snapshot()["time"], 0)
        self.assertEqual(sim.snapshot()["cleaned"], 0)
        self.assertEqual(sim.snapshot()["robot"], initial["robot"])

    def test_manual_return_docks_without_claiming_complete_clean(self):
        sim = Simulator()
        sim.start()
        for _ in range(100):
            sim.step(0.1)
        sim.return_to_dock()
        for _ in range(1000):
            state = sim.step(0.1)
            if state["status"] == "docked":
                break
        self.assertEqual(state["status"], "docked")
        self.assertLess(state["cleaned"], state["total"])
        self.assertLess(math.dist(sim.position, sim.home["dock"][:2]), 0.03)

    def test_complete_physical_mission_cleans_and_docks(self):
        sim = Simulator(seed=7)
        sim.start()
        visited_rooms = set()
        previous_cleaned = 0
        for _ in range(5500):
            before = sim.position
            state = sim.step(0.2)
            # Actuators move the base continuously at its configured speed.
            self.assertLessEqual(math.dist(before, sim.position), sim.home["robot"]["max_speed"] * 0.2 + 0.003)
            self.assertTrue(sim.planner.is_clear(sim.position, sim.planner.radius - 0.006), str(state["robot"]))
            self.assertGreaterEqual(state["cleaned"], previous_cleaned)
            previous_cleaned = state["cleaned"]
            visited_rooms.add(state["room"])
            if state["status"] == "docked":
                break
        self.assertEqual(state["status"], "docked", f'time={state["time"]}, coverage={state["coverage"]}, target={state["target"]}')
        self.assertEqual(state["cleaned"], state["total"])
        self.assertEqual(state["dust"], [])
        self.assertGreaterEqual(state["coverage"], 99.5)
        self.assertGreater(state["distance"], 100)
        self.assertLess(state["time"], 1100)
        self.assertLess(math.dist(sim.position, sim.home["dock"][:2]), 0.03)
        self.assertLess(abs(state["robot"]["yaw"]), 0.04)
        self.assertEqual(visited_rooms, {room["name"] for room in sim.home["rooms"]})


if __name__ == "__main__":
    unittest.main()
