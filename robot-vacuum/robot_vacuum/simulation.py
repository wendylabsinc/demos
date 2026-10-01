"""A deterministic, physically stepped MuJoCo vacuum cleaning mission."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .planner import CoveragePlanner, Point
from .world import build_world


class Simulator:
    """Controller, physics, dirt collection, coverage and charging state.

    ``step`` advances simulated seconds. The robot starts paused and its pose
    changes only through velocity actuators and MuJoCo's contact solver. Dirt and
    coverage follow the swept brush footprint measured from that actual pose.
    """

    def __init__(self, home_path: str | Path | None = None, seed: int = 7):
        if home_path is None:
            from .paths import asset_dir
            home_path = asset_dir("assets") / "home.json"
        self.home = json.loads(Path(home_path).read_text())
        self.seed = seed
        self.planner = CoveragePlanner(self.home)
        self.floor_points = self.planner.cleanable_points()
        self._dust = self._make_dust(500)
        self._dust_positions = np.asarray([(item["x"], item["y"]) for item in self._dust])
        self.model = mujoco.MjModel.from_xml_string(build_world(self.home, self._dust))
        self.data = mujoco.MjData(self.model)
        self._dust_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'dust_{item["id"]}') for item in self._dust]
        self.reset()

    def _make_dust(self, count: int) -> list[dict]:
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self.floor_points), count, replace=len(self.floor_points) < count)
        particles = []
        reach = self.planner.brush_radius - 0.035
        for identifier, index in enumerate(indices):
            point = self.floor_points[index].copy()
            for _ in range(8):
                candidate = point + rng.uniform(-0.033, 0.033, 2)
                nearest = self.planner.point(self.planner.nearest(tuple(candidate)))
                if self.planner.is_clear(tuple(candidate), 0) and math.dist(candidate, nearest) <= reach:
                    point = candidate
                    break
            particles.append({"id": identifier, "x": float(point[0]), "y": float(point[1])})
        return particles

    @property
    def position(self) -> Point:
        return float(self.data.qpos[0]), float(self.data.qpos[1])

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.home["dock"]
        mujoco.mj_forward(self.model, self.data)
        self.status = "paused"
        self._resume_status = "cleaning"
        self.distance = 0.0
        self._remainder = 0.0
        self._waypoints: deque[Point] = deque()
        self._mission_started = False
        self._cleanup = False
        self._early_return = False
        self._covered = np.zeros(len(self.floor_points), dtype=bool)
        self._remaining = np.ones(len(self._dust), dtype=bool)
        self._trail = [list(self.position)]
        self.model.geom_rgba[self._dust_geom_ids, 3] = 1
        self._initial_waypoints = 1
        self._passed_waypoints = 0

    def start(self) -> None:
        if self.status == "docked" and not self._remaining.any():
            return
        if self.status in ("cleaning", "returning"):
            return
        if not self._mission_started or self.status == "docked":
            self._waypoints = deque(self.planner.coverage_route(self.position))
            self._initial_waypoints = max(1, len(self._waypoints))
            self._passed_waypoints = 0
            self._mission_started = True
            self._cleanup = False
            self._early_return = False
            self.status = "cleaning"
        else:
            self.status = self._resume_status

    def pause(self) -> None:
        if self.status in ("cleaning", "returning"):
            self._resume_status = self.status
            self.status = "paused"
            self.data.ctrl[:] = 0
            # Freezing the simulation also freezes its velocities.
            self.data.qvel[:] = 0

    def return_to_dock(self) -> None:
        self._early_return = True
        self._mission_started = True
        self._waypoints = deque(self.planner.route(self.position, tuple(self.home["dock"][:2])))
        self.status = "returning"
        self._resume_status = "returning"

    def _next_phase(self) -> None:
        if self.status == "returning":
            dock = tuple(self.home["dock"][:2])
            if math.dist(self.position, dock) > 0.025:
                self._waypoints = deque(self.planner.route(self.position, dock))
                return
            yaw_error = self._angle(float(self.home["dock"][2]) - float(self.data.qpos[2]))
            if abs(yaw_error) > 0.035:
                self.data.ctrl[:] = [0, 0, float(np.clip(5 * yaw_error, -3.2, 3.2))]
                return
            self.status = "docked"
            self.data.ctrl[:] = 0
            self.data.qvel[:] = 0
            return
        self._cleanup = True
        if self._remaining.any():
            candidates = self._dust_positions[self._remaining]
        elif np.mean(self._covered) < 0.995:
            candidates = self.floor_points[~self._covered]
        else:
            self.status = "returning"
            self._waypoints = deque(self.planner.route(self.position, tuple(self.home["dock"][:2])))
            return
        nearest_index = int(np.argmin(np.sum((candidates - self.position) ** 2, axis=1)))
        goal = self.planner.point(self.planner.nearest(tuple(candidates[nearest_index])))
        self._waypoints = deque(self.planner.route(self.position, goal))

    @staticmethod
    def _angle(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def _control(self) -> None:
        while self._waypoints:
            distance = math.dist(self.position, self._waypoints[0])
            tolerance = 0.019 if self._cleanup or len(self._waypoints) == 1 else 0.042
            if distance > tolerance:
                break
            self._waypoints.popleft()
            self._passed_waypoints += 1
        if not self._waypoints:
            self._next_phase()
            if not self._waypoints:
                return
        target_x, target_y = self._waypoints[0]
        x, y = self.position
        yaw = float(self.data.qpos[2])
        error = self._angle(math.atan2(target_y - y, target_x - x) - yaw)
        distance = math.hypot(target_x - x, target_y - y)
        turn = float(np.clip(5.0 * error, -3.2, 3.2))
        forward = min(self.home["robot"]["max_speed"], distance * 2.6)
        if abs(error) > 0.70:
            forward = 0.0
        else:
            forward *= max(0.0, math.cos(error)) ** 3
        self.data.ctrl[:] = [forward * math.cos(yaw), forward * math.sin(yaw), turn]

    @staticmethod
    def _swept_distance_squared(points: np.ndarray, start: Point, end: Point) -> np.ndarray:
        a, b = np.asarray(start), np.asarray(end)
        delta = b - a
        square_length = float(np.dot(delta, delta))
        if square_length < 1e-12:
            return np.sum((points - b) ** 2, axis=1)
        along = np.clip(((points - a) @ delta) / square_length, 0, 1)
        return np.sum((points - (a + along[:, None] * delta)) ** 2, axis=1)

    def _collect(self, start: Point, end: Point) -> None:
        radius_squared = self.planner.brush_radius ** 2
        self._covered |= self._swept_distance_squared(self.floor_points, start, end) <= radius_squared
        active = np.flatnonzero(self._remaining)
        if len(active):
            collected = active[self._swept_distance_squared(self._dust_positions[active], start, end) <= radius_squared]
            self._remaining[collected] = False
            if len(collected):
                self.model.geom_rgba[np.asarray(self._dust_geom_ids)[collected], 3] = 0

    def update_visuals(self) -> None:
        """Refresh visibility after a renderer is created or the mission resets."""
        self.model.geom_rgba[self._dust_geom_ids, 3] = self._remaining.astype(float)

    def step(self, dt: float = 0.05) -> dict:
        if not math.isfinite(dt) or dt <= 0 or dt > 5:
            raise ValueError("step duration must be finite and between 0 and 5 seconds")
        if self.status not in ("cleaning", "returning"):
            return self.snapshot()
        initial = self.position
        collection_start = initial
        collection_elapsed = 0.0
        self._remainder += dt
        timestep = float(self.model.opt.timestep)
        while self._remainder >= timestep - 1e-12 and self.status in ("cleaning", "returning"):
            previous = self.position
            self._control()
            if self.status == "docked":
                break
            mujoco.mj_step(self.model, self.data)
            self.distance += math.dist(previous, self.position)
            self._remainder -= timestep
            collection_elapsed += timestep
            if collection_elapsed >= 0.05 - 1e-12:
                self._collect(collection_start, self.position)
                collection_start = self.position
                collection_elapsed = 0.0
        if collection_elapsed > 0:
            self._collect(collection_start, self.position)
        if math.dist(self._trail[-1], self.position) > 0.075:
            self._trail.append([float(v) for v in self.position])
        return self.snapshot()

    def snapshot(self) -> dict:
        x, y = self.position
        yaw = self._angle(float(self.data.qpos[2]))
        room_name = "Hallway"
        for room in self.home["rooms"]:
            x0, y0, x1, y1 = room["bounds"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                room_name = room["name"]
                break
        cleaned = int(np.count_nonzero(~self._remaining))
        coverage = float(np.count_nonzero(self._covered) / len(self._covered) * 100)
        time = float(self.data.time)
        return {
            "status": self.status,
            "robot": {"x": x, "y": y, "yaw": yaw},
            "time": round(time, 3),
            "distance": round(self.distance, 3),
            "battery": round(max(0, 100 - self.distance * 0.033 - time / 1500), 1),
            "coverage": round(coverage, 2),
            "cleaned": cleaned,
            "total": len(self._dust),
            "dust": [particle.copy() for particle, remaining in zip(self._dust, self._remaining) if remaining],
            "trail": self._trail[-4000:],
            "room": room_name,
            "progress": round((cleaned / len(self._dust)) * 100, 2),
            "target": list(self._waypoints[0]) if self._waypoints else None,
        }
