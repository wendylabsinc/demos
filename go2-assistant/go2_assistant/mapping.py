"""Bounded lidar mapping in the robot's initial coordinate frame.

This module has no scene, room, world-pose, or simulator interfaces. A scan is
an array of metres and matching angles in radians, measured counterclockwise
from the robot's forward axis. ``odometry_delta`` is an incremental body-frame
(forward, left, yaw) displacement since the preceding scan. ``lidar_origin``
is the measured planar sensor mount relative to the base. A max-range return
marks free space along the ray, never an occupied endpoint.

Local correlative scan matching corrects odometry against previous occupied
returns. This is bounded local SLAM; it does not provide loop closure or global
relocalization. Loading a map requires three consecutive, consistent, uniquely
matching scans near its saved pose, initial origin, or one of at most eight
recorded poses before navigation or map updates are enabled. Unknown cells and
the complete circular robot
footprint are excluded from routes. Grid bounds limit memory, not room geometry.
"""
from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def voxel_scan(points, *, sectors=72, min_height=-.2, max_height=.4, max_range=12.):
    """Project an accumulated body-frame voxel map into conservative sectors.

    Each finite sector contains the nearest occupied voxel in the robot-height
    band. A missing sector remains NaN (unknown); it is never converted to a
    synthetic maximum-range return.
    """
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not 8 <= len(points) <= 200_000
            or not np.isfinite(points).all() or not isinstance(sectors, int) or not 36 <= sectors <= 720
            or not all(math.isfinite(v) for v in (min_height, max_height, max_range))
            or not min_height < max_height or not .1 <= max_range <= 30.):
        raise ValueError("Invalid bounded voxel map or projection settings.")
    radii = np.hypot(points[:, 0], points[:, 1])
    usable = ((points[:, 2] >= min_height) & (points[:, 2] <= max_height)
              & (radii >= .05) & (radii <= max_range))
    if np.count_nonzero(usable) < 8:
        raise ValueError("Voxel map has too few robot-height obstacle returns.")
    angles = np.arctan2(points[usable, 1], points[usable, 0])
    distances = radii[usable]
    increment = 2 * math.pi / sectors
    bins = np.floor((angles + math.pi) / increment + .5).astype(int) % sectors
    ranges = np.full(sectors, np.inf, dtype=np.float64)
    np.minimum.at(ranges, bins, distances)
    ranges[~np.isfinite(ranges)] = np.nan
    scan_angles = -math.pi + np.arange(sectors) * increment
    return ranges, scan_angles


class OccupancyMapper:
    def __init__(self, resolution=.1, robot_radius=.4, half_extent_m=12., map_id=None):
        if not (.05 <= resolution <= .5 and 0.05 <= robot_radius <= 1.0
                and 2. <= half_extent_m <= 25.):
            raise ValueError("Invalid bounded map dimensions or robot footprint.")
        self.resolution = float(resolution)
        self.robot_radius = float(robot_radius)
        self.half_extent_m = math.ceil(half_extent_m / resolution) * resolution
        self.size = int(round(2 * self.half_extent_m / resolution)) + 1
        if self.size > 601:
            raise ValueError("Map may have at most 601 cells per side.")
        self.map_id = str(map_id or ("lidar-" + uuid.uuid4().hex))
        if not self.map_id or len(self.map_id) > 128:
            raise ValueError("A bounded map identity is required.")
        self.log_odds = np.zeros((self.size, self.size), dtype=np.float32)
        self.observed = np.zeros((self.size, self.size), dtype=bool)
        self._pose = np.zeros(3, dtype=np.float64)
        self.lock = threading.RLock()
        self.revision = self.scans = 0
        self.timestamp = None
        self.localization = {"valid": False, "method": "uninitialized", "score": 0.}
        self._loaded = False
        self._reload_pose = None
        self._reload_count = 0
        self._reload_timestamp = None
        self._anchors = [self._pose.copy()]
        self._walk_cache = None
        self._score_cache = None

    def pose(self):
        with self.lock:
            return dict(zip(("x", "y", "yaw"), map(float, self._pose)))

    def _cell(self, x, y):
        return (int(round((x + self.half_extent_m) / self.resolution)),
                int(round((y + self.half_extent_m) / self.resolution)))

    def _xy(self, cell):
        return (cell[0] * self.resolution - self.half_extent_m,
                cell[1] * self.resolution - self.half_extent_m)

    def _inside(self, cell):
        return 0 <= cell[0] < self.size and 0 <= cell[1] < self.size

    @staticmethod
    def _shift(array, dx, dy, fill=0):
        result = np.full_like(array, fill)
        n = array.shape[0]
        if abs(dx) >= n or abs(dy) >= n:
            return result
        src_x = slice(max(0, -dx), min(n, n - dx))
        src_y = slice(max(0, -dy), min(n, n - dy))
        dst_x = slice(max(0, dx), min(n, n + dx))
        dst_y = slice(max(0, dy), min(n, n + dy))
        result[dst_y, dst_x] = array[src_y, src_x]
        return result

    def _score_field(self):
        if self._score_cache is None:
            occupied = self.observed & (self.log_odds >= .65)
            field = occupied.astype(np.float32)
            # A short likelihood field tolerates cell quantization and noise.
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    distance = math.hypot(dx, dy)
                    if 0 < distance <= 3:
                        weight = math.exp(-.5 * (distance / 1.2) ** 2)
                        np.maximum(field, self._shift(occupied, dx, dy) * weight, out=field)
            self._score_cache = field
        return self._score_cache

    def _match(self, predicted, points, *, trim=True):
        occupied_count = int(np.count_nonzero(self.observed & (self.log_odds >= .65)))
        if occupied_count < 12 or len(points) < 12:
            return predicted, 0., False
        field = self._score_field()
        points = points[::max(1, math.ceil(len(points) / 180))]

        def score(pose):
            c, s = math.cos(pose[2]), math.sin(pose[2])
            x = pose[0] + points[:, 0] * c - points[:, 1] * s
            y = pose[1] + points[:, 0] * s + points[:, 1] * c
            ix = np.rint((x + self.half_extent_m) / self.resolution).astype(int)
            iy = np.rint((y + self.half_extent_m) / self.resolution).astype(int)
            valid = (ix >= 0) & (iy >= 0) & (ix < self.size) & (iy < self.size)
            values = np.zeros(len(points), dtype=np.float32)
            values[valid] = field[iy[valid], ix[valid]]
            # New geometry has no match; cap its effect with the best 75%.
            count = max(12, int(len(values) * .75)) if trim else len(values)
            return float(np.mean(np.partition(values, -count)[-count:]))

        best = predicted.copy()
        best_score = score(best)
        def objective(pose, raw):
            return raw - .04 * math.hypot(pose[0] - predicted[0], pose[1] - predicted[1]) \
                - .02 * abs(_angle(pose[2] - predicted[2]))
        best_objective = objective(best, best_score)
        # Search a finite neighbourhood, then refine. No unbounded optimizer.
        for translation_step, yaw_step, radius in ((.12, .06, 2), (.04, .02, 1), (.015, .008, 1)):
            center = best.copy()
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for yaw in range(-radius, radius + 1):
                        candidate = center + (dx * translation_step, dy * translation_step, yaw * yaw_step)
                        raw = score(candidate)
                        value = objective(candidate, raw)
                        if value > best_objective + 1e-7:
                            best, best_score, best_objective = candidate, raw, value
        best[2] = _angle(best[2])
        return best, best_score, True

    def update_scan(self, ranges, angles, odometry_delta=(0., 0., 0.), *, max_range=8.,
                    timestamp=None, lidar_origin=(0., 0.)):
        """Estimate base pose, then integrate this measured scan. No absolute pose input."""
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        delta = np.asarray(odometry_delta, dtype=np.float64)
        origin = np.asarray(lidar_origin, dtype=np.float64)
        if (ranges.ndim != 1 or angles.shape != ranges.shape or not 8 <= ranges.size <= 2048
                or not np.isfinite(angles).all() or delta.shape != (3,)
                or not np.isfinite(delta).all() or origin.shape != (2,) or not np.isfinite(origin).all()
                or np.linalg.norm(origin) > 1. or not .1 <= max_range <= 30.
                or np.linalg.norm(delta[:2]) > 1.5 or abs(delta[2]) > math.pi):
            raise ValueError("Invalid lidar scan, measured mount, or incremental odometry.")
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(stamp):
            raise ValueError("Scan timestamp must be finite.")
        valid = (~np.isnan(ranges)) & (ranges >= .05)
        if np.count_nonzero(valid) < 8:
            raise ValueError("Lidar scan has too few usable returns.")
        distances = np.minimum(ranges[valid], max_range)
        scan_angles = angles[valid]
        hits = ranges[valid] < max_range - .02
        local = np.column_stack((distances * np.cos(scan_angles), distances * np.sin(scan_angles))) + origin
        with self.lock:
            if self.timestamp is not None and stamp <= self.timestamp:
                raise ValueError("Lidar scans must have increasing capture timestamps.")
            predicted = self._pose.copy()
            c, s = math.cos(predicted[2]), math.sin(predicted[2])
            predicted[0] += delta[0] * c - delta[1] * s
            predicted[1] += delta[0] * s + delta[1] * c
            predicted[2] = _angle(predicted[2] + delta[2])
            # All measured endpoints constrain tracking. Discarding a fixed
            # fraction lets biased odometry paint ghost walls and then match
            # those same walls perfectly on subsequent stationary scans.
            matched, quality, attempted = self._match(predicted, local[hits], trim=False)
            ambiguous = False
            if self._loaded:
                candidates = [(quality, matched)] if attempted else []
                for anchor in self._anchors:
                    estimate, score, usable = self._match(np.asarray(anchor), local[hits], trim=False)
                    if usable:
                        candidates.append((score, estimate))
                candidates.sort(key=lambda candidate: candidate[0], reverse=True)
                # Cluster after ranking, so a weaker saved-pose candidate can
                # never discard a stronger measured match near another anchor.
                distinct = []
                for candidate in candidates:
                    if all(np.linalg.norm(candidate[1][:2] - other[1][:2]) > .1
                           or abs(_angle(candidate[1][2] - other[1][2])) > .05 for other in distinct):
                        distinct.append(candidate)
                candidates = distinct
                if candidates:
                    quality, matched = candidates[0]
                    attempted = True
                    ambiguous = len(candidates) > 1 and candidates[1][0] >= quality - .08
                else:
                    attempted = False
            if (attempted and quality < .35) or (self._loaded and (not attempted or quality < .65 or ambiguous)):
                self._reload_pose = self._reload_timestamp = None
                self._reload_count = 0
                self.localization = {"valid": False, "method": "ambiguous_relocalization" if ambiguous else "scan_mismatch", "score": quality}
                self.timestamp = stamp
                # Preserve the last map and pose; a bad scan cannot repaint it.
                return {"pose": self.pose(), "localization": dict(self.localization),
                        "map_id": self.map_id, "timestamp": stamp}
            if self._loaded:
                consistent = (self._reload_pose is not None
                              and 0 < stamp - self._reload_timestamp <= 1.
                              and np.linalg.norm(matched[:2] - self._reload_pose[:2]) <= .1
                              and abs(_angle(matched[2] - self._reload_pose[2])) <= .05)
                self._reload_count = self._reload_count + 1 if consistent else 1
                self._reload_pose, self._reload_timestamp = matched.copy(), stamp
                if self._reload_count < 3:
                    self.localization = {"valid": False, "method": "confirming_relocalization",
                                         "score": quality, "confirmed_scans": self._reload_count,
                                         "required_scans": 3}
                    self.timestamp = stamp
                    # Confirmation is read-only. Uncertain poses must not paint
                    # new walls into the map used to validate the next scan.
                    return {"pose": self.pose(), "localization": dict(self.localization),
                            "map_id": self.map_id, "timestamp": stamp}
            self._pose = matched if attempted else predicted
            if not self._inside(self._cell(*self._pose[:2])):
                self.localization = {"valid": False, "method": "map_bounds", "score": quality}
                self.timestamp = stamp
                return {"pose": self.pose(), "localization": dict(self.localization),
                        "map_id": self.map_id, "timestamp": stamp}
            self._loaded = False
            self._reload_pose = self._reload_timestamp = None
            self._reload_count = 0
            self.localization = {"valid": True,
                                 "method": "scan_match" if attempted else "incremental_odometry",
                                 "score": quality, "correction_m": float(np.linalg.norm(self._pose[:2] - predicted[:2]))}
            self._integrate(local, hits, origin)
            self.timestamp = stamp
            self.scans += 1
            self.revision += 1
            if len(self._anchors) < 8 and all(np.linalg.norm(self._pose[:2] - anchor[:2]) > 2.
                                             for anchor in self._anchors):
                self._anchors.append(self._pose.copy())
            self._walk_cache = None
            return {"pose": self.pose(), "localization": dict(self.localization),
                    "map_id": self.map_id, "timestamp": stamp}

    def update_voxels(self, points, pose, *, timestamp=None, min_height=-.2,
                      max_height=.4, max_range=12., sectors=72):
        """Integrate an accumulated occupied-voxel map paired with odometry.

        Occupied cells come directly from every body-height voxel. Known-free
        evidence is limited to the line from the robot to the nearest occupied
        voxel in each populated angular sector. Empty sectors remain unknown.
        Unlike ``update_scan``, this path performs no scan matching: the bridge
        has already expressed the map in the odometry frame.
        """
        points = np.asarray(points, dtype=np.float64)
        pose_values = np.asarray(pose, dtype=np.float64)
        ranges, angles = voxel_scan(points, sectors=sectors, min_height=min_height,
                                    max_height=max_height, max_range=max_range)
        stamp = time.monotonic() if timestamp is None else float(timestamp)
        if pose_values.shape != (3,) or not np.isfinite(pose_values).all() or not math.isfinite(stamp):
            raise ValueError("Voxel odometry pose and capture timestamp must be finite.")
        usable_points = ((points[:, 2] >= min_height) & (points[:, 2] <= max_height)
                         & (np.hypot(points[:, 0], points[:, 1]) >= .05)
                         & (np.hypot(points[:, 0], points[:, 1]) <= max_range))
        local_occupied = points[usable_points, :2]
        populated = np.isfinite(ranges)
        local_nearest = np.column_stack((ranges[populated] * np.cos(angles[populated]),
                                         ranges[populated] * np.sin(angles[populated])))
        with self.lock:
            if self.timestamp is not None and stamp <= self.timestamp:
                raise ValueError("Voxel maps must have increasing capture timestamps.")
            if not self._inside(self._cell(*pose_values[:2])):
                self.localization = {"valid": False, "method": "map_bounds", "score": 0.}
                self.timestamp = stamp
                return {"pose": self.pose(), "localization": dict(self.localization),
                        "map_id": self.map_id, "timestamp": stamp, "ranges": ranges}
            self._pose = pose_values.copy()
            self._pose[2] = _angle(self._pose[2])
            self.localization = {"valid": True, "method": "voxel_map_odometry",
                                 "score": 1., "observed_sectors": int(np.count_nonzero(populated)),
                                 "total_sectors": sectors}
            self._integrate(local_nearest, np.ones(len(local_nearest), dtype=bool), np.zeros(2))
            c, s = math.cos(self._pose[2]), math.sin(self._pose[2])
            world = local_occupied @ np.array(((c, s), (-s, c))) + self._pose[:2]
            cells = np.rint((world + self.half_extent_m) / self.resolution).astype(int)
            inside = ((cells[:, 0] >= 0) & (cells[:, 1] >= 0)
                      & (cells[:, 0] < self.size) & (cells[:, 1] < self.size))
            cells = cells[inside]
            if len(cells):
                self.observed[cells[:, 1], cells[:, 0]] = True
                self.log_odds[cells[:, 1], cells[:, 0]] = np.maximum(
                    .95, self.log_odds[cells[:, 1], cells[:, 0]])
            self.timestamp = stamp
            self.scans += 1
            self.revision += 1
            self._loaded = False
            self._walk_cache = self._score_cache = None
            return {"pose": self.pose(), "localization": dict(self.localization),
                    "map_id": self.map_id, "timestamp": stamp, "ranges": ranges}

    def _integrate(self, local, hits, origin):
        c, s = math.cos(self._pose[2]), math.sin(self._pose[2])
        endpoints = local @ np.array(((c, s), (-s, c))) + self._pose[:2]
        start = self._cell(self._pose[0] + origin[0] * c - origin[1] * s,
                           self._pose[1] + origin[0] * s + origin[1] * c)
        old_occupied = (self.observed & (self.log_odds >= .65)) if self._score_cache is not None else None
        free = np.zeros_like(self.observed)
        occupied = np.zeros_like(self.observed)
        ends = np.rint((endpoints + self.half_extent_m) / self.resolution).astype(int)
        delta = ends - np.asarray(start)
        steps = np.maximum(np.max(np.abs(delta), axis=1), 1)
        sample = np.arange(int(np.max(steps)) + 1)
        # np.linspace computes step first, then multiplies each sample. Keep
        # that operation order so half-cell rounding matches the scalar walk.
        x = np.rint(start[0] + (delta[:, 0] / steps)[:, None] * sample[None, :]).astype(int)
        y = np.rint(start[1] + (delta[:, 1] / steps)[:, None] * sample[None, :]).astype(int)
        in_bounds = (x >= 0) & (y >= 0) & (x < self.size) & (y < self.size)
        endpoint_inside = ((ends[:, 0] >= 0) & (ends[:, 1] >= 0)
                           & (ends[:, 0] < self.size) & (ends[:, 1] < self.size))
        occupied_hits = hits & endpoint_inside
        occupied[ends[occupied_hits, 1], ends[occupied_hits, 0]] = True
        # Exactly the cell-sized samples of the former per-ray walk. Padding
        # past a short ray and occupied endpoints never produce free evidence.
        usable = in_bounds & (sample[None, :] <= steps[:, None])
        usable &= ~(occupied_hits[:, None] & (sample[None, :] == steps[:, None]))
        free[y[usable], x[usable]] = True
        free &= ~occupied
        self.log_odds[free] = np.maximum(-4., self.log_odds[free] - .6)
        # A new obstacle closes a formerly clear route on its first return.
        self.log_odds[occupied] = np.minimum(4., np.maximum(0., self.log_odds[occupied]) + .95)
        self.observed |= free | occupied
        if old_occupied is not None and not np.array_equal(old_occupied, self.observed & (self.log_odds >= .65)):
            self._score_cache = None

    def _walkable(self):
        if self._walk_cache is None:
            free = self.observed & (self.log_odds < -.2)
            safe = free.copy()
            radius = self.robot_radius + self.resolution / math.sqrt(2.)
            cells = math.ceil(radius / self.resolution)
            for dx in range(-cells, cells + 1):
                for dy in range(-cells, cells + 1):
                    if math.hypot(dx, dy) * self.resolution <= radius:
                        safe &= self._shift(free, dx, dy, False)
            self._walk_cache = safe
        return self._walk_cache

    def free(self, x, y):
        with self.lock:
            cell = self._cell(x, y)
            return bool(self.localization["valid"] and self._inside(cell) and self._walkable()[cell[1], cell[0]])

    def line_free(self, first, second):
        with self.lock:
            count = max(1, math.ceil(math.dist(first, second) / (self.resolution * .4)))
            return all(self.free(first[0] + (second[0] - first[0]) * i / count,
                                 first[1] + (second[1] - first[1]) * i / count)
                       for i in range(count + 1))

    def route(self, start, goal):
        """Return an A* route entirely inside observed, footprint-safe free cells."""
        with self.lock:
            if not self.free(*start) or not self.free(*goal):
                return []
            initial, target = self._cell(*start), self._cell(*goal)
            safe = self._walkable()
            queue = [(math.dist(initial, target), 0., initial)]
            previous, costs = {}, {initial: 0.}
            while queue:
                _, cost, cell = heapq.heappop(queue)
                if cost > costs[cell]:
                    continue
                if cell == target:
                    cells = [cell]
                    while cell != initial:
                        cell = previous[cell]
                        cells.append(cell)
                    points = [tuple(start)] + [self._xy(c) for c in cells[-2::-1]] + [tuple(goal)]
                    # Smooth only with the same inflated, observed-space check.
                    path, index = [points[0]], 0
                    while index < len(points) - 1:
                        following = len(points) - 1
                        while following > index + 1 and not self.line_free(points[index], points[following]):
                            following -= 1
                        path.append(points[following]); index = following
                    return path
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    neighbour = (cell[0] + dx, cell[1] + dy)
                    if not self._inside(neighbour) or not safe[neighbour[1], neighbour[0]]:
                        continue
                    if dx and dy and not (safe[cell[1], neighbour[0]] and safe[neighbour[1], cell[0]]):
                        continue
                    new_cost = cost + math.hypot(dx, dy)
                    if new_cost < costs.get(neighbour, math.inf):
                        costs[neighbour], previous[neighbour] = new_cost, cell
                        heapq.heappush(queue, (new_cost + math.dist(neighbour, target), new_cost, neighbour))
            return []

    def plan(self, goal_x, goal_y):
        return self.route(tuple(self._pose[:2]), (goal_x, goal_y))

    def frontiers(self, limit=20):
        """Reachable safe viewpoints just inside the observed/unknown boundary."""
        if not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("Frontier count must be between one and fifty.")
        with self.lock:
            start = self._cell(*self._pose[:2])
            safe = self._walkable()
            if not self.localization["valid"] or not self._inside(start) or not safe[start[1], start[0]]:
                return []
            unknown = ~self.observed
            boundary = np.zeros_like(unknown)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                boundary |= self._shift(unknown, dx, dy)
            # Only free/unknown interfaces are frontiers. Unknown behind a
            # measured wall must not turn the wall into an exploration goal.
            boundary &= self.observed & (self.log_odds < -.2)
            nearby_unknown = np.zeros_like(unknown)
            radius = math.ceil(self.robot_radius / self.resolution) + 3
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if math.hypot(dx, dy) <= radius:
                        nearby_unknown |= self._shift(boundary, dx, dy)
            reachable, queue = {start}, [start]
            candidates = []
            for cell in queue:
                if nearby_unknown[cell[1], cell[0]] and math.dist(cell, start) * self.resolution > .6:
                    candidates.append(cell)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_cell = (cell[0] + dx, cell[1] + dy)
                    if (next_cell not in reachable and self._inside(next_cell)
                            and safe[next_cell[1], next_cell[0]]):
                        reachable.add(next_cell); queue.append(next_cell)
            candidates.sort(key=lambda cell: math.dist(cell, start))
            selected = []
            for cell in candidates:
                point = self._xy(cell)
                if all(math.dist(point, other) >= .7 for other in selected):
                    selected.append(point)
                    if len(selected) == limit:
                        break
            return selected

    def status(self):
        with self.lock:
            return {"map_id": self.map_id,
                    "source": "accumulated occupied voxels" if self.localization.get("method") == "voxel_map_odometry" else "measured lidar rays",
                    "resolution_m": self.resolution, "size_cells": self.size,
                    "robot_radius_m": self.robot_radius, "revision": self.revision,
                    "scans": self.scans, "observed_cells": int(np.count_nonzero(self.observed)),
                    "occupied_cells": int(np.count_nonzero(self.observed & (self.log_odds >= .65))),
                    "pose": self.pose(), "localization": dict(self.localization), "timestamp": self.timestamp}

    def snapshot(self):
        """Bounded JSON-safe grid for visualization; -1 unknown, 0 free, 100 blocked."""
        with self.lock:
            grid = np.full(self.log_odds.shape, -1, dtype=np.int8)
            grid[self.observed & (self.log_odds < -.2)] = 0
            grid[self.observed & (self.log_odds >= -.2)] = 100
            return {**self.status(), "origin": [-self.half_extent_m, -self.half_extent_m],
                    "cells": grid.tolist()}

    def save(self, path):
        with self.lock:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            metadata = {"version": 1, "map_id": self.map_id, "resolution": self.resolution,
                        "robot_radius": self.robot_radius, "half_extent_m": self.half_extent_m,
                        "pose": self._pose.tolist(), "anchors": [anchor.tolist() for anchor in self._anchors],
                        "revision": self.revision, "scans": self.scans}
            with destination.open("wb") as output:
                np.savez_compressed(output, metadata=json.dumps(metadata),
                                    log_odds=self.log_odds, observed=self.observed)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Saved map exceeds the bounded file size.")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("version") != 1:
                raise ValueError("Unsupported saved map version.")
            mapper = cls(metadata["resolution"], metadata["robot_radius"], metadata["half_extent_m"], metadata["map_id"])
            grid, observed = data["log_odds"], data["observed"]
            pose = np.asarray(metadata["pose"], dtype=float)
            if (grid.shape != mapper.log_odds.shape or observed.shape != grid.shape
                    or not np.isfinite(grid).all() or np.max(np.abs(grid)) > 4.01
                    or observed.dtype != np.bool_ or pose.shape != (3,) or not np.isfinite(pose).all()
                    or not mapper._inside(mapper._cell(*pose[:2]))):
                raise ValueError("Saved map contains invalid measurements or pose.")
            mapper.log_odds, mapper.observed, mapper._pose = grid.astype(np.float32), observed.copy(), pose
            anchors = np.asarray(metadata.get("anchors", [[0., 0., 0.]]), dtype=float)
            if anchors.ndim != 2 or anchors.shape[1] != 3 or not 1 <= len(anchors) <= 8 or not np.isfinite(anchors).all():
                raise ValueError("Saved map has invalid bounded localization anchors.")
            mapper._anchors = list(anchors)
            mapper.revision, mapper.scans = int(metadata["revision"]), int(metadata["scans"])
            mapper._loaded = True
            mapper.localization = {"valid": False, "method": "awaiting_scan_match", "score": 0.}
            return mapper
