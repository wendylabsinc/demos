"""Obstacle-aware floor coverage on a metric navigation grid.

The navigation graph represents robot centres. Cleaning samples represent floor
that its brush can reach from that graph, so dirt is never hidden inside a wall,
under an excluded furniture footprint, or in a disconnected room.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Iterable

import numpy as np

Point = tuple[float, float]
Cell = tuple[int, int]


class CoveragePlanner:
    """Inflate obstacles, connect reachable cells, and sweep alternating lanes."""

    def __init__(self, home: dict, resolution: float = 0.12, safety: float = 0.05):
        self.home = home
        self.resolution = resolution
        self.radius = float(home["robot"]["radius"])
        self.brush_radius = float(home["robot"]["brush_radius"])
        self.clearance = self.radius + safety
        self.width = float(home["width"])
        self.height = float(home["height"])
        self.nx = math.ceil(self.width / resolution)
        self.ny = math.ceil(self.height / resolution)
        self.obstacles = [self.bounds(item) for item in home["walls"] + home["furniture"]]
        xx, yy = np.meshgrid(
            (np.arange(self.nx) + 0.5) * resolution,
            (np.arange(self.ny) + 0.5) * resolution,
        )
        self.free = self.points_clear(np.column_stack([xx.ravel(), yy.ravel()])).reshape(self.ny, self.nx)
        self.reachable = np.zeros_like(self.free)
        dock = tuple(home["dock"][:2])
        candidates = np.argwhere(self.free)
        if not len(candidates):
            raise ValueError("The home has no collision-free robot positions")
        positions = np.column_stack([(candidates[:, 1] + 0.5) * resolution, (candidates[:, 0] + 0.5) * resolution])
        start_y, start_x = candidates[np.argmin(np.sum((positions - dock) ** 2, axis=1))]
        self._flood((int(start_x), int(start_y)))
        self.cells = [(int(x), int(y)) for y, x in np.argwhere(self.reachable)]
        self.positions = np.asarray([self.point(cell) for cell in self.cells])
        self._neighbors: dict[Cell, list[tuple[Cell, float]]] = {}

    @staticmethod
    def bounds(item: dict) -> tuple[float, float, float, float]:
        x, y = item["center"]
        w, h = item["size"]
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2

    def points_clear(self, points: np.ndarray, margin: float | None = None) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        clearance = self.clearance if margin is None else margin
        good = (
            (points[:, 0] >= clearance)
            & (points[:, 0] <= self.width - clearance)
            & (points[:, 1] >= clearance)
            & (points[:, 1] <= self.height - clearance)
        )
        for x0, y0, x1, y1 in self.obstacles:
            dx = np.maximum(np.maximum(x0 - points[:, 0], points[:, 0] - x1), 0)
            dy = np.maximum(np.maximum(y0 - points[:, 1], points[:, 1] - y1), 0)
            if clearance == 0:
                good &= (dx > 0) | (dy > 0)
            else:
                good &= dx * dx + dy * dy >= clearance * clearance
        return good

    def is_clear(self, point: Point, margin: float | None = None) -> bool:
        return bool(self.points_clear(np.asarray([point]), margin)[0])

    def segment_clear(self, start: Point, end: Point) -> bool:
        count = max(2, math.ceil(math.dist(start, end) / 0.035) + 1)
        samples = np.linspace(start, end, count)
        # A small sampling allowance prevents a segment grazing a curved corner.
        return bool(np.all(self.points_clear(samples, self.clearance + 0.001)))

    def point(self, cell: Cell) -> Point:
        return ((cell[0] + 0.5) * self.resolution, (cell[1] + 0.5) * self.resolution)

    def nearest(self, point: Point) -> Cell:
        index = int(np.argmin(np.sum((self.positions - point) ** 2, axis=1)))
        return self.cells[index]

    def _adjacent(self, cell: Cell, mask: np.ndarray) -> Iterable[tuple[Cell, float]]:
        x, y = cell
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            tx, ty = x + dx, y + dy
            if not (0 <= tx < self.nx and 0 <= ty < self.ny and mask[ty, tx]):
                continue
            if dx and dy and not (mask[y, tx] and mask[ty, x]):
                continue
            yield (tx, ty), math.hypot(dx, dy)

    def _flood(self, start: Cell) -> None:
        queue = deque([start])
        self.reachable[start[1], start[0]] = True
        while queue:
            cell = queue.popleft()
            for neighbor, _ in self._adjacent(cell, self.free):
                if not self.reachable[neighbor[1], neighbor[0]]:
                    self.reachable[neighbor[1], neighbor[0]] = True
                    queue.append(neighbor)

    def route(self, start: Point, goal: Point) -> list[Point]:
        """A* with safe diagonal movement and collision-checked simplification."""
        goal_cell = self.nearest(goal)
        final = goal if self.is_clear(goal) else self.point(goal_cell)
        if self.segment_clear(start, final):
            return [final]
        start_cell = self.nearest(start)
        queue: list[tuple[float, float, Cell]] = [(0, 0, start_cell)]
        costs = {start_cell: 0.0}
        parents: dict[Cell, Cell] = {}
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cell == goal_cell:
                break
            if cost > costs[cell]:
                continue
            if cell not in self._neighbors:
                self._neighbors[cell] = list(self._adjacent(cell, self.reachable))
            for neighbor, distance in self._neighbors[cell]:
                new_cost = cost + distance
                if new_cost >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = new_cost
                parents[neighbor] = cell
                heuristic = math.hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1])
                heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
        else:
            raise ValueError(f"No route to {goal}")
        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parents[cells[-1]])
        raw = [self.point(cell) for cell in reversed(cells)]
        raw.append(final)
        result: list[Point] = []
        anchor = start
        i = 0
        while i < len(raw):
            j = i + 1
            while j < len(raw) and self.segment_clear(anchor, raw[j]):
                j += 1
            waypoint = raw[j - 1]
            if math.dist(anchor, waypoint) > 0.001:
                result.append(waypoint)
                anchor = waypoint
            i = j
        return result or [final]

    def coverage_route(self, start: Point) -> list[Point]:
        """Boustrophedon lanes. A* links lanes split by furniture or walls."""
        lane_step = max(1, round(self.brush_radius * 1.5 / self.resolution))
        result: list[Point] = []
        current = start
        # Finish each room before crossing a doorway. Whole-house rows would
        # repeatedly detour through a door whenever a partition splits a lane.
        for room in self.home["rooms"]:
            x0, y0, x1, y1 = room["bounds"]
            room_mask = self.reachable.copy()
            for y in range(self.ny):
                py = (y + 0.5) * self.resolution
                if not y0 <= py <= y1:
                    room_mask[y] = False
            for x in range(self.nx):
                px = (x + 0.5) * self.resolution
                if not x0 <= px <= x1:
                    room_mask[:, x] = False
            valid_rows = np.flatnonzero(np.any(room_mask, axis=1))
            if not len(valid_rows):
                continue
            rows = list(range(int(valid_rows[0]), int(valid_rows[-1]) + 1, lane_step))
            if int(valid_rows[-1]) not in rows:
                rows.append(int(valid_rows[-1]))
            for row_number, y in enumerate(rows):
                columns = np.flatnonzero(room_mask[y])
                if not len(columns):
                    continue
                intervals: list[tuple[int, int]] = []
                first = previous = int(columns[0])
                for x in columns[1:]:
                    if x > previous + 1:
                        intervals.append((first, previous))
                        first = int(x)
                    previous = int(x)
                intervals.append((first, previous))
                backwards = row_number % 2 == 1
                if backwards:
                    intervals.reverse()
                for first, last in intervals:
                    begin, end = (last, first) if backwards else (first, last)
                    lane_start = self.point((begin, y))
                    lane_end = self.point((end, y))
                    connector = self.route(current, lane_start)
                    result.extend(connector)
                    if math.dist(lane_start, lane_end) > 0.01:
                        result.append(lane_end)
                    current = lane_end
        return result

    def cleanable_points(self, resolution: float = 0.10) -> np.ndarray:
        """Floor samples within brush reach of the dock-connected graph."""
        nx, ny = math.ceil(self.width / resolution), math.ceil(self.height / resolution)
        reachable_floor = np.zeros((ny, nx), dtype=bool)
        # Leave a 3.5 cm reserve for the controller's final waypoint tolerance.
        reach = self.brush_radius - 0.035
        for x, y in self.positions:
            x0, x1 = max(0, int((x - reach) / resolution)), min(nx, math.ceil((x + reach) / resolution))
            y0, y1 = max(0, int((y - reach) / resolution)), min(ny, math.ceil((y + reach) / resolution))
            xx, yy = np.meshgrid((np.arange(x0, x1) + 0.5) * resolution, (np.arange(y0, y1) + 0.5) * resolution)
            reachable_floor[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= reach ** 2
        yy, xx = np.where(reachable_floor)
        points = np.column_stack([(xx + 0.5) * resolution, (yy + 0.5) * resolution])
        return points[self.points_clear(points, 0)]
