"""Evolving yard entities visible to the same MuJoCo cameras as the drone.

Oil is a conservative, thin-film gravity flow over ray-measured ground heights.
Fire uses a coarse fuel/energy balance, heat diffusion, wind advection, radiative
and buoyant heat loss, and transported smoke. These reduced models demonstrate
changing physical fields; they are not validated combustion or fluid CFD.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .worker import WorkerPhysics
from .yard import BOUNDS

CELL = 0.5
GRID_X = np.arange(BOUNDS["x"][0] + CELL / 2, BOUNDS["x"][1], CELL)
GRID_Y = np.arange(BOUNDS["y"][0] + CELL / 2, BOUNDS["y"][1], CELL)
NX, NY = len(GRID_X), len(GRID_Y)
FIELD_HZ = 25
AMBIENT_TEMPERATURE = 293.15
OIL_DENSITY = 850.0
OIL_VISCOSITY = 0.09
GRAVITY = 9.81


def append_hazard_geometries(world: ET.Element) -> None:
    for row, y in enumerate(GRID_Y):
        for col, x in enumerate(GRID_X):
            index = row * NX + col
            for kind, geom_type, size, z in (
                ("oil", "box", f"{CELL * .492} {CELL * .492} .003", .006),
                ("fire", "cylinder", f"{CELL * .32} .03", .03),
                ("smoke", "sphere", ".05", .1),
            ):
                ET.SubElement(world, "geom", {"name": f"field_{kind}_{index}", "type": geom_type,
                    "size": size, "pos": f"{x} {y} {z}", "rgba": "0 0 0 0", "group": "3",
                    "contype": "0", "conaffinity": "0"})


def _laplacian(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, 1, mode="edge")
    return (padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:]
            + padded[1:-1, :-2] - 4 * field) / CELL ** 2


def _advect(field: np.ndarray, wind: np.ndarray, dt: float) -> np.ndarray:
    """First-order upwind transport with no influx at the yard boundaries."""
    result = field.copy()
    for axis, speed in ((1, wind[0]), (0, wind[1])):
        upstream = np.roll(field, 1 if speed >= 0 else -1, axis=axis)
        boundary = [slice(None), slice(None)]
        boundary[axis] = 0 if speed >= 0 else -1
        upstream[tuple(boundary)] = 0
        result -= abs(speed) * dt / CELL * (field - upstream)
    return np.maximum(result, 0)


class YardEnvironment:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.worker = WorkerPhysics(model, data)
        self.drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        shape = (NY, NX)
        self.oil = np.zeros(shape)  # metres of liquid column
        self.fuel = np.zeros(shape)  # kg / m^2
        self.temperature = np.full(shape, AMBIENT_TEMPERATURE)
        self.smoke = np.zeros(shape)
        self.smoke_height = np.zeros(shape)
        self.wind = np.array([0.35, 0.1])  # metres / second
        self.terrain = self._ground_heights()
        self.blocked = self.terrain > .1
        self.x, self.y = np.meshgrid(GRID_X, GRID_Y)
        self.field_time = 0.0
        self._pending = 0.0
        self._active = False
        self.released_oil_volume = 0.0
        self.burned_fuel_mass = 0.0
        self._last_release: dict | None = None
        self.geom_ids = {kind: np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"field_{kind}_{index}")
            for index in range(NX * NY)]) for kind in ("oil", "fire", "smoke")}
        self._visuals()

    def _ground_heights(self) -> np.ndarray:
        heights = np.zeros((NY, NX))
        geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        geom_id = np.zeros(1, dtype=np.int32)
        for row, y in enumerate(GRID_Y):
            for col, x in enumerate(GRID_X):
                origin = np.array([x, y, 9.0])
                distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]),
                                        geom_group, 1, self.worker.body_id, geom_id)
                if distance >= 0:
                    heights[row, col] = max(0, origin[2] - distance)
        return heights

    def inject(self, kind: str, position=None) -> dict:
        if kind not in ("fire", "oil"):
            raise ValueError("Inject fire or oil")
        if position is None:
            position = self.data.xpos[self.drone_id, :2].copy()
        try:
            point = np.asarray(position, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Choose a finite ground position") from error
        if point.shape not in ((2,), (3,)) or not np.isfinite(point).all():
            raise ValueError("Choose a finite ground position")
        x, y = point[:2]
        if not BOUNDS["x"][0] + .5 <= x <= BOUNDS["x"][1] - .5 or not BOUNDS["y"][0] + .5 <= y <= BOUNDS["y"][1] - .5:
            raise ValueError("Incident position must be inside the yard fence")
        col, row = int((x - BOUNDS["x"][0]) / CELL), int((y - BOUNDS["y"][0]) / CELL)
        if self.blocked[row, col]:
            raise ValueError("Choose open ground outside a construction obstacle")
        radius = .6 if kind == "fire" else .4
        weights = np.exp(-((self.x - x) ** 2 + (self.y - y) ** 2) / (2 * radius ** 2))
        weights[self.blocked] = 0
        if kind == "oil":
            volume = .018  # An 18 litre finite spill, with no endless hidden source.
            self.oil += volume * weights / (weights.sum() * CELL ** 2)
            self.released_oil_volume += volume
        else:
            # A finite combustible fuel patch. Its energy source remains on the
            # surface; the plume above it is transported by the full wind.
            self.fuel += 6.0 * weights
            ignition = AMBIENT_TEMPERATURE + 900 * weights
            self.temperature = np.maximum(self.temperature, ignition)
        self._last_release = {"kind": kind, "position": [float(x), float(y), 0.0],
                              "simulationTime": float(self.data.time)}
        self._active = True
        self._visuals()
        # Static field geom poses are model values; update data transforms for
        # a camera taken before the next physics step. No joint state is changed.
        self.data.geom_xpos[self.geom_ids["oil"]] = self.model.geom_pos[self.geom_ids["oil"]]
        self.data.geom_xpos[self.geom_ids["fire"]] = self.model.geom_pos[self.geom_ids["fire"]]
        self.data.geom_xpos[self.geom_ids["smoke"]] = self.model.geom_pos[self.geom_ids["smoke"]]
        return {"accepted": True, "message": f"Released {kind} at the selected ground position.",
                "position": self._last_release["position"]}

    def clear(self) -> None:
        self.oil.fill(0)
        self.fuel.fill(0)
        self.temperature.fill(AMBIENT_TEMPERATURE)
        self.smoke.fill(0)
        self.smoke_height.fill(0)
        self._active = False
        self._visuals()

    def move_worker(self, dx: float, dy: float) -> dict:
        return self.worker.move(dx, dy)

    def step(self, dt: float) -> None:
        self.worker.step(dt)
        self._pending += dt
        interval = 1 / FIELD_HZ
        while self._pending >= interval - 1e-12:
            if self._active:
                self._fields(interval)
            self._pending -= interval
            self.field_time += interval
        # Model transforms are read by mj_step and the real camera renderer.

    def _oil_flow(self, dt: float) -> None:
        head = self.terrain + self.oil
        mobility = OIL_DENSITY * GRAVITY / (3 * OIL_VISCOSITY)
        change = np.zeros_like(self.oil)
        for axis in (0, 1):
            lower = [slice(None), slice(None)]
            upper = [slice(None), slice(None)]
            lower[axis], upper[axis] = slice(None, -1), slice(1, None)
            lower, upper = tuple(lower), tuple(upper)
            gradient = (head[lower] - head[upper]) / CELL
            donor = np.where(gradient >= 0, self.oil[lower], self.oil[upper])
            flux = mobility * donor ** 3 * gradient * dt / CELL
            # Each donor can lose at most one fourth of its volume to an edge
            # during this explicit finite-volume update, maintaining positivity.
            flux = np.clip(flux, -self.oil[upper] * .24, self.oil[lower] * .24)
            flux[self.blocked[lower] | self.blocked[upper]] = 0
            change[lower] -= flux
            change[upper] += flux
        self.oil = np.maximum(0, self.oil + change)

    def _fields(self, dt: float) -> None:
        self._oil_flow(dt)
        t = self.temperature
        activation = np.clip((t - 470) / 380, 0, 1)
        solid_burn = np.minimum(self.fuel / dt, .008 * self.fuel * activation)
        oil_burn = np.minimum(self.oil / dt, .004 * self.oil * activation)
        self.fuel -= solid_burn * dt
        self.oil -= oil_burn * dt
        burned = solid_burn + oil_burn * OIL_DENSITY
        self.burned_fuel_mass += float(burned.sum() * CELL ** 2 * dt)
        excess = t - AMBIENT_TEMPERATURE
        heat_release = burned * 16e6  # W/m^2, reduced effective fuel heat
        radiation = .35 * 5.670374e-8 * (t ** 4 - AMBIENT_TEMPERATURE ** 4)
        buoyant = (45 + 90 * np.sqrt(np.maximum(excess, 0) / AMBIENT_TEMPERATURE)) * excess
        change = (heat_release - radiation - buoyant) / 3500.0 + .03 * _laplacian(t)
        t = np.clip(t + dt * change, AMBIENT_TEMPERATURE, 1800)
        # Ground-coupled fuel temperature transports much slower than airborne
        # smoke. Advecting the entire thermal source at the airspeed would
        # incorrectly blow the burning surface off its stationary fuel patch.
        transported = _advect(t - AMBIENT_TEMPERATURE, self.wind * .05, dt)
        self.temperature = transported + AMBIENT_TEMPERATURE
        self.temperature[self.blocked] = AMBIENT_TEMPERATURE
        self.smoke = _advect(self.smoke, self.wind, dt) + dt * (.24 * burned - .16 * self.smoke)
        rise = np.sqrt(2 * GRAVITY * .7 * np.maximum(self.temperature - AMBIENT_TEMPERATURE, 0)
                       / np.maximum(self.temperature, AMBIENT_TEMPERATURE))
        self.smoke_height = np.minimum(5.5, _advect(self.smoke_height * self.smoke, self.wind, dt)
                                       / np.maximum(self.smoke, 1e-6) + dt * rise)
        self.smoke_height[self.smoke < .001] = 0
        self._visuals()

    def _visuals(self) -> None:
        oil, temperature, smoke = self.oil.ravel(), self.temperature.ravel(), self.smoke.ravel()
        ids = self.geom_ids["oil"]
        self.model.geom_size[ids, 2] = np.maximum(.003, oil * .5)
        self.model.geom_pos[ids, 2] = self.model.geom_size[ids, 2] + .004
        self.model.geom_rgba[ids] = np.column_stack([
            np.full(len(ids), .035), np.full(len(ids), .046), np.full(len(ids), .07),
            np.where(oil > .00015, np.clip(oil / .003, .45, .97), 0)])
        ids = self.geom_ids["fire"]
        heat = np.clip((temperature - 500) / 600, 0, 1)
        height = .03 + 1.2 * heat
        self.model.geom_size[ids, 0] = CELL * (.18 + .18 * heat)
        self.model.geom_size[ids, 1] = height / 2
        self.model.geom_pos[ids, 2] = height / 2 + .015
        self.model.geom_rgba[ids] = np.column_stack([
            np.full(len(ids), 1.), .15 + .6 * heat, np.full(len(ids), .025),
            np.where(temperature > 510, .78 + .2 * heat, 0)])
        ids = self.geom_ids["smoke"]
        radius = np.clip(.18 + np.sqrt(smoke) * 2, .18, .55)
        self.model.geom_size[ids, 0] = radius
        self.model.geom_pos[ids, 2] = .2 + self.smoke_height.ravel()
        self.model.geom_rgba[ids] = np.column_stack([
            np.full(len(ids), .18), np.full(len(ids), .18), np.full(len(ids), .2),
            np.where(smoke > .001, np.clip(smoke * 5, 0, .22), 0)])

    def snapshot(self) -> dict:
        visible = (self.oil > .00015) | (self.temperature > 510) | (self.smoke > .001)
        cells = []
        for row, col in np.argwhere(visible):
            cells.append({"x": float(GRID_X[col]), "y": float(GRID_Y[row]),
                          "oilDepth": round(float(self.oil[row, col]), 6),
                          "temperatureK": round(float(self.temperature[row, col]), 2),
                          "fuelKgM2": round(float(self.fuel[row, col]), 5),
                          "smoke": round(float(self.smoke[row, col]), 5),
                          "smokeHeight": round(float(self.smoke_height[row, col]), 3)})
        return {**self.worker.snapshot(), "hazardFields": {
            "time": round(self.field_time, 3), "cellSize": CELL, "width": NX, "height": NY,
            "origin": [float(GRID_X[0]), float(GRID_Y[0])], "wind": self.wind.tolist(), "cells": cells,
            "oilVolumeM3": round(float(self.oil.sum() * CELL ** 2), 8),
            "releasedOilVolumeM3": self.released_oil_volume,
            "burnedFuelKg": round(self.burned_fuel_mass, 6), "lastRelease": self._last_release,
            "model": "Coarse conservative oil film and fuel/energy/advection fire approximation"}}
