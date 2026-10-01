"""A manually commanded worker with planar inertial dynamics and contacts.

The upright worker is supported by two planar slide joints. Forces move its
75 kg body; joint velocity, inertia and collision response determine its pose.
This is a physical mobile actor, not a human gait or biomechanical model.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .yard import BOUNDS

WORKER_MASS = 75.0
WORKER_SPEED = 1.2
WORKER_ACCELERATION = 0.9
WORKER_SPAWN = np.array([2.0, 0.0, 0.9])
WORKER_MARKER_PATH = Path(__file__).with_name("worker-marker-7.png")


def append_worker_assets(root: ET.Element) -> None:
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", {"name": "worker_tag_texture", "type": "2d", "file": str(WORKER_MARKER_PATH)})
    ET.SubElement(assets, "material", {"name": "worker_tag", "texture": "worker_tag_texture",
        "texrepeat": "1 1", "texuniform": "false", "rgba": "1 1 1 1", "specular": "0", "shininess": "0"})


def append_worker(world: ET.Element) -> None:
    body = ET.SubElement(world, "body", {"name": "worker", "pos": "2 0 0.9"})
    for name, axis in (("worker_x", "1 0 0"), ("worker_y", "0 1 0")):
        ET.SubElement(body, "joint", {"name": name, "type": "slide", "axis": axis,
                                      "damping": "8", "frictionloss": "8"})
    ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": str(WORKER_MASS), "diaginertia": "3 3 2"})
    ET.SubElement(body, "geom", {"name": "worker_vest", "type": "capsule", "fromto": "0 0 -0.18 0 0 0.34",
                                "size": "0.18", "rgba": "0.9 0.92 0.05 1", "group": "0"})
    ET.SubElement(body, "geom", {"name": "worker_stripe", "type": "box", "pos": "0 -0.17 0.18",
                                "size": "0.14 0.022 0.035", "rgba": "0.92 0.97 0.94 1",
                                "contype": "0", "conaffinity": "0", "group": "4"})
    ET.SubElement(body, "geom", {"name": "worker_head", "type": "sphere", "pos": "0 0 0.58",
                                "size": "0.115", "rgba": "0.7 0.48 0.32 1", "group": "0"})
    ET.SubElement(body, "geom", {"name": "worker_helmet", "type": "ellipsoid", "pos": "0 0 0.69",
                                "size": "0.15 0.15 0.07", "rgba": "0.98 0.9 0.08 1", "group": "0"})
    for side in (-1, 1):
        x = side * 0.085
        ET.SubElement(body, "geom", {"name": f"worker_leg_{side}", "type": "capsule",
                                    "fromto": f"{x} 0 -0.68 {x} 0 -0.24", "size": "0.065",
                                    "rgba": "0.18 0.23 0.28 1", "group": "0"})
        ET.SubElement(body, "geom", {"name": f"worker_foot_{side}", "type": "box",
                                    "pos": f"{x} -0.03 -0.85", "size": "0.075 0.14 0.05",
                                    "rgba": "0.09 0.11 0.13 1", "group": "0"})
        ET.SubElement(body, "geom", {"name": f"worker_arm_{side}", "type": "capsule",
            "fromto": f"{side * .2} 0 .3 {side * .29} 0 -.17", "size": ".055",
            "rgba": ".72 .51 .34 1", "group": "0"})
    # These are real textured scene geometry. Perception must detect their
    # pixels; their body transforms are never supplied to a tracking algorithm.
    ET.SubElement(body, "geom", {"name": "worker_roof_tag", "type": "box", "pos": "0 0 .79",
        "size": ".24 .24 .001", "material": "worker_tag", "contype": "0", "conaffinity": "0", "group": "4"})
    ET.SubElement(body, "geom", {"name": "worker_front_tag", "type": "box", "pos": "-.36 0 .25",
        "quat": ".70710678 0 -.70710678 0", "size": ".225 .225 .001", "material": "worker_tag",
        "contype": "0", "conaffinity": "0", "group": "4"})


class WorkerPhysics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model, self.data = model, data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "worker")
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ("worker_x", "worker_y")]
        self.dof_ids = np.array([model.jnt_dofadr[joint] for joint in joints])
        self.goal = data.xpos[self.body_id, :2].copy()

    @property
    def position(self) -> np.ndarray:
        return self.data.xpos[self.body_id].copy()

    def move(self, dx: float, dy: float) -> dict:
        try:
            delta = np.array([float(dx), float(dy)])
        except (TypeError, ValueError):
            return {"accepted": False, "message": "Worker movement needs finite numbers."}
        if not np.isfinite(delta).all() or np.max(np.abs(delta)) > 1.0:
            return {"accepted": False, "message": "Move the worker by at most one metre per command."}
        lower = np.array([BOUNDS["x"][0] + .5, BOUNDS["y"][0] + .5])
        upper = np.array([BOUNDS["x"][1] - .5, BOUNDS["y"][1] - .5])
        self.goal = np.clip(self.goal + delta, lower, upper)
        return {"accepted": True, "message": "Worker movement commanded.",
                "workerPosition": self.position.tolist(), "workerGoal": self.goal.tolist(), "workerManual": True}

    def stop(self) -> None:
        self.goal = self.position[:2]

    def step(self, dt: float) -> None:
        del dt
        velocity = self.data.qvel[self.dof_ids]
        delta = self.goal - self.position[:2]
        desired = 1.8 * delta
        speed = float(np.linalg.norm(desired))
        if speed > WORKER_SPEED:
            desired *= WORKER_SPEED / speed
        acceleration = 3.0 * (desired - velocity)
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude > WORKER_ACCELERATION:
            acceleration *= WORKER_ACCELERATION / magnitude
        # These external generalized forces are integrated by MuJoCo. The
        # controller never assigns a body pose, qpos or qvel.
        self.data.qfrc_applied[self.dof_ids] = WORKER_MASS * acceleration

    def snapshot(self) -> dict:
        return {"workerPosition": np.round(self.position, 5).tolist(),
                "workerVelocity": np.round(self.data.qvel[self.dof_ids], 5).tolist(),
                "workerGoal": np.round(self.goal, 5).tolist(), "workerManual": True,
                "workerModel": "75 kg planar inertial actor with force control and collisions"}
