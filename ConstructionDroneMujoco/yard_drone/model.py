"""A 1.2 kg inspection quadrotor and collision geometry for the yard.

Each rotor applies a force at its hub and a reaction torque. MuJoCo integrates
the motor's first-order response, the free airframe, gravity and contacts.
All geometry is generated here, so the demo needs no downloaded model assets.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .yard import HOME, OBSTACLES
from .environment import append_hazard_geometries
from .worker import append_worker, append_worker_assets

MASS = 1.2
INERTIA = np.array([0.022, 0.022, 0.035])
GRAVITY = 9.81
ARM = 0.22
ROTOR_HEIGHT = 0.025
ROTOR_XY = np.array([[ARM, -ARM], [-ARM, -ARM], [-ARM, ARM], [ARM, ARM]])
ROTOR_SPIN = np.array([1.0, -1.0, 1.0, -1.0])
ROTOR_MAX_THRUST = 7.0
ROTOR_TORQUE_COEFF = 0.025
MOTOR_TIME_CONSTANT = 0.025
PROP_RADIUS = 0.13
PHYSICS_HZ = 500
TIMESTEP = 1.0 / PHYSICS_HZ
SENSOR_WIDTH = 13


def _numbers(values) -> str:
    return " ".join(str(float(value)) for value in values)


def build_xml() -> str:
    root = ET.Element("mujoco", {"model": "construction_inspection_quadrotor"})
    ET.SubElement(root, "compiler", {"inertiafromgeom": "false", "autolimits": "true"})
    ET.SubElement(root, "option", {
        "timestep": str(TIMESTEP), "integrator": "RK4", "gravity": "0 0 -9.81",
        "density": "0", "viscosity": "0", "iterations": "50",
    })
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "640", "offheight": "480"})
    ET.SubElement(visual, "map", {"znear": "0.0002", "zfar": "20"})
    ET.SubElement(root, "size", {"memory": "64M"})
    append_worker_assets(root)
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {
        "friction": "0.8 0.02 0.005", "solref": "0.006 1", "solimp": "0.95 0.99 0.001",
    })
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 10", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "size": "12 10 0.05", "rgba": "0.35 0.33 0.29 1",
    })
    for index, obstacle in enumerate(OBSTACLES):
        color = obstacle["color"].lstrip("#")
        rgba = [int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [1]
        ET.SubElement(world, "geom", {
            "name": f"obstacle_{index}", "type": "box", "pos": _numbers(obstacle["position"]),
            "size": _numbers(obstacle["size"]), "rgba": _numbers(rgba),
        })
    # Boundary fences are also physical collision geometry.
    for axis, value in ((0, -9.0), (0, 9.0), (1, -8.0), (1, 8.0)):
        position = [0.0, 0.0, .55]
        size = [.035, 8., .55] if axis == 0 else [9., .035, .55]
        position[axis] = value
        ET.SubElement(world, "geom", {"name": f"fence_{axis}_{value}", "type": "box",
            "pos": _numbers(position), "size": _numbers(size), "rgba": ".36 .42 .35 1"})

    body = ET.SubElement(world, "body", {"name": "drone", "pos": _numbers(HOME)})
    ET.SubElement(body, "freejoint", {"name": "drone_joint"})
    ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": str(MASS), "diaginertia": _numbers(INERTIA)})
    ET.SubElement(body, "geom", {
        "name": "airframe", "type": "box", "size": "0.115 0.085 0.045", "rgba": "0.17 0.2 0.22 1",
    })
    ET.SubElement(body, "geom", {
        "name": "battery", "type": "box", "pos": "0 0 0.055", "size": "0.085 0.06 0.025",
        "rgba": "0.95 0.55 0.16 1", "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(body, "geom", {
        "name": "camera", "type": "sphere", "pos": "0.105 0 -0.04", "size": "0.03",
        "rgba": "0.08 0.1 0.12 1", "contype": "0", "conaffinity": "0",
    })
    for rotor, (x, y) in enumerate(ROTOR_XY):
        ET.SubElement(body, "geom", {
            "name": f"arm_{rotor}", "type": "capsule", "fromto": f"0 0 0 {x} {y} 0",
            "size": "0.015", "rgba": "0.28 0.31 0.32 1",
        })
        ET.SubElement(body, "geom", {
            "name": f"motor_{rotor}", "type": "cylinder", "pos": f"{x} {y} 0.025", "size": "0.025 0.025",
            "rgba": "0.13 0.14 0.15 1",
        })
        ET.SubElement(body, "geom", {
            "name": f"prop_{rotor}", "type": "cylinder", "pos": f"{x} {y} 0.055",
            "size": f"{PROP_RADIUS} 0.002", "rgba": "0.25 0.28 0.29 0.25",
            "contype": "0", "conaffinity": "0",
        })
        ET.SubElement(body, "geom", {
            "name": f"foot_{rotor}", "type": "sphere", "pos": f"{x * .6} {y * .6} -0.09",
            "size": "0.03", "rgba": "0.2 0.22 0.22 1",
        })
        ET.SubElement(body, "site", {"name": f"rotor_{rotor}", "pos": f"{x} {y} {ROTOR_HEIGHT}", "size": "0.005"})
    ET.SubElement(body, "site", {"name": "imu", "pos": "0 0 0", "size": "0.004"})
    ET.SubElement(body, "camera", {"name": "survey", "pos": "0 0 -0.075", "quat": "1 0 0 0", "fovy": "65"})
    ET.SubElement(body, "camera", {"name": "tracking", "pos": "0.14 0 -0.07",
        "xyaxes": "0 -1 0 0.70710678 0 0.70710678", "fovy": "90"})
    for geom in body.findall("geom"):
        geom.set("group", "2")
    append_worker(world)
    append_hazard_geometries(world)

    actuators = ET.SubElement(root, "actuator")
    for rotor in range(4):
        ET.SubElement(actuators, "general", {
            "name": f"rotor_{rotor}", "site": f"rotor_{rotor}",
            "gear": f"0 0 1 0 0 {ROTOR_SPIN[rotor] * ROTOR_TORQUE_COEFF}",
            "dyntype": "filter", "dynprm": str(MOTOR_TIME_CONSTANT),
            "gaintype": "fixed", "gainprm": "1", "biastype": "none",
            "ctrlrange": f"0 {ROTOR_MAX_THRUST}",
        })
    sensors = ET.SubElement(root, "sensor")
    for kind, name in (("framepos", "position"), ("framelinvel", "velocity"), ("framequat", "attitude")):
        ET.SubElement(sensors, kind, {"name": name, "objtype": "site", "objname": "imu"})
    ET.SubElement(sensors, "gyro", {"name": "rates", "site": "imu"})
    return ET.tostring(root, encoding="unicode")


def build_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml())
