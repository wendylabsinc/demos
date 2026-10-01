"""Build the furnished MuJoCo home and a constrained planar mobile robot."""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

import numpy as np


def color(value: str, scale: float = 1.0) -> str:
    rgb = [min(1, int(value[i:i + 2], 16) / 255 * scale) for i in (1, 3, 5)]
    return " ".join(f"{v:.4f}" for v in rgb) + " 1"


def camera_axes(position: tuple, target: tuple) -> str:
    forward = np.asarray(target) - np.asarray(position)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return " ".join(f"{v:.6f}" for v in np.concatenate([right, up]))


def build_world(home: dict, dust: list[dict] | None = None) -> str:
    """MJCF uses metres, seconds, kilograms and radians throughout.

    The robot has two slide joints and a yaw hinge. Its controller supplies
    world-space velocity targets derived from bounded forward and yaw commands;
    only MuJoCo integration updates its pose after reset.
    """
    parts: list[str] = []
    number = 0

    def geom(kind: str, pos: tuple, size: tuple, rgba: str, name: str | None = None, collision: bool = False, extra: str = "") -> None:
        nonlocal number
        number += 1
        identifier = escape(name or f"decoration_{number}")
        p = " ".join(f"{v:.5f}" for v in pos)
        s = " ".join(f"{v:.5f}" for v in size)
        contact = 'contype="1" conaffinity="1"' if collision else 'contype="0" conaffinity="0" mass="0"'
        parts.append(f'<geom name="{identifier}" type="{kind}" pos="{p}" size="{s}" rgba="{rgba}" {contact} {extra}/>')

    def box(x: float, y: float, z: float, w: float, d: float, h: float, rgba: str, **kwargs) -> None:
        geom("box", (x, y, z), (w / 2, d / 2, h / 2), rgba, **kwargs)

    width, height = home["width"], home["height"]
    for index, room in enumerate(home["rooms"]):
        x0, y0, x1, y1 = room["bounds"]
        floor_color = color(room["color"])
        box((x0 + x1) / 2, (y0 + y1) / 2, -0.035, x1 - x0, y1 - y0, 0.065, floor_color, name=f"floor_{index}")
        if room.get("floor") == "tile":
            for x in np.arange(x0 + 0.02, x1, 0.58):
                box(float(x), (y0 + y1) / 2, 0.0005, 0.012, y1 - y0, 0.002, "0.63 0.69 0.65 1")
            for y in np.arange(y0 + 0.02, y1, 0.58):
                box((x0 + x1) / 2, float(y), 0.0005, x1 - x0, 0.012, 0.002, "0.63 0.69 0.65 1")
        else:
            for j, y in enumerate(np.arange(y0 + 0.1, y1, 0.22)):
                box((x0 + x1) / 2, float(y), 0.0002, x1 - x0, 0.008, 0.002, color(room["color"], 0.88))
                for x in np.arange(x0 + (0.6 if j % 2 else 1.2), x1, 1.7):
                    box(float(x), float(y - 0.105), 0.0003, 0.008, 0.21, 0.002, color(room["color"], 0.91))

    for index, rug in enumerate(home.get("rugs", [])):
        x, y = rug["center"]
        w, d = rug["size"]
        box(x, y, 0.004, w, d, 0.007, color(rug["color"], 0.83), name=f"rug_border_{index}")
        box(x, y, 0.008, w - 0.10, d - 0.10, 0.007, color(rug["color"]))
        for offset in np.arange(-d / 2 + 0.16, d / 2, 0.18):
            box(x, y + float(offset), 0.012, w - 0.16, 0.012, 0.002, color(rug["color"], 0.91))

    for index, wall in enumerate(home["walls"]):
        x, y = wall["center"]
        w, d = wall["size"]
        box(x, y, 0.19, w, d, 0.38, "0.88 0.89 0.84 1", name=f"wall_{index}", collision=True)
        box(x, y, 0.391, w + 0.016, d + 0.016, 0.028, "0.98 0.97 0.93 1")
        box(x, y, 0.042, w + 0.022, d + 0.022, 0.078, "0.72 0.69 0.61 1")

    wood = "0.47 0.32 0.21 1"
    dark = "0.15 0.20 0.20 1"
    cream = "0.96 0.94 0.86 1"
    for index, item in enumerate(home["furniture"]):
        x, y = item["center"]
        w, d = item["size"]
        h = item["height"]
        tint = color(item["color"])
        kind = item["kind"]
        # A low plinth is both the footprint obstacle and part of the object.
        box(x, y, 0.075, w, d, 0.15, color(item["color"], 0.74), name=f"furniture_{index}", collision=True)
        if kind == "sofa":
            box(x, y, 0.23, w, d, 0.32, tint)
            box(x - w / 2 + 0.10, y, 0.52, 0.20, d, 0.40, color(item["color"], 0.86))
            for end in (-1, 1):
                box(x, y + end * (d / 2 - 0.12), 0.48, w, 0.24, 0.36, tint)
            for j in (-1, 0, 1):
                box(x + 0.06, y + j * d / 3.35, 0.41, w - 0.26, d / 3.5, 0.14, color(item["color"], 1.06))
            box(x - 0.13, y - d * 0.25, 0.60, 0.24, 0.36, 0.10, "0.88 0.74 0.48 1", extra='euler="0 -0.35 0"')
        elif kind in ("chair", "office_chair", "dining_chair"):
            box(x, y, h * 0.51, w * 0.94, d * 0.92, 0.14, tint)
            box(x, y + d * 0.36, h * 0.77, w * 0.93, 0.12, h * 0.43, tint)
            if kind != "dining_chair":
                for side in (-1, 1):
                    box(x + side * w * 0.42, y, h * 0.65, 0.10, d, 0.12, color(item["color"], 0.82))
            else:
                for dx in (-1, 1):
                    for dy in (-1, 1):
                        box(x + dx * w * 0.35, y + dy * d * 0.35, 0.21, 0.05, 0.05, 0.37, wood)
        elif kind in ("coffee", "dining", "desk"):
            box(x, y, h - 0.035, w, d, 0.085, tint)
            for dx in (-1, 1):
                for dy in (-1, 1):
                    box(x + dx * (w / 2 - 0.09), y + dy * (d / 2 - 0.09), h / 2, 0.065, 0.065, h - 0.08, wood)
            if kind == "coffee":
                box(x - 0.20, y + 0.04, h + 0.025, 0.34, 0.22, 0.04, "0.68 0.33 0.24 1")
                geom("cylinder", (x + 0.25, y - 0.08, h + 0.055), (0.067, 0.035), cream)
            elif kind == "dining":
                for dx in (-0.4, 0.4):
                    geom("cylinder", (x + dx, y, h + 0.02), (0.13, 0.012), cream)
                geom("cylinder", (x, y + 0.08, h + 0.09), (0.065, 0.075), "0.74 0.46 0.29 1")
                geom("ellipsoid", (x, y + 0.08, h + 0.24), (0.16, 0.12, 0.12), "0.36 0.51 0.29 1")
            else:
                box(x, y + 0.14, h + 0.24, 0.68, 0.065, 0.38, dark)
                box(x, y + 0.10, h + 0.24, 0.61, 0.008, 0.31, "0.32 0.55 0.62 1")
                box(x, y + 0.14, h + 0.04, 0.08, 0.10, 0.12, dark)
                box(x, y - 0.17, h + 0.018, 0.45, 0.16, 0.025, dark)
                box(x + 0.37, y - 0.16, h + 0.015, 0.10, 0.14, 0.02, dark)
                box(x - w * 0.32, y + 0.01, h + 0.04, 0.28, 0.32, 0.07, "0.75 0.46 0.26 1")
        elif kind == "bed":
            box(x, y, 0.27, w, d, 0.36, wood)
            box(x, y, 0.48, w - 0.04, d - 0.05, 0.19, cream)
            box(x, y - 0.29, 0.59, w - 0.06, d * 0.71, 0.09, tint)
            box(x, y + d / 2 - 0.07, 0.65, w, 0.14, 0.82, color(item["color"], 0.77))
            for dx in (-0.48, 0.48):
                box(x + dx, y + d * 0.31, 0.63, 0.76, 0.46, 0.13, "0.97 0.96 0.91 1")
            box(x, y - d * 0.35, 0.647, w - 0.05, 0.40, 0.022, "0.79 0.66 0.43 1")
        elif kind == "nightstand":
            box(x, y, h / 2, w, d, h, tint)
            for z in (0.22, 0.43):
                box(x, y - d / 2 - 0.003, z, w - 0.05, 0.014, 0.18, color(item["color"], 0.91))
                box(x, y - d / 2 - 0.017, z, 0.14, 0.035, 0.025, dark)
            geom("cylinder", (x, y, h + 0.10), (0.035, 0.09), dark)
            geom("cylinder", (x, y, h + 0.26), (0.13, 0.09), cream)
        elif kind == "counter":
            box(x, y, h / 2, w, d, h, tint)
            box(x, y, h + 0.015, w + 0.025, d + 0.025, 0.06, cream)
            if w > d:
                for offset in np.arange(-w / 2 + 0.40, w / 2, 0.8):
                    box(x + float(offset), y - d / 2 - 0.004, h * 0.49, 0.74, 0.01, h * 0.77, color(item["color"], 0.95))
                    box(x + float(offset), y - d / 2 - 0.015, h * 0.78, 0.20, 0.025, 0.025, dark)
                box(x + w * 0.28, y, h + 0.053, 0.62, d * 0.74, 0.03, dark)
                for dx in (-0.15, 0.15):
                    for dy in (-0.13, 0.13):
                        geom("cylinder", (x + w * 0.28 + dx, y + dy, h + 0.077), (0.095, 0.006), "0.36 0.39 0.37 1")
                box(x - w * 0.25, y, h + 0.052, 0.62, d * 0.73, 0.025, "0.55 0.60 0.60 1")
                box(x - w * 0.25, y, h + 0.068, 0.51, d * 0.57, 0.015, "0.34 0.42 0.43 1")
        elif kind == "tv":
            box(x, y, h / 2, w, d, h, tint)
            box(x, y, h + 0.045, 0.10, 0.16, 0.09, dark)
            box(x, y + 0.08, h + 0.44, w * 0.83, 0.08, 0.78, dark)
            box(x, y + 0.031, h + 0.44, w * 0.78, 0.012, 0.69, "0.24 0.35 0.38 1")
            box(x, y + 0.023, h + 0.33, w * 0.70, 0.004, 0.07, "0.37 0.50 0.49 1")
        elif kind == "wardrobe":
            box(x, y, h / 2, w, d, h, tint)
            for dy in (-d * 0.24, d * 0.24):
                box(x - w / 2 - 0.005, y + dy, h / 2, 0.018, d * 0.47, h * 0.92, color(item["color"], 1.03))
                box(x - w / 2 - 0.019, y + dy * 0.30, h * 0.52, 0.028, 0.035, 0.21, dark)
        elif kind == "bookcase":
            box(x + w / 2 - 0.055, y, h / 2, 0.11, d, h, tint)
            for dy in (-1, 1):
                box(x, y + dy * (d / 2 - 0.045), h / 2, w, 0.09, h, tint)
            for shelf in range(4):
                z = 0.15 + shelf * 0.29
                box(x, y, z, w, d, 0.065, tint)
                for book in range(10):
                    palette = ("0.63 0.33 0.24 1", "0.36 0.51 0.44 1", "0.72 0.62 0.38 1", "0.31 0.42 0.51 1")
                    box(x - 0.045, y - d / 2 + 0.15 + book * 0.16, z + 0.13, w * 0.74, 0.11, 0.19 + 0.035 * (book % 2), palette[(book + shelf) % 4])
        elif kind == "plant":
            geom("cylinder", (x, y, 0.19), (min(w, d) * 0.42, 0.19), "0.66 0.40 0.27 1")
            geom("cylinder", (x, y, 0.38), (min(w, d) * 0.44, 0.018), "0.74 0.48 0.32 1")
            geom("cylinder", (x, y, 0.56), (0.025, 0.20), "0.29 0.39 0.19 1")
            for j in range(6):
                angle = j * math.pi / 3
                geom("ellipsoid", (x + math.cos(angle) * w * 0.24, y + math.sin(angle) * d * 0.24, 0.57 + j * 0.045), (w * 0.27, d * 0.19, 0.16), color(item["color"], 0.85 + j * 0.04), extra=f'euler="0 0 {angle:.3f}"')
            geom("ellipsoid", (x, y, h * 0.9), (w * 0.27, d * 0.27, h * 0.16), tint)

    dock_x, dock_y, _ = home["dock"]
    box(dock_x - 0.30, dock_y, 0.12, 0.10, 0.46, 0.24, "0.23 0.28 0.27 1", name="charging_dock")
    box(dock_x - 0.19, dock_y, 0.008, 0.33, 0.42, 0.014, "0.30 0.35 0.34 1")
    box(dock_x - 0.24, dock_y, 0.22, 0.012, 0.17, 0.014, "0.44 0.82 0.55 1")

    for particle in dust or []:
        geom("ellipsoid", (particle["x"], particle["y"], 0.028), (0.023, 0.019, 0.013), "0.48 0.32 0.18 1", name=f'dust_{particle["id"]}')

    radius = home["robot"]["radius"]
    parts.append('<body name="vacuum" pos="0 0 0.095">')
    parts.append(f'<joint name="robot_x" type="slide" axis="1 0 0" damping="0.3" armature="0.02" limited="true" range="0 {width}"/>')
    parts.append(f'<joint name="robot_y" type="slide" axis="0 1 0" damping="0.3" armature="0.02" limited="true" range="0 {height}"/>')
    parts.append('<joint name="robot_yaw" type="hinge" axis="0 0 1" damping="0.04" armature="0.01"/>')
    parts.append(f'<geom name="robot_collision" type="cylinder" size="{radius} 0.063" mass="2.7" rgba="0.17 0.23 0.24 1" contype="1" conaffinity="1"/>')
    geom("cylinder", (0, 0, 0.048), (radius * 0.93, 0.018), "0.84 0.88 0.83 1", name="robot_shell")
    geom("cylinder", (0, 0, 0.071), (radius * 0.74, 0.006), "0.94 0.95 0.88 1")
    geom("cylinder", (-0.035, 0, 0.084), (0.036, 0.012), "0.24 0.32 0.32 1", name="robot_lidar")
    box(0.080, 0, 0.080, 0.05, 0.055, 0.008, "0.31 0.67 0.51 1", name="robot_button")
    box(radius * 0.92, 0, 0.010, 0.018, 0.17, 0.026, "0.15 0.20 0.20 1", name="robot_bumper")
    for side in (-1, 1):
        geom("cylinder", (0, side * radius * 0.88, -0.031), (0.046, 0.016), "0.10 0.13 0.13 1", extra='euler="1.5708 0 0"')
    geom("cylinder", (radius * 0.78, -radius * 0.73, -0.077), (0.032, 0.007), "0.24 0.31 0.27 1")
    for angle in (0, 2.094, 4.188):
        box(radius * 0.78 + math.cos(angle) * 0.035, -radius * 0.73 + math.sin(angle) * 0.035, -0.077, 0.10, 0.012, 0.008, "0.34 0.43 0.31 1", extra=f'euler="0 0 {angle}"')
    parts.append('</body>')
    overview_pos = (width * 0.93, -height * 0.97, max(width, height) * 1.42)
    target = (width * 0.5, height * 0.49, 0)
    parts.append(f'<camera name="overview" pos="{" ".join(map(str, overview_pos))}" xyaxes="{camera_axes(overview_pos, target)}" fovy="43"/>')
    parts.append(f'<camera name="top" pos="{width / 2} {height / 2} 19" xyaxes="1 0 0 0 1 0" fovy="44"/>')
    max_speed = home["robot"]["max_speed"]
    return f'''<mujoco model="Maple House robot vacuum">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" iterations="40"/>
  <size nconmax="200" njmax="1000"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.30 0.30 0.28" diffuse="0.25 0.25 0.23" specular="0.1 0.1 0.1"/>
    <rgba haze="0.93 0.93 0.88 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.93 0.94 0.91" rgb2="0.83 0.86 0.81" width="512" height="512"/>
  </asset>
  <default>
    <geom friction="0.5 0.005 0.0001" solref="0.015 1" solimp="0.95 0.99 0.001" condim="3"/>
  </default>
  <worldbody>
    <light name="sun" pos="3 -3 15" dir="0.2 0.3 -1" diffuse="0.60 0.59 0.53" ambient="0.08 0.08 0.08" castshadow="true"/>
    <light name="fill" pos="9 8 12" dir="-0.1 -0.1 -1" diffuse="0.20 0.21 0.20" castshadow="false"/>
    {''.join(parts)}
  </worldbody>
  <actuator>
    <velocity name="drive_x" joint="robot_x" kv="32" ctrlrange="-{max_speed} {max_speed}" forcerange="-12 12"/>
    <velocity name="drive_y" joint="robot_y" kv="32" ctrlrange="-{max_speed} {max_speed}" forcerange="-12 12"/>
    <velocity name="steering" joint="robot_yaw" kv="2.8" ctrlrange="-3.2 3.2" forcerange="-2 2"/>
  </actuator>
</mujoco>'''
