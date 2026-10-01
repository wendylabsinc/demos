"""Procedural MuJoCo world with lidar, cameras and physical bumper contacts.

The chassis has planar dynamics. Bounded velocity actuators follow a bicycle
model, while MuJoCo integrates inertia and collision response. Tires, suspension
and slip are not modeled. All navigation observations come from ray casts, not
the obstacle manifest that is provided separately for the viewer. Pedestrians
have bounded planar dynamics and range-based autonomous steering. An explicit
seed is available for reproducible tests; normal runs sample fresh environments.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import threading
import time
from xml.sax.saxutils import escape

import mujoco
import numpy as np

from .types import Command, Observation, Pose


LENGTH = 0.48
WIDTH = 0.26
WHEELBASE = 0.32
ARENA_WIDTH = 14.0
ARENA_HEIGHT = 12.0
BEAMS = 180
RANGE_MAX = 6.0
PERSON_RADIUS = 0.23
PHYSICS_TIMESTEP = .0025


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _approach(value: float, target: float, increment: float) -> float:
    return value + _clamp(target - value, increment)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A drive command must contain finite numbers")
    return value


class Simulation:
    """The simulator backend. Positive steering turns the car left."""

    max_speed = .45
    max_steering = .55
    camera_name = "front_rgbd"

    def __init__(self, seed: int | None = None, *, population: int = 3):
        self.seed = seed
        self.rng = random.Random(seed)
        self.population = max(0, min(5, int(population)))
        self.lock = threading.RLock()
        self._geom_group = np.array([1, 1, 0, 1, 0, 0], dtype=np.uint8)
        self._actor_groups = np.array([1, 1, 1, 1, 0, 0], dtype=np.uint8)
        self._hit_id = np.zeros(1, dtype=np.int32)
        self._people: dict[int, tuple[float, float]] = {}
        self._actors: dict[int, dict] = {}
        self.reset()

    def _layout(self) -> list[dict]:
        """Sample separated furniture; keep an open perimeter for circulation."""
        result = []
        colors = ("#5d6270", "#69635f", "#666973", "#555d68", "#4e6b58")
        for _ in range(250):
            if len(result) >= 8:
                break
            x, y = self.rng.uniform(-4.6, 4.6), self.rng.uniform(-3.4, 3.4)
            if self.rng.random() < .4:
                obj = {"kind": "circle", "x": x, "y": y,
                       "r": self.rng.uniform(.25, .48), "height": self.rng.uniform(1., 2.)}
                radius = obj["r"]
            else:
                obj = {"kind": "box", "x": x, "y": y,
                       "w": self.rng.uniform(.65, 1.65), "h": self.rng.uniform(.55, 1.3),
                       "height": self.rng.uniform(.55, .95)}
                radius = math.hypot(obj["w"], obj["h"]) / 2
            if math.hypot(x + 5., y + 3.5) < radius + 1.7:
                continue
            if any(math.hypot(x-o["x"], y-o["y"]) < radius + o["_radius"] + 1.15
                   for o in result):
                continue
            obj["_radius"] = radius
            obj["color"] = self.rng.choice(colors)
            result.append(obj)
        for obj in result:
            del obj["_radius"]
        return result

    def _xml(self) -> str:
        person_assets = Path(__file__).resolve().parent.parent / "assets" / "person"
        props = []
        for index, obstacle in enumerate(self.obstacles):
            color = obstacle["color"].lstrip("#")
            rgba = " ".join(str(int(color[k:k + 2], 16) / 255)
                            for k in (0, 2, 4)) + " 1"
            z = obstacle["height"] / 2
            if obstacle["kind"] == "box":
                kind = "box"
                size = f'{obstacle["w"] / 2} {obstacle["h"] / 2} {z}'
            else:
                kind = "cylinder"
                size = f'{obstacle["r"]} {z}'
            props.append(f'<geom name="prop{index}" type="{kind}" '
                         f'pos="{obstacle["x"]} {obstacle["y"]} {z}" '
                         f'size="{size}" rgba="{escape(rgba)}" '
                         'contype="2" conaffinity="5" group="0"/>')
        pedestrians = "".join(
            f'<body name="person{i}" pos="0 0 0">'
            f'<joint name="person{i}_x" type="slide" axis="1 0 0" damping=".2"/>'
            f'<joint name="person{i}_y" type="slide" axis="0 1 0" damping=".2"/>'
            f'<joint name="person{i}_yaw" type="hinge" axis="0 0 1" damping=".1"/>'
            f'<geom name="person{i}_body" type="capsule" '
            'fromto="0 0 .24 0 0 1.25" size=".23" mass="8" '
            'rgba=".25 .25 .25 1" group="3" contype="4" conaffinity="7"/>'
            '<geom type="mesh" mesh="human" material="human_clothing" '
            'mass=".001" group="4" contype="0" conaffinity="0"/>'
            '</body>' for i in range(5)
        )
        actor_motors = "".join(
            f'<velocity name="person{i}_{axis}_motor" joint="person{i}_{axis}" '
            f'kv="{45 if axis != "yaw" else 12}" forcelimited="true" '
            f'forcerange="{-16 if axis != "yaw" else -6} {16 if axis != "yaw" else 6}"/>'
            for i in range(5) for axis in ("x", "y", "yaw")
        )
        wheels = "".join(
            f'<body name="wheel{i}" pos="{x} {y} .055">'
            f'<joint name="wheel{i}" type="hinge" axis="0 1 0" damping=".001"/>'
            '<geom type="cylinder" size=".05 .027" euler="90 0 0" '
            'rgba=".05 .06 .075 1" group="2" contype="0" conaffinity="0" mass=".02"/>'
            '</body>'
            for i, (x, y) in enumerate(((-.16, .14), (-.16, -.14), (.16, .14), (.16, -.14)))
        )
        return f"""<mujoco model="sassy-rosmaster-r2">
  <compiler angle="degree" autolimits="true"/>
  <option gravity="0 0 0" timestep="0.0025" integrator="implicitfast"/>
  <default><geom friction=".5 .005 .0001" solref=".004 1" solimp=".99 .999 .001"/></default>
  <asset>
    <texture name="human_texture" type="2d" file="{escape(str(person_assets / 'person.png'))}"/>
    <material name="human_clothing" texture="human_texture" texuniform="false"/>
    <mesh name="human" file="{escape(str(person_assets / 'person.obj'))}"/>
  </asset>
  <visual><global offwidth="416" offheight="312"/><quality offsamples="0"/></visual>
  <worldbody>
    <light pos="-3 -3 8" diffuse=".8 .8 .8"/>
    <geom type="plane" size="7 6 .1" rgba=".18 .2 .23 1" group="5" contype="0" conaffinity="0"/>
    <geom name="wallwest" type="box" pos="-7.1 0 .5" size=".1 6.2 .5" rgba=".3 .32 .36 1" contype="2" conaffinity="5"/>
    <geom name="walleast" type="box" pos="7.1 0 .5" size=".1 6.2 .5" rgba=".3 .32 .36 1" contype="2" conaffinity="5"/>
    <geom name="wallsouth" type="box" pos="0 -6.1 .5" size="7.2 .1 .5" rgba=".3 .32 .36 1" contype="2" conaffinity="5"/>
    <geom name="wallnorth" type="box" pos="0 6.1 .5" size="7.2 .1 .5" rgba=".3 .32 .36 1" contype="2" conaffinity="5"/>
    {''.join(props)}
    <body name="car">
      <joint name="car_x" type="slide" axis="1 0 0" damping=".03"/>
      <joint name="car_y" type="slide" axis="0 1 0" damping=".03"/>
      <joint name="car_yaw" type="hinge" axis="0 0 1" damping=".03"/>
      <geom name="bumper" type="box" size=".24 .13 .065" pos="0 0 .13" mass="2.6" rgba=".10 .11 .14 1" group="2" contype="1" conaffinity="6"/>
      <geom type="box" size=".17 .11 .012" pos="0 0 .22" mass=".15" rgba=".09 .11 .12 1" group="2" contype="0" conaffinity="0"/>
      <geom type="box" size=".075 .06 .02" pos="-.08 0 .255" mass=".1" rgba=".12 .36 .19 1" group="2" contype="0" conaffinity="0"/>
      <geom type="cylinder" size=".055 .04" pos="0 0 .33" mass=".08" rgba=".12 .14 .17 1" group="2" contype="0" conaffinity="0"/>
      <geom type="box" size=".02 .065 .025" pos=".20 0 .25" mass=".05" rgba=".16 .17 .20 1" group="2" contype="0" conaffinity="0"/>
      <site name="lidar" pos="0 0 .33"/>
      <camera name="front_rgbd" pos=".20 0 .33" xyaxes="0 -1 0 -.258819 0 .965926" fovy="62"/>
      {wheels}
    </body>
    {pedestrians}
  </worldbody>
  <actuator>
    <velocity name="drive_x" joint="car_x" kv="22" forcelimited="true" forcerange="-20 20"/>
    <velocity name="drive_y" joint="car_y" kv="22" forcelimited="true" forcerange="-20 20"/>
    <velocity name="turn" joint="car_yaw" kv="8" forcelimited="true" forcerange="-4 4"/>
    {actor_motors}
  </actuator>
</mujoco>"""

    def reset(self) -> None:
        """Start a fresh sample; an explicit seed reproduces the sample sequence."""
        with self.lock:
            self.scenario_id = self.rng.getrandbits(64)
            self.obstacles = self._layout()
            self.model = mujoco.MjModel.from_xml_string(self._xml())
            self.data = mujoco.MjData(self.model)
            self._car_body = self.model.body("car").id
            self._bumper = self.model.geom("bumper").id
            self._person_bodies = [self.model.body(f"person{i}").id for i in range(5)]
            self._person_qpos = [int(self.model.jnt_qposadr[self.model.joint(f"person{i}_x").id])
                                 for i in range(5)]
            self._person_qvel = [int(self.model.jnt_dofadr[self.model.joint(f"person{i}_x").id])
                                 for i in range(5)]
            self._people.clear()
            self._actors.clear()
            self.data.qpos[:3] = [-5.0, -3.5, self.rng.uniform(-.3, .3)]
            self._speed = self._steering = self._wheel_angle = 0.0
            self.collision_count = 0
            self.collision = False
            self._last_collision_time = -1.0
            self.max_penetration = 0.0
            self.actor_contact_count = 0
            for i, address in enumerate(self._person_qpos):
                self.data.qpos[address:address+3] = (30.+i*2, 30., 0.)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(self.population):
                self.add_person()

    def camera_snapshot(self):
        """Copy physics state atomically; the camera worker owns its GL context.

        No geometry classifications accompany the image. Camera inference must
        identify people from RGB, with depth used only for measured distance.
        """
        with self.lock:
            model = self.model
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, self.data)
            pose = Pose(*map(float, data.qpos[:3]))
            return model, data, pose, time.monotonic()

    def _contacts(self) -> bool:
        touching = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._bumper in (int(contact.geom1), int(contact.geom2)):
                touching = touching or contact.dist <= .001
                self.max_penetration = max(self.max_penetration, -float(contact.dist))
        return touching

    def step(self, command: Command, dt: float = .05) -> None:
        with self.lock:
            self._step(command, dt)

    def _step(self, command: Command, dt: float = .05) -> None:
        dt = _finite(dt)
        if not 0 < dt <= .25:
            raise ValueError("The simulation step must be between 0 and .25 seconds")
        requested_speed = _clamp(_finite(command.speed), self.max_speed)
        requested_steering = _clamp(_finite(command.steering), self.max_steering)
        steps = max(1, math.ceil(dt / PHYSICS_TIMESTEP))
        substep = dt / steps
        self.model.opt.timestep = substep
        self._actor_commands(dt)
        touched = False
        for _ in range(steps):
            braking = abs(requested_speed) < abs(self._speed) or requested_speed * self._speed < 0
            target = 0.0 if requested_speed * self._speed < 0 else requested_speed
            self._speed = _approach(self._speed, target, (1.6 if braking else .8) * substep)
            self._steering = _approach(self._steering, requested_steering, 1.5 * substep)
            yaw = float(self.data.qpos[2])
            self.data.ctrl[:3] = (self._speed * math.cos(yaw), self._speed * math.sin(yaw),
                                 self._speed * math.tan(self._steering) / WHEELBASE)
            mujoco.mj_step(self.model, self.data)
            touched = self._contacts() or touched
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                if contact.dist <= 0 and self._bumper not in (contact.geom1, contact.geom2):
                    self.actor_contact_count += 1
            # Noncolliding wheel visuals rotate with actual measured travel.
            speed = float(self.data.qvel[0]) * math.cos(yaw) + float(self.data.qvel[1]) * math.sin(yaw)
            self._wheel_angle += speed * substep / .05
        self.data.qpos[3:7] = self._wheel_angle
        self.data.qvel[3:7] = 0.0
        mujoco.mj_forward(self.model, self.data)
        touched = touched or self._contacts()
        if touched and not self.collision and self.data.time - self._last_collision_time > .5:
            self.collision_count += 1
            self._last_collision_time = self.data.time
        self.collision = touched
        for index, address in enumerate(self._person_qpos):
            if index in self._actors:
                self._people[index] = tuple(map(float, self.data.qpos[address:address+2]))

    def read(self) -> Observation:
        with self.lock:
            pose = Pose(float(self.data.qpos[0]), float(self.data.qpos[1]),
                        math.atan2(math.sin(float(self.data.qpos[2])), math.cos(float(self.data.qpos[2]))))
            origin = self.data.site_xpos[self.model.site("lidar").id].copy()
            ranges = []
            increment = math.tau / BEAMS
            for index in range(BEAMS):
                angle = pose.yaw - math.pi + index * increment
                direction = np.array([math.cos(angle), math.sin(angle), 0.0])
                distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                         self._geom_group, True, self._car_body, self._hit_id)
                ranges.append(RANGE_MAX if distance < 0 or distance > RANGE_MAX else max(0., float(distance)))
            return Observation(pose=pose, ranges=ranges, angle_min=-math.pi,
                               angle_increment=increment, range_max=RANGE_MAX,
                               captured_at=time.monotonic(), people=[], collision=self.collision)

    def _actor_commands(self, dt: float) -> None:
        """Humans react to their own current range scans, without a route script."""
        for index, actor in self._actors.items():
            q = self._person_qpos[index]
            x, y, yaw = map(float, self.data.qpos[q:q+3])
            origin = np.array([x, y, .13])
            angles = [math.tau*i/48 for i in range(48)]
            distances = []
            for angle in angles:
                direction = np.array([math.cos(angle), math.sin(angle), 0.])
                distance = mujoco.mj_ray(self.model, self.data, origin, direction,
                                         self._actor_groups, True, self._person_bodies[index], self._hit_id)
                distances.append(8. if distance < 0 else min(8., float(distance)))
            actor["change_at"] -= dt
            desired = actor["heading"]
            def wrap(a):
                return math.atan2(math.sin(a), math.cos(a))
            front = min(d for a, d in zip(angles, distances) if abs(wrap(a-desired)) < .55)
            approaching = min(d for a, d in zip(angles, distances)
                              if math.cos(a-desired) > -.1)
            if front < 1.3 or approaching < .8 or actor["change_at"] <= 0:
                # Neighboring rays must all clear the body and its braking margin.
                choices = []
                for i, angle in enumerate(angles):
                    clearance = min(distances[(i+j)%48] for j in range(-4, 5))
                    if any(d < .8 and math.cos(a-angle) > -.15
                           for a,d in zip(angles, distances)):
                        continue
                    if clearance > 1.1:
                        inertia = math.cos(wrap(angle-desired))
                        choices.append((min(clearance, 4.)+.45*inertia+self.rng.uniform(-.25,.25), angle))
                if choices:
                    desired = max(choices)[1]
                    actor["heading"] = desired
                actor["change_at"] = self.rng.uniform(2., 7.)
            direction_clearance = min(d for a,d in zip(angles, distances) if abs(wrap(a-desired)) < .55)
            # Brake early for any nearby body, then turn toward verified space.
            speed = actor["speed"] * max(0., min(1., (direction_clearance-.85)/1.))
            if any(d < .7 and math.cos(a-yaw) > -.1 for a,d in zip(angles, distances)):
                speed = 0.
            angle_error = wrap(desired-yaw)
            if abs(angle_error) > .6:
                speed = 0.
            self.data.ctrl[3+3*index:6+3*index] = (
                speed*math.cos(yaw), speed*math.sin(yaw), _clamp(2.5*angle_error, 1.8))

    def _person_clear(self, x: float, y: float) -> bool:
        margin = PERSON_RADIUS + .12
        if abs(x) >= ARENA_WIDTH / 2 - margin or abs(y) >= ARENA_HEIGHT / 2 - margin:
            return False
        for obstacle in self.obstacles:
            dx, dy = abs(x - obstacle["x"]), abs(y - obstacle["y"])
            if obstacle["kind"] == "circle":
                if math.hypot(dx, dy) < obstacle["r"] + margin:
                    return False
            else:
                gap_x = max(0.0, dx - obstacle["w"] / 2)
                gap_y = max(0.0, dy - obstacle["h"] / 2)
                if math.hypot(gap_x, gap_y) < margin:
                    return False
        if math.hypot(x-float(self.data.qpos[0]), y-float(self.data.qpos[1])) < 2.2:
            return False
        return all(math.hypot(x - px, y - py) > 1.3 for px, py in self._people.values())

    def _clear_approach(self, x: float, y: float) -> bool:
        """The encounter control needs a visible corridor, not a hidden spawn."""
        cx, cy = map(float, self.data.qpos[:2])
        dx, dy = x-cx, y-cy
        distance = math.hypot(dx, dy)
        direction = np.array([dx/distance, dy/distance, 0.])
        for offset in (-PERSON_RADIUS-.12, 0., PERSON_RADIUS+.12):
            origin = np.array([cx-offset*direction[1], cy+offset*direction[0], .33])
            hit = mujoco.mj_ray(self.model, self.data, origin, direction,
                               self._geom_group, True, self._car_body, self._hit_id)
            if 0 <= hit < distance:
                return False
        return True

    def add_person(self, *, ahead=False) -> bool:
        """Add a walking actor at a freshly sampled, nonoverlapping free position."""
        with self.lock:
            available = next((i for i in range(5) if i not in self._actors), None)
            if available is None:
                return False
            if ahead:
                cx, cy, yaw = map(float, self.data.qpos[:3])
                candidates = [(cx+d*math.cos(yaw)-offset*math.sin(yaw),
                               cy+d*math.sin(yaw)+offset*math.cos(yaw))
                              for d in (2.4, 2.8, 3.2) for offset in (0., -.4, .4)]
            else:
                candidates = ((self.rng.uniform(-5.8, 5.8), self.rng.uniform(-4.8, 4.8))
                              for _ in range(250))
            for x, y in candidates:
                if self._person_clear(x, y) and (not ahead or self._clear_approach(x, y)):
                    heading = math.atan2(cy-y, cx-x) if ahead else self.rng.uniform(-math.pi, math.pi)
                    self._people[available] = (x, y)
                    self._actors[available] = {"heading": heading,
                        "speed": self.rng.uniform(.22, .38), "change_at": self.rng.uniform(2., 7.)}
                    q = self._person_qpos[available]
                    self.data.qpos[q:q+3] = (x, y, heading)
                    self.data.qvel[self._person_qvel[available]:self._person_qvel[available]+3] = 0.
                    mujoco.mj_forward(self.model, self.data)
                    return True
            return False

    def clear_people(self) -> None:
        with self.lock:
            self._people.clear()
            self._actors.clear()
            self.data.ctrl[3:] = 0.
            for i, address in enumerate(self._person_qpos):
                self.data.qpos[address:address+3] = (30.+i*2, 30., 0.)
                self.data.qvel[self._person_qvel[i]:self._person_qvel[i]+3] = 0.
            mujoco.mj_forward(self.model, self.data)

    def scene(self) -> dict:
        with self.lock:
            return self._scene()

    def _scene(self) -> dict:
        x, y, yaw = map(float, self.data.qpos[:3])
        return {"width": ARENA_WIDTH, "height": ARENA_HEIGHT,
                "car": {"x": x, "y": y, "yaw": yaw, "length": LENGTH, "width": WIDTH,
                        "wheelbase": WHEELBASE, "model": "ROSMASTER R2",
                        "steering": self._steering, "wheel_angle": self._wheel_angle},
                "obstacles": [dict(item) for item in self.obstacles],
                "people": [{"id": index, "x": px, "y": py,
                            "yaw": float(self.data.qpos[self._person_qpos[index]+2])}
                           for index, (px, py) in self._people.items()],
                "scenario_id": str(self.scenario_id),
                "collision_count": self.collision_count, "engine": "MuJoCo",
                "physics": "Planar chassis and pedestrian dynamics, physical contacts, ideal Ackermann drive"}
