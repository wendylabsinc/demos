"""Go2 arena with torque-driven locomotion and raw egocentric camera frames.

The simulator supplies localization and raycast ranges, never semantic detections.
Object/person locations remain private scene state. External operators can change
scene actors. AI decisions must inspect rendered camera pixels.
"""
from __future__ import annotations

import heapq
import math
from pathlib import Path
import tempfile
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image

from ._physics.simulation import Simulation, TIMESTEP
from ._physics.sensors import instrumented_model


ASSETS = Path(__file__).resolve().parent / "assets"
MAP_ID = "wendy-go2-arena-v1"
HOME_ROOMS = {"living room": (0.0, -1.8), "kitchen": (2.5, 2.3),
              "bedroom": (-2.5, 1.9), "hallway": (0.0, -0.1)}
# Each wall is (center_x, center_y, half_width, half_depth), in metres.
HOME_WALLS = [(-4.6, 0., .1, 4.1), (4.6, 0., .1, 4.1),
              (0., -4.1, 4.7, .1), (0., 4.1, 4.7, .1),
              ]
HOME_BOUNDS = (-4.5, 4.5, -4., 4.)
FOOTPRINT = .40
GRID = .20


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _home_model(asset_dir):
    """Derive a new scene without changing the pinned model or its policy."""
    model = instrumented_model(asset_dir, sandbox=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "instrumented.xml"
        mujoco.mj_saveLastXML(str(path), model)
        scene = ET.parse(path).getroot()
    world = scene.find("worldbody")
    for i, (x, y, sx, sy) in enumerate(HOME_WALLS):
        ET.SubElement(world, "geom", name=f"home_wall_{i}", type="box", group="0",
                      pos=f"{x} {y} .7", size=f"{sx} {sy} .7", rgba=".6 .63 .65 1")
    # Scene layout is private simulator state. Sensor APIs expose only camera pixels.
    rng = np.random.default_rng()
    key_xy = rng.uniform([-3., 1.2], [3., 3.1])
    table = ET.SubElement(world, 'body', name='low_table', mocap='true', pos=f'{key_xy[0]} {key_xy[1]} .35')
    ET.SubElement(table, 'geom', type='box', size='.48 .34 .025', rgba='.43 .29 .17 1', group='0')
    for x in (-.38,.38):
        for y in (-.25,.25):
            ET.SubElement(table,'geom',type='box',pos=f'{x} {y} -.175',size='.025 .025 .175',rgba='.27 .18 .11 1',group='0')
    keys = ET.SubElement(world, 'body', name='search_object', mocap='true', pos=f'{key_xy[0]} {key_xy[1]} .60', quat='0.70710678 0 0.70710678 0')
    ET.SubElement(table, 'geom', type='capsule', fromto='0 .13 .025 0 .13 .34', size='.012', rgba='.15 .16 .17 1', group='0')
    ET.SubElement(table, 'geom', type='capsule', fromto='0 .13 .31 0 0 .31', size='.012', rgba='.15 .16 .17 1', group='0')
    # A visible steel ring, two key blades and teeth. Its color carries no label.
    for i in range(12):
        a,b=2*math.pi*i/12,2*math.pi*(i+1)/12
        ET.SubElement(keys,'geom',type='capsule',fromto=f'{.075*math.cos(a)} {.075*math.sin(a)} 0 {.075*math.cos(b)} {.075*math.sin(b)} 0',size='.009',rgba='.75 .77 .8 1',group='0',contype='0',conaffinity='0')
    for y in (-.03,.03):
        ET.SubElement(keys,'geom',type='box',pos=f'.13 {y} 0',size='.11 .014 .008',rgba='.8 .81 .83 1',group='0',contype='0',conaffinity='0')
        for x in (.15,.19,.23):
            ET.SubElement(keys,'geom',type='box',pos=f'{x} {y+.025} 0',size='.008 .025 .008',rgba='.8 .81 .83 1',group='0',contype='0',conaffinity='0')
    barrier = ET.SubElement(world, "body", name="demo_barrier", mocap="true", pos="0 3 .4")
    ET.SubElement(barrier, "geom", name="demo_barrier_geom", type="box", size=".2 .65 .4", rgba=".95 .35 .1 1", group="0")
    person = ET.SubElement(world, "body", name="person_marker", mocap="true", pos="2 -1.3 .8")
    def person_geom(name, kind, **attributes):
        ET.SubElement(person, "geom", name="person_" + name, type=kind, group="0",
                      contype="0", conaffinity="0", **attributes)
    person_geom("shirt", "capsule", fromto="0 0 -.08 0 0 .38", size=".19", rgba=".19 .32 .45 1")
    person_geom("head", "sphere", pos="0 0 .68", size=".15", rgba=".92 .72 .57 1")
    person_geom("neck", "capsule", fromto="0 0 .48 0 0 .55", size=".07", rgba=".92 .72 .57 1")
    for side, y in (("left", .13), ("right", -.13)):
        person_geom(side + "_leg", "capsule", fromto=f"0 {y} -.12 0 {y} -.67",
                    size=".08", rgba=".11 .12 .13 1")
        person_geom(side + "_shoe", "ellipsoid", pos=f"-.035 {y} -.76", size=".14 .09 .06",
                    rgba=".05 .05 .06 1")
        shoulder, hand = .25 if y > 0 else -.25, .33 if y > 0 else -.33
        person_geom(side + "_sleeve", "capsule", fromto=f"0 {shoulder} .32 0 {hand} .08",
                    size=".065", rgba=".19 .32 .45 1")
        person_geom(side + "_arm", "capsule", fromto=f"0 {hand} .08 0 {hand} -.17",
                    size=".055", rgba=".92 .72 .57 1")
    # A second room camera is unnecessary. The robot's real optical frame sees
    # only visible geometry; depth preserves wall occlusion.
    scene.find("worldbody/body[@name='base']/camera[@name='front']").set("fovy", "70")
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))


class HomePlanner:
    """A* on a supplied static map with a conservative circular footprint."""
    def __init__(self, walls=None, footprint=FOOTPRINT):
        self.walls = list(HOME_WALLS if walls is None else walls)
        self.footprint = footprint

    def free(self, x, y):
        xmin, xmax, ymin, ymax = HOME_BOUNDS
        if not xmin + self.footprint < x < xmax - self.footprint:
            return False
        if not ymin + self.footprint < y < ymax - self.footprint:
            return False
        for wx, wy, sx, sy in self.walls:
            dx = max(abs(x - wx) - sx, 0)
            dy = max(abs(y - wy) - sy, 0)
            if math.hypot(dx, dy) <= self.footprint:
                return False
        return True

    def line_free(self, first, second):
        distance = math.dist(first, second)
        count = max(1, math.ceil(distance / .04))
        return all(self.free(first[0] + (second[0] - first[0]) * i / count,
                             first[1] + (second[1] - first[1]) * i / count)
                   for i in range(count + 1))

    def route(self, start, goal):
        if not self.free(*start) or not self.free(*goal):
            raise ValueError("goal or robot footprint intersects the supplied map")
        def cell(point):
            return (round(point[0] / GRID), round(point[1] / GRID))
        def xy(node):
            return (node[0] * GRID, node[1] * GRID)
        source, target = cell(start), cell(goal)
        queue = [(math.dist(source, target), 0., source)]
        costs, previous = {source: 0.}, {}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost > costs[current]:
                continue
            if current == target:
                nodes = [target]
                while nodes[-1] != source:
                    nodes.append(previous[nodes[-1]])
                path = [start] + [xy(node) for node in reversed(nodes)][1:] + [goal]
                result = [start]
                index = 1
                while index < len(path):
                    last = index
                    while last + 1 < len(path) and self.line_free(result[-1], path[last + 1]):
                        last += 1
                    result.append(path[last])
                    index = last + 1
                return result[1:]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                next_node = (current[0] + dx, current[1] + dy)
                if not self.line_free(xy(current), xy(next_node)):
                    continue
                next_cost = cost + math.hypot(dx, dy)
                if next_cost < costs.get(next_node, float("inf")):
                    costs[next_node], previous[next_node] = next_cost, current
                    heapq.heappush(queue, (next_cost + math.dist(next_node, target), next_cost, next_node))
        raise ValueError("no route exists in the supplied home map")


class SimBackend:
    is_simulator = True

    def __init__(self, asset_dir=None):
        self.assets = Path(asset_dir) if asset_dir else ASSETS
        if not (self.assets / "policy/policy.onnx").is_file():
            raise RuntimeError("Go2 assets are missing. Run python scripts/prepare_assets.py first.")
        self.lock = threading.RLock()
        self.elapsed = 0.
        self.sim = Simulation(asset_dir=self.assets, model=_home_model(self.assets), monotonic=lambda: self.elapsed)
        self.token = self.sim.arm()
        self.planner = HomePlanner()
        self.state, self.reason = "idle", "Ready"
        self.path, self.goal = [], None
        self.last_velocity = (0., 0., 0.)
        self.wall_contacts = 0
        self.scan_remaining = 0.
        self.action = None
        self.camera_id = self.sim.model.camera("front").id
        self.person_id = self.sim.model.body("person_marker").mocapid[0]
        self.key_id = int(self.sim.model.body("search_object").mocapid[0])
        self.table_id = int(self.sim.model.body("low_table").mocapid[0])
        self.capture_id = 0
        self.sim.model.vis.global_.offwidth = 960
        self.sim.model.vis.global_.offheight = 540
        self.renderer = None
        self.overview_renderer = None
        # OpenGL contexts must always be created, used and destroyed by the same
        # thread. Web requests and the physics timer can arrive on other threads.
        self._render_thread = threading.get_ident()
        self.rgb = self.depth = None
        self.detections = []
        self.capture_time = 0.
        self.last_capture = -float("inf")
        self.overview = None
        self.camera_healthy = True
        self.sensor_fault = False
        # Give contacts and the standing controller time to settle.
        for _ in range(250):
            self._step((0., 0., 0.))
        self._capture()

    def _pose(self):
        rotation = self.sim.data.xmat[self.sim.body_id].reshape(3, 3)
        return {"x": float(self.sim.data.qpos[0]), "y": float(self.sim.data.qpos[1]),
                "yaw": math.atan2(rotation[1, 0], rotation[0, 0])}

    def observe(self):
        with self.lock:
            if self.rgb is None or (self.elapsed - self.last_capture >= .2
                                    and threading.get_ident() == self._render_thread):
                self._capture()
            healthy = self.sim.mode not in {"fault", "fallen", "damping"} and self.camera_healthy and not self.sensor_fault
            return {"map_id": MAP_ID, "timestamp": time.monotonic(), "pose": self._pose(),
                    "detections": [dict(value) for value in self.detections] if healthy else [],
                    "healthy": healthy, "reason": self.reason if healthy else "Simulator sensor or locomotion fault",
                    "localization": "exact MuJoCo pose, simulator assumption",
                    "map_source": "supplied static home floorplan"}

    def navigate(self, x, y):
        with self.lock:
            if not all(math.isfinite(value) for value in (x, y)):
                raise ValueError("navigation goal must be finite")
            if self.sensor_fault or not self.camera_healthy or self.sim.mode not in {"standing", "moving"}:
                if self.sim.mode == "lying":
                    raise RuntimeError("Go2 is lying down. Ask it to stand up before navigating.")
                raise RuntimeError("simulator is not ready for motion")
            pose = self._pose()
            route = self.planner.route((pose["x"], pose["y"]), (x, y))
            self.path, self.goal = route, (float(x), float(y))
            self.state, self.reason = "navigating", "Following a route through the supplied map"
            self.scan_remaining = 0.
            self.action = None

    def scan(self):
        with self.lock:
            if self.sensor_fault or not self.camera_healthy or self.sim.mode not in {"standing", "moving"}:
                raise RuntimeError("simulator sensors are unavailable")
            self.path, self.goal = [], None
            self.scan_remaining = 2 * math.pi
            self.state, self.reason = "navigating", "Looking around with the robot camera"
            self.action = None

    def perform_action(self, action):
        if action == "spin":
            return self.spin(turns=1.)
        if action == "lie_down":
            return self.rest()
        if action == "stand_up":
            return self.stand()
        return {"accepted": False, "reason": f"Unsupported Go2 action: {action}"}

    def rest(self):
        """Lower to the pinned Unitree lying pose through actuator torques."""
        return self._posture_action("lie_down")

    def stand(self):
        return self._posture_action("stand_up")

    def _posture_action(self, action):
        with self.lock:
            if (self.sensor_fault or not self.camera_healthy or self.sim.owner != self.token
                    or self.sim.mode in {"fallen", "damping", "fault", "paused", "lowlevel"}):
                return {"accepted": False, "reason": "Go2 is not ready for a posture change"}
            if self.sim.mode in {"standing_up", "standing_down"}:
                return {"accepted": False, "reason": "Go2 is already settling a posture change safely"}
            self.stop()
            try:
                if action == "lie_down":
                    self.sim.stand_down(self.token)
                else:
                    self.sim.stand_up(self.token)
            except (ValueError, RuntimeError, PermissionError) as error:
                return {"accepted": False, "reason": str(error)}
            target = "lying" if action == "lie_down" else "standing"
            self.action = {"name": action, "target": target, "started": self.elapsed}
            self.state, self.reason = "navigating", ("Lying down" if action == "lie_down" else "Standing up")
            self._check_action()
            return {"accepted": True, "action": action, "posture": target}

    def spin(self, turns=.5, direction="left"):
        with self.lock:
            if (type(turns) not in (int, float) or not math.isfinite(turns)
                    or not 0 < turns <= 1 or direction not in {"left", "right"}):
                return {"accepted": False, "reason": "Spin requires 0 < turns <= 1 and left or right direction"}
            if self.sensor_fault or not self.camera_healthy or self.sim.mode not in {"standing", "moving"}:
                return {"accepted": False, "reason": "Go2 must be standing with healthy sensors before spinning"}
            pose = self._pose()
            if not self.planner.free(pose["x"], pose["y"]):
                return {"accepted": False, "reason": "Go2 needs room for its turning footprint"}
            self.stop()
            self.action = {"name": "spin", "remaining": 2 * math.pi * turns,
                           "direction": 1 if direction == "left" else -1,
                           "started": self.elapsed}
            self.state, self.reason = "navigating", "Turning in place"
            return {"accepted": True, "action": "spin", "turns": turns, "direction": direction}

    def stop(self):
        with self.lock:
            self.path, self.goal = [], None
            self.scan_remaining = 0.
            self.action = None
            settling = self.sim.mode in {"standing_up", "standing_down"}
            self.state, self.reason = "idle", ("Stopped motion; the posture transition is settling safely" if settling else "Stopped")
            if self.sim.owner == self.token:
                self.sim.stop(self.token)
            self.last_velocity = (0., 0., 0.)

    def set_person(self, x, y):
        """Move only the test person marker; this is not a navigation observation."""
        with self.lock:
            if (type(x) not in (int, float) or type(y) not in (int, float)
                    or not math.isfinite(x) or not math.isfinite(y) or not self.planner.free(x, y)):
                raise ValueError("person marker must be inside free floor space")
            self.sim.data.mocap_pos[self.person_id] = [x, y, .8]
            mujoco.mj_forward(self.sim.model, self.sim.data)
            self.last_capture = -float("inf")

    def get_person(self):
        with self.lock:
            position = self.sim.data.mocap_pos[self.person_id]
            return {"x": float(position[0]), "y": float(position[1])}

    def search_points(self, room=None):
        if room:
            if room not in HOME_ROOMS:
                raise ValueError("unknown room")
            return [HOME_ROOMS[room]]
        return [HOME_ROOMS[name] for name in ("living room", "kitchen", "bedroom")]

    def _clearance(self):
        """Actual MuJoCo range rays in the robot's forward body-height band."""
        rotation = self.sim.data.xmat[self.sim.body_id].reshape(3, 3)
        origin = self.sim.data.site_xpos[self.sim.model.site("lidar").id]
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        geom = np.empty(1, dtype=np.int32)
        distances = []
        for angle in np.linspace(-.45, .45, 13):
            ray = rotation @ np.array([math.cos(angle), math.sin(angle), 0.])
            distance = mujoco.mj_ray(self.sim.model, self.sim.data, origin, ray, group, True, self.sim.body_id, geom)
            if 0 < distance <= 12:
                distances.append(distance)
        return min(distances, default=12.)

    def _velocity(self):
        if self.sensor_fault or self.sim.mode in {"fallen", "fault", "damping"}:
            self.state, self.reason = "fault", "Simulator sensor or locomotion fault"
            self.path = []
            self.scan_remaining = 0.
            self.action = None
            return (0., 0., 0.)
        if self.action:
            if self.action["name"] == "spin" and self.action.get("phase") != "stopping":
                return (0., 0., .5 * self.action["direction"])
            return (0., 0., 0.)
        if self.scan_remaining > 0:
            return (0., 0., .5)
        if not self.path:
            return (0., 0., 0.)
        pose = self._pose()
        while self.path and math.dist((pose["x"], pose["y"]), self.path[0]) < .22:
            self.path.pop(0)
        if not self.path:
            self.state, self.reason = "arrived", "Reached the requested location"
            return (0., 0., 0.)
        x, y = self.path[0]
        error = _angle(math.atan2(y - pose["y"], x - pose["x"]) - pose["yaw"])
        if abs(error) > .18:
            return (0., 0., max(-.5, min(.5, 1.5 * error)))
        if self._clearance() < .50:
            self.state, self.reason = "blocked", "Lidar reports an obstacle in the route"
            return (0., 0., 0.)
        self.state = "navigating"
        return (.55, 0., max(-.4, min(.4, 1.2 * error)))

    def _step(self, velocity):
        self.elapsed += TIMESTEP
        if self.sim.owner == self.token and self.sim.mode in {"standing", "moving"}:
            self.sim.command_velocity(*velocity, self.token)
        self.sim.step()
        for contact in self.sim.data.contact:
            if any((mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or "").startswith("home_wall_")
                   for geom in (contact.geom1, contact.geom2)):
                self.wall_contacts += 1

    def _check_action(self):
        if self.action is None:
            return
        action = self.action
        timeout = 20. if action["name"] == "spin" else 8.
        if self.elapsed - action["started"] > timeout:
            self.action = None
            self.state, self.reason = "fault", "The bounded action timed out and motion stopped"
            if self.sim.owner == self.token:
                self.sim.stop(self.token)
            return
        if action["name"] == "spin":
            if action["remaining"] <= .06:
                action["phase"] = "stopping"
                self.sim.stop(self.token)
                if abs(self.sim.data.qvel[5]) < .12 and np.linalg.norm(self.sim.applied_command) < .01:
                    self.action = None
                    self.state, self.reason = "arrived", "Finished the bounded turn and stopped"
        elif self.sim.mode == action["target"]:
            self.action = None
            self.state, self.reason = "arrived", ("Lying down and holding the resting posture"
                                                   if action["target"] == "lying" else "Standing and ready")

    def tick(self, dt=.1):
        if not math.isfinite(dt) or not 0 < dt <= 1:
            raise ValueError("tick must advance between zero and one simulation second")
        with self.lock:
            velocity = self._velocity()
            for index in range(round(dt / TIMESTEP)):
                if index % 25 == 0:
                    velocity = self._velocity()
                old_yaw = self._pose()["yaw"]
                self._step(velocity)
                if self.action and self.action["name"] == "spin":
                    delta = _angle(self._pose()["yaw"] - old_yaw) * self.action["direction"]
                    self.action["remaining"] -= max(0., delta)
                self._check_action()
                if self.scan_remaining > 0:
                    self.scan_remaining -= max(0., _angle(self._pose()["yaw"] - old_yaw))
                    if self.scan_remaining <= 0:
                        self.state, self.reason = "arrived", "Finished looking around"
                self.last_velocity = tuple(float(value) for value in self.sim.command)
            if self.elapsed - self.last_capture >= .2:
                self._capture()

    def _capture(self):
        return self._render_call(self._capture_owned)

    def _render_call(self, callback):
        if threading.get_ident() != self._render_thread:
            raise RuntimeError("camera rendering must run on the thread that created the simulator")
        return callback()

    def _capture_owned(self):
        if self.sensor_fault:
            self.detections = []
            return
        try:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.sim.model, height=360, width=640)
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.sim.data, camera="front")
            self.rgb = self.renderer.render().copy()
            self.capture_time = time.monotonic()
            self.detections = []
            self.capture_id += 1
            self.last_capture = self.elapsed
            self.camera_healthy = True
        except Exception as error:
            self.camera_healthy = False
            self.reason = f"Camera renderer failed: {error}"
            self.detections = []
            raise RuntimeError(self.reason) from error

    def frame(self, view="overview"):
        with self.lock:
            if view == "robot":
                if self.rgb is None:
                    self._capture()
                return Image.fromarray(self.rgb.copy())
            if view != "overview":
                raise ValueError("view must be overview or robot")
            return self._render_call(self._overview_owned)

    def _overview_owned(self):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0., 0., 0.]
        camera.distance, camera.azimuth, camera.elevation = 13., 90., -65.
        if self.overview_renderer is None:
            self.overview_renderer = mujoco.Renderer(self.sim.model, height=540, width=960)
        self.overview_renderer.update_scene(self.sim.data, camera=camera)
        return Image.fromarray(self.overview_renderer.render().copy())

    def status(self):
        with self.lock:
            return {"state": self.state, "reason": self.reason, "pose": self._pose(),
                    "goal": self.goal, "route": list(self.path), "velocity": self.last_velocity,
                    "sim_seconds": self.elapsed, "wall_contacts": self.wall_contacts,
                    "map_id": MAP_ID, "map_source": "live raycast observations; room geometry only for fixture validation",
                    "perception": "raw egocentric RGB camera; no semantic detector", "localization": "exact simulated pose",
                    "posture": self.sim.mode,
                    "action": dict(self.action) if self.action else None}

    def close(self):
        with self.lock:
            self.stop()
            self._render_call(self._close_owned)

    def _close_owned(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.overview_renderer:
            self.overview_renderer.close()
            self.overview_renderer = None
