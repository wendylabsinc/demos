"""Torque-driven Go2 with measured lidar mapping and learned RGB-D vision.

The scene builder is simulator ground truth only. Robot decisions use range
returns, biased incremental body odometry, scan matching, and pixel recognition.
No fixture coordinates, supplied floorplan, or exact world pose enter control.
"""
from __future__ import annotations

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
from ._physics.sensors import instrumented_model, CAMERA_POSITION
from .mapping import OccupancyMapper
from .perception import ModelPerception


ASSETS = Path(__file__).resolve().parents[1] / "assets"
SCENE_ASSETS = Path(__file__).resolve().parent / "scene_assets"
OVERVIEW_SIZE = (720, 405)
CAMERA_SIZE = (640, 360)
DISPLAY_CAMERA_SIZE = (320, 180)
CAMERA_PITCH = math.radians(10.)
PERSON_WALK_SPEED = .5
LIDAR_ORIGIN = (.28, 0.)
LIDAR_MAX = 8.
LIDAR_ANGLES = np.linspace(-math.pi, math.pi, 360, endpoint=False)


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _cup_meshes(asset):
    """Continuous ceramic shell and handle surfaces, with shared smooth vertices."""
    def mesh(name, vertices, faces):
        ET.SubElement(asset, "mesh", name=name,
                      vertex=" ".join(str(value) for row in vertices for value in row),
                      face=" ".join(str(value) for row in faces for value in row))
    segments = 64
    profiles = ((.076, .0), (.078, .155), (.063, .155), (.062, .018))
    vertices = [(r*math.cos(2*math.pi*i/segments), r*math.sin(2*math.pi*i/segments), z)
                for r,z in profiles for i in range(segments)]
    faces = []
    for ring in range(len(profiles)-1):
        for i in range(segments):
            j = (i+1)%segments
            a,b,c,d=ring*segments+i,ring*segments+j,(ring+1)*segments+j,(ring+1)*segments+i
            faces.extend(((a,b,c),(a,c,d)))
    inner_center, outer_center = len(vertices), len(vertices)+1
    vertices.extend(((0.,0.,.018),(0.,0.,0.)))
    for i in range(segments):
        j=(i+1)%segments
        faces.extend(((inner_center,3*segments+i,3*segments+j),(outer_center,j,i)))
    mesh("ceramic_shell", vertices, faces)
    vertices, faces = [], []
    for i in range(48):
        a=2*math.pi*i/48
        for j in range(12):
            b=2*math.pi*j/12
            vertices.append((.012*math.sin(b), .075+(.059+.012*math.cos(b))*math.cos(a),
                             .083+(.059+.012*math.cos(b))*math.sin(a)))
    for i in range(48):
        for j in range(12):
            a,b,c,d=i*12+j,((i+1)%48)*12+j,((i+1)%48)*12+(j+1)%12,i*12+(j+1)%12
            faces.extend(((a,b,c),(a,c,d)))
    mesh("ceramic_handle", vertices, faces)


def _scene_model(asset_dir):
    """Build physical fixtures. This data is never a map or perception input."""
    model = instrumented_model(asset_dir, sandbox=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "instrumented.xml"
        mujoco.mj_saveLastXML(str(path), model)
        scene = ET.parse(path).getroot()
    world = scene.find("worldbody")
    asset = scene.find("asset")
    if asset is None:
        asset = ET.SubElement(scene, "asset")
    person_assets = SCENE_ASSETS / "person"
    ET.SubElement(asset, "texture", name="human_texture", type="2d", file=str(person_assets / "person.png"))
    ET.SubElement(asset, "material", name="human_clothing", texture="human_texture", texuniform="false")
    ET.SubElement(asset, "mesh", name="human", file=str(person_assets / "person_body.obj"))

    def geom(name, kind, pos=None, size=None, rgba=".6 .63 .65 1", parent=world, **kw):
        attributes = dict(name=name, type=kind, group="0", rgba=rgba, **kw)
        if pos is not None:
            attributes["pos"] = " ".join(map(str, pos))
        if size is not None:
            attributes["size"] = " ".join(map(str, size))
        return ET.SubElement(parent, "geom", **attributes)

    _cup_meshes(asset)

    # Fixture layout belongs solely to scene generation and operator rendering.
    walls = [(-4.6, 0., .1, 4.1), (4.6, 0., .1, 4.1),
             (0., -4.1, 4.7, .1), (0., 4.1, 4.7, .1),
             (-3.75, .8, .75, .08), (0., .8, 1.4, .08),
             (3.75, .8, .75, .08), (0., 2.5, .08, 1.5)]
    for i, (x, y, sx, sy) in enumerate(walls):
        geom(f"home_wall_{i}", "box", (x, y, .95), (sx, sy, .95), ".78 .77 .73 1")
    # Living room: a recognizable sofa, coffee table, rug and hollow mug.
    geom("sofa_seat", "box", (-3.6, -2.3, .36), (.55, 1.05, .16), ".26 .31 .33 1")
    geom("sofa_back", "box", (-4.12, -2.3, .65), (.12, 1.05, .48), ".26 .31 .33 1")
    for y in (-3.3, -1.3):
        geom("sofa_arm_" + str(y), "box", (-3.6, y, .56), (.55, .13, .28), ".26 .31 .33 1")
    geom("living_rug", "box", (-1.9, -2.3, .004), (1.0, 1.25, .004), ".43 .40 .34 1", contype="0", conaffinity="0")
    geom("coffee_table_top", "box", (2.7, -.15, .42), (.6, .38, .045), ".48 .29 .16 1")
    for i, (dx, dy) in enumerate(((-.48, -.26), (-.48, .26), (.48, -.26), (.48, .26))):
        geom(f"coffee_table_leg_{i}", "box", (2.7+dx, -.15+dy, .20), (.045, .045, .20), ".29 .17 .1 1")
    cup = ET.SubElement(world, "body", name="ceramic_cup", pos="2.5 -.15 .47")
    for part in ("shell", "handle"):
        ET.SubElement(cup, "geom", name="cup_"+part, type="mesh", mesh="ceramic_"+part,
                      group="0", rgba=".88 .86 .8 1", contype="0", conaffinity="0")
    # Kitchen: cabinets, counter, sink, hob and tall fridge. Their surfaces are
    # visible to the learned vision model; no room label is embedded in pixels.
    geom("kitchen_cabinets", "box", (3.7, 3.52, .43), (.72, .48, .43), ".61 .45 .3 1")
    geom("kitchen_counter", "box", (3.7, 3.52, .9), (.76, .5, .045), ".79 .78 .72 1")
    geom("kitchen_fridge", "box", (3.85, 1.8, .9), (.47, .48, .9), ".77 .79 .8 1")
    geom("fridge_handle", "capsule", size=(.025,), fromto="3.33 1.6 .9 3.33 1.6 1.4", rgba=".18 .19 .2 1")
    geom("sink_basin", "box", (3.36, 3.47, .951), (.23, .27, .009), ".18 .20 .21 1")
    geom("sink_tap", "capsule", size=(.023,), fromto="3.36 3.75 .95 3.36 3.75 1.2", rgba=".63 .65 .66 1")
    geom("sink_spout", "capsule", size=(.023,), fromto="3.36 3.75 1.2 3.36 3.51 1.2", rgba=".63 .65 .66 1")
    geom("hob", "box", (4.05, 3.47, .955), (.25, .29, .008), ".08 .09 .10 1")
    for i, (dx, dy) in enumerate(((-.13,-.15),(-.13,.15),(.13,-.15),(.13,.15))):
        geom(f"hob_ring_{i}", "cylinder", (4.05+dx,3.47+dy,.968), (.08,.006), ".34 .35 .36 1")
    bottle = ET.SubElement(world, "body", name="water_bottle", pos="2.95 3.27 .95")
    geom("bottle_body", "cylinder", (0,0,.15), (.07,.15), ".40 .49 .48 1", parent=bottle)
    geom("bottle_shoulder", "ellipsoid", (0,0,.30), (.07,.07,.055), ".40 .49 .48 1", parent=bottle)
    geom("bottle_neck", "cylinder", (0,0,.35), (.027,.045), ".40 .49 .48 1", parent=bottle)
    geom("bottle_cap", "cylinder", (0,0,.40), (.032,.017), ".18 .2 .2 1", parent=bottle)
    # Bedroom: bed, mattress, pillows, nightstand and real key-ring silhouette.
    geom("bed_frame", "box", (-3.35,2.9,.25), (.95,.95,.22), ".37 .25 .16 1")
    geom("bed_mattress", "box", (-3.35,2.9,.54), (.9,.92,.12), ".86 .86 .79 1")
    geom("bed_headboard", "box", (-3.35,3.9,.63), (.97,.08,.63), ".37 .25 .16 1")
    geom("bed_blanket", "box", (-3.35,2.64,.67), (.9,.66,.025), ".28 .39 .43 1")
    for i,x in enumerate((-3.8,-2.9)):
        geom(f"bed_pillow_{i}", "ellipsoid", (x,3.48,.72), (.38,.25,.08), ".9 .9 .87 1")
    geom("nightstand", "box", (-1.55,3.4,.34), (.3,.34,.34), ".47 .32 .19 1")
    keys = ET.SubElement(world, "body", name="house_keys", pos="-1.55 3.4 .70")
    for i in range(16):
        a,b=2*math.pi*i/16,2*math.pi*(i+1)/16
        geom(f"key_ring_{i}", "capsule", size=(.007,), parent=keys,
             fromto=f"{.07*math.cos(a)} {.07*math.sin(a)} 0 {.07*math.cos(b)} {.07*math.sin(b)} 0", rgba=".68 .68 .63 1")
    for i,y in enumerate((-.038,.038)):
        geom(f"key_blade_{i}", "box", (.14,y,.003), (.09,.015,.009), ".67 .67 .59 1", parent=keys)
        for j in range(3):
            geom(f"key_tooth_{i}_{j}", "box", (.11+.04*j,y+.023,.003), (.010,.014,.009), ".67 .67 .59 1", parent=keys)
    person = ET.SubElement(world, "body", name="scene_person", mocap="true", pos="2 -.8 0", quat="0 0 0 1")
    ET.SubElement(person,"geom",name="human_visual",type="mesh",mesh="human",material="human_clothing",
                  group="0",contype="0",conaffinity="0")
    # The licensed mannequin body has lowered arms and no helmet. Anatomical
    # features are ordinary visible geometry, with no labels supplied to vision.
    skin = ".72 .51 .38 1"
    geom("human_neck", "capsule", size=(.055,), parent=person,
         fromto="-.02 0 1.34 -.02 0 1.43", rgba=skin, contype="0", conaffinity="0")
    geom("human_head", "ellipsoid", (-.02,0,1.535), (.105,.112,.155), skin,
         parent=person, contype="0", conaffinity="0")
    geom("human_hair", "ellipsoid", (-.046,0,1.655), (.102,.117,.051), ".075 .045 .025 1",
         parent=person, contype="0", conaffinity="0")
    geom("human_nose", "ellipsoid", (.085,0,1.545), (.032,.022,.029), skin,
         parent=person, contype="0", conaffinity="0")
    for side, y in (("left", .046),("right",-.046)):
        geom("human_eye_"+side, "ellipsoid", (.072,y,1.575), (.016,.021,.012), ".92 .90 .85 1",
             parent=person, contype="0", conaffinity="0")
        geom("human_pupil_"+side, "ellipsoid", (.085,y,1.576), (.005,.007,.008), ".08 .06 .045 1",
             parent=person, contype="0", conaffinity="0")
        geom("human_ear_"+side, "ellipsoid", (-.025,math.copysign(.112,y),1.537), (.022,.015,.034), skin,
             parent=person, contype="0", conaffinity="0")
    geom("human_mouth", "capsule", size=(.005,), parent=person,
         fromto=".076 -.026 1.488 .076 .026 1.488", rgba=".30 .13 .11 1", contype="0", conaffinity="0")
    camera = scene.find("worldbody/body[@name='base']/camera[@name='front']")
    camera.set("fovy", "70")
    camera.attrib.pop('quat', None)
    camera.set("xyaxes", f"0 -1 0 {-math.sin(CAMERA_PITCH)} 0 {math.cos(CAMERA_PITCH)}")
    return mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))


class SimBackend:
    is_simulator = True
    sensor_mapping = True

    def __init__(self, asset_dir=None, *, mapping_path=None, mapper=None, perception=None):
        self.assets = Path(asset_dir) if asset_dir else ASSETS
        if not (self.assets / "policy/policy.onnx").is_file():
            raise RuntimeError("Go2 assets are missing. Run python scripts/prepare_assets.py first.")
        self.lock = threading.RLock()
        self.elapsed = 0.
        self.sim = Simulation(asset_dir=self.assets, model=_scene_model(self.assets), monotonic=lambda: self.elapsed)
        self.token = self.sim.arm()
        self.mapping_path = Path(mapping_path) if mapping_path else None
        self.mapper = mapper or (OccupancyMapper.load(self.mapping_path)
                                 if self.mapping_path and self.mapping_path.exists() else OccupancyMapper())
        self.map_id = self.mapper.map_id
        self.perception = perception if perception is not None else ModelPerception()
        self._odom_delta = np.zeros(3, dtype=np.float64)
        self._last_scan = -float("inf")
        self._scan_ranges = np.full(len(LIDAR_ANGLES), LIDAR_MAX, dtype=np.float32)
        self._sensor_time = 0.
        self._gyro_yaw_delta = 0.
        self._map_saved = 0.
        self.state, self.reason = "idle", "Ready"
        self.path, self.goal = [], None
        self._route_revision = -1
        self.last_velocity = (0., 0., 0.)
        self.wall_contacts = 0
        self.scan_remaining = 0.
        self.action = None
        self.camera_id = self.sim.model.camera("front").id
        self.person_id = self.sim.model.body("scene_person").mocapid[0]
        self._person_target = None
        self.sim.model.vis.global_.offwidth = OVERVIEW_SIZE[0]
        self.sim.model.vis.global_.offheight = OVERVIEW_SIZE[1]
        self.sim.model.vis.quality.offsamples = 0
        self.renderer = None
        self.overview_renderer = None
        self.live_renderer = None
        # OpenGL contexts must always be created, used and destroyed by the same
        # thread. Web requests and the physics timer can arrive on other threads.
        self._render_thread = threading.get_ident()
        self._visual_revision = 0
        self._frame_cache = {}
        self._render_counts = {"overview": 0, "robot": 0, "perception": 0}
        self.rgb = self.depth = None
        self.capture_time = 0.
        self.last_capture = -float("inf")
        self.overview = None
        self.camera_healthy = True
        self.sensor_fault = False
        # Give contacts and the standing controller time to settle.
        for _ in range(250):
            self._step((0., 0., 0.))
        self._odom_delta[:] = 0.
        self._sample_lidar()
        self._capture()
        # First renderer/model initialization can be slower than normal. Take
        # another real scan when necessary before exposing a ready backend.
        if time.monotonic() - self._sensor_time >= .1:
            self._sample_lidar()

    def _pose(self):
        return self.mapper.pose()

    def observe(self):
        with self.lock:
            if self.rgb is None or (self.elapsed - self.last_capture >= .2
                                    and threading.get_ident() == self._render_thread):
                self._capture()
            vision = self.perception.poll()
            localized = self.mapper.status()["localization"]["valid"]
            healthy = (self.sim.mode not in {"fault", "fallen", "damping"} and self.camera_healthy
                       and not self.sensor_fault and localized)
            return {"map_id": self.map_id, "timestamp": self._sensor_time, "pose": self._pose(),
                    "detections": vision.get("detections", []) if healthy else [],
                    "room": vision.get("room") if healthy else None, "vision": self.perception.status(),
                    "healthy": healthy,
                    "reason": self.reason if healthy else "Measured sensor or localization unavailable",
                    "localization": self.mapper.status()["localization"],
                    "map_source": "measured lidar rays"}

    def map_snapshot(self):
        return self.mapper.snapshot()

    def set_search_target(self, label):
        self.perception.set_target(label)

    def navigate(self, x, y):
        with self.lock:
            if not all(math.isfinite(value) for value in (x, y)):
                raise ValueError("navigation goal must be finite")
            if self.sensor_fault or not self.camera_healthy or self.sim.mode not in {"standing", "moving"}:
                if self.sim.mode == "lying":
                    raise RuntimeError("Go2 is lying down. Ask it to stand up before navigating.")
                raise RuntimeError("simulator is not ready for motion")
            pose = self._pose()
            route = self.mapper.plan(x, y)
            if not route:
                raise ValueError("No footprint-safe route exists in measured free space")
            self.path, self.goal = route, (float(x), float(y))
            self._route_revision = self.mapper.revision
            self.state, self.reason = "navigating", "Following a route through measured free space"
            self.scan_remaining = 0.
            self.action = None

    def scan(self):
        with self.lock:
            if self.sensor_fault or not self.camera_healthy or self.sim.mode not in {"standing", "moving"}:
                raise RuntimeError("simulator sensors are unavailable")
            self.path, self.goal = [], None
            self.scan_remaining = .8
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
            if not self.mapper.free(pose["x"], pose["y"]):
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
            return {"accepted": True, "settling": settling, "reason": self.reason}

    def set_person(self, x, y):
        """Move the operator actor continuously; never a robot observation."""
        with self.lock:
            if (type(x) not in (int, float) or type(y) not in (int, float)
                    or not math.isfinite(x) or not math.isfinite(y) or not self._scene_floor_free(x, y)):
                raise ValueError("person must be inside clear scene floor space")
            position = self.sim.data.mocap_pos[self.person_id, :2].copy()
            target = np.array([float(x), float(y)])
            distance = float(np.linalg.norm(target - position))
            samples = max(1, math.ceil(distance / .08))
            if not all(self._scene_floor_free(*(position + (target-position)*index/samples))
                       for index in range(1, samples+1)):
                raise ValueError("person movement requires a clear straight floor corridor")
            self._person_target = target if distance > 1e-8 else None

    def _advance_person(self, dt):
        """Only an explicit operator target causes actor movement, at 0.5m/s."""
        if self._person_target is None:
            return
        position = self.sim.data.mocap_pos[self.person_id, :2]
        delta = self._person_target - position
        distance = float(np.linalg.norm(delta))
        maximum = PERSON_WALK_SPEED * dt
        if distance <= maximum:
            position[:] = self._person_target
            self._person_target = None
        elif distance > 0.:
            position[:] += delta * (maximum / distance)
        self._visual_revision += 1

    def _scene_floor_free(self, x, y):
        # Only operator placement consults physical world geometry. Navigation
        # and vision never call this method or receive its coordinates.
        if not (-4.3 < x < 4.3 and -3.8 < y < 3.8):
            return False
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        geom = np.empty(1, dtype=np.int32)
        for angle in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            distance = mujoco.mj_ray(self.sim.model, self.sim.data, np.array([x, y, .4]),
                       np.array([math.cos(angle), math.sin(angle), 0.]), group, True,
                       self.sim.model.body("scene_person").id, geom)
            if 0 <= distance < .25:
                return False
        return True

    def get_person(self):
        with self.lock:
            position = self.sim.data.mocap_pos[self.person_id]
            return {"x": float(position[0]), "y": float(position[1])}

    def search_points(self, room=None):
        # A room name cannot reveal a location. Only reachable frontiers in the
        # sensor-built map become exploration viewpoints.
        return self.mapper.frontiers(limit=20)

    def _sample_lidar(self):
        """Stabilized planar range sensor plus incremental wheel/IMU analogue.

        MuJoCo world state is used only to synthesize physical sensor readings.
        Scan matching receives body-frame ranges and biased velocity integrals,
        never an absolute world position or heading.
        """
        rotation = self.sim.data.xmat[self.sim.body_id].reshape(3, 3)
        sensor_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        origin = self.sim.data.site_xpos[self.sim.model.site("lidar").id].copy()
        group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        geom = np.empty(1, dtype=np.int32)
        ranges = np.full(len(LIDAR_ANGLES), LIDAR_MAX, dtype=np.float32)
        for index, angle in enumerate(LIDAR_ANGLES):
            heading = sensor_yaw + angle
            ray = np.array([math.cos(heading), math.sin(heading), 0.])
            distance = mujoco.mj_ray(self.sim.model, self.sim.data, origin, ray, group, True,
                                     self.sim.body_id, geom)
            if .1 <= distance < LIDAR_MAX:
                ranges[index] = distance
        self._sensor_time = time.monotonic()
        self._scan_ranges = ranges
        self.mapper.update_scan(ranges, LIDAR_ANGLES, tuple(self._odom_delta),
                                max_range=LIDAR_MAX, timestamp=self._sensor_time,
                                lidar_origin=LIDAR_ORIGIN)
        self._odom_delta[:] = 0.
        self._last_scan = self.elapsed
        self._refresh_route()
        if self.mapping_path and self.elapsed - self._map_saved > 5.:
            self.mapper.save(self.mapping_path)
            self._map_saved = self.elapsed

    def _refresh_route(self):
        """Every measured-map revision can invalidate a previously accepted route."""
        if not self.path or self.goal is None or self._route_revision == self.mapper.revision:
            return
        pose = self._pose()
        start = (pose["x"], pose["y"])
        segments = list(zip([start] + self.path[:-1], self.path))
        if not all(self.mapper.line_free(first, second) for first, second in segments):
            self.path = self.mapper.plan(*self.goal)
            if not self.path:
                self.state, self.reason = "blocked", "New range measurements closed the route; stopped"
                self.last_velocity = (0., 0., 0.)
                if self.sim.owner == self.token:
                    self.sim.stop(self.token)
            else:
                self.reason = "Replanned around newly measured obstacles"
        self._route_revision = self.mapper.revision

    def _clearance(self):
        return float(np.min(self._scan_ranges[np.abs(LIDAR_ANGLES) <= .45]))

    def _velocity(self):
        if (self.sensor_fault or self.sim.mode in {"fallen", "fault", "damping"}
                or not self.mapper.status()["localization"]["valid"]):
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
        self._refresh_route()
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
        self._advance_person(TIMESTEP)
        self.sim.step()
        rotation = self.sim.data.xmat[self.sim.body_id].reshape(3, 3)
        body_velocity = rotation.T @ self.sim.data.qvel[:3]
        gyro = self.sim.data.sensor("imu_gyro").data
        dyaw = (float(gyro[2]) + .0008) * TIMESTEP
        c, sine = math.cos(self._odom_delta[2]), math.sin(self._odom_delta[2])
        forward = float(body_velocity[0]) * 1.015 * TIMESTEP
        left = (float(body_velocity[1]) + .0005) * TIMESTEP
        self._odom_delta[0] += c * forward - sine * left
        self._odom_delta[1] += sine * forward + c * left
        self._odom_delta[2] += dyaw
        self._gyro_yaw_delta = dyaw
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
                if abs(self.sim.data.sensor("imu_gyro").data[2]) < .12 and np.linalg.norm(self.sim.applied_command) < .01:
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
                self._step(velocity)
                if self.action and self.action["name"] == "spin":
                    delta = self._gyro_yaw_delta * self.action["direction"]
                    self.action["remaining"] -= max(0., delta)
                self._check_action()
                if self.scan_remaining > 0:
                    self.scan_remaining -= max(0., self._gyro_yaw_delta)
                    if self.scan_remaining <= 0:
                        self.state, self.reason = "arrived", "Finished looking around"
                self.last_velocity = tuple(float(value) for value in self.sim.command)
            if self.elapsed - self.last_capture >= .2:
                self._capture()
            # Rendering and perception submission can take time. Acquire the
            # actual navigation scan after that work, so its capture timestamp
            # remains current when the shared controller validates it. Never
            # refresh a timestamp without taking another measurement.
            if not self.sensor_fault and (self.elapsed - self._last_scan >= .1
                                          or time.monotonic() - self._sensor_time >= .1):
                self._sample_lidar()

    def _capture(self):
        return self._render_call(self._capture_owned)

    def _render_call(self, callback):
        if threading.get_ident() != self._render_thread:
            raise RuntimeError("camera rendering must run on the thread that created the simulator")
        return callback()

    def _capture_owned(self):
        if self.sensor_fault:
            return
        try:
            self.rgb = self._robot_rgb_owned()
            self.renderer.enable_depth_rendering()
            self.renderer.update_scene(self.sim.data, camera="front")
            self._display_options(self.renderer)
            self.depth = self.renderer.render().copy()
            self.renderer.disable_depth_rendering()
            self.capture_time = time.monotonic()
            self.perception.submit(self.rgb, self.depth, self._camera_calibration(), self.capture_time)
            self.last_capture = self.elapsed
            self.camera_healthy = True
            self._render_counts["perception"] += 1
            self._frame_cache["robot"] = (self._frame_key(), Image.fromarray(self.rgb).resize(
                DISPLAY_CAMERA_SIZE, Image.Resampling.BILINEAR))
        except Exception as error:
            self.camera_healthy = False
            self.reason = f"Camera renderer failed: {error}"
            raise RuntimeError(self.reason) from error

    def _camera_calibration(self):
        """Fixed mount + estimated map pose + simulated IMU roll/pitch.

        Vertical base height .34m and sensor extrinsics are simulator calibration
        assumptions. Absolute MuJoCo camera/world pose is never passed to vision.
        """
        pose = self._pose()
        gravity_axis = self.sim.data.xmat[self.sim.body_id].reshape(3, 3)[2, :]
        roll = math.atan2(gravity_axis[1], gravity_axis[2])
        pitch = math.asin(float(np.clip(-gravity_axis[0], -1., 1.)))
        cr,sr,cp,sp,cy,sy = math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(pose["yaw"]),math.sin(pose["yaw"])
        body_rotation = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ np.array([[cp,sp*sr,sp*cr],[0,cr,-sr],[-sp,cp*sr,cp*cr]])
        sine, cosine = math.sin(CAMERA_PITCH), math.cos(CAMERA_PITCH)
        optical_in_body = np.array([[0.,sine,cosine],[-1.,0.,0.],[0.,-cosine,sine]])
        transform = np.eye(4)
        transform[:3,:3] = body_rotation @ optical_in_body
        transform[:3,3] = np.array([pose["x"],pose["y"],.34]) + body_rotation @ np.array(CAMERA_POSITION)
        width, height = CAMERA_SIZE
        focal = height / (2 * math.tan(math.radians(70) / 2))
        return {"fx":focal,"fy":focal,"cx":(width-1)/2,"cy":(height-1)/2,
                "world_from_camera":transform,"capture_pose":dict(pose)}

    def frame(self, view="overview"):
        with self.lock:
            if view not in {"overview", "robot"}:
                raise ValueError("view must be overview or robot")
            key = self._frame_key()
            cached = self._frame_cache.get(view)
            if cached is None or cached[0] != key:
                if view == "robot":
                    picture = self._render_call(lambda: Image.fromarray(self._live_robot_rgb_owned()))
                else:
                    picture = self._render_call(self._overview_owned)
                self._frame_cache[view] = (key, picture)
            return self._frame_cache[view][1].copy()

    def _frame_key(self):
        return (self.sim.epoch, self.sim._steps, self._visual_revision)

    @staticmethod
    def _display_options(renderer):
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    def _robot_rgb_owned(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.sim.model, height=CAMERA_SIZE[1], width=CAMERA_SIZE[0])
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.sim.data, camera="front")
        self._display_options(self.renderer)
        result = self.renderer.render().copy()
        self._render_counts["robot"] += 1
        return result

    def _live_robot_rgb_owned(self):
        # The display needs fewer pixels than learned recognition. Keep the
        # calibrated RGB-D sensor at 640x360 and render the paced feed at 320x180.
        if self.live_renderer is None:
            self.live_renderer = mujoco.Renderer(self.sim.model, height=DISPLAY_CAMERA_SIZE[1], width=DISPLAY_CAMERA_SIZE[0])
        self.live_renderer.update_scene(self.sim.data, camera="front")
        self._display_options(self.live_renderer)
        self._render_counts["robot"] += 1
        return self.live_renderer.render().copy()

    def _overview_owned(self):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0., 0., 0.]
        camera.distance, camera.azimuth, camera.elevation = 13., 90., -65.
        if self.overview_renderer is None:
            self.overview_renderer = mujoco.Renderer(self.sim.model, height=OVERVIEW_SIZE[1], width=OVERVIEW_SIZE[0])
        self.overview_renderer.update_scene(self.sim.data, camera=camera)
        self._display_options(self.overview_renderer)
        self._render_counts["overview"] += 1
        return Image.fromarray(self.overview_renderer.render().copy())

    def status(self):
        with self.lock:
            return {"state": self.state, "reason": self.reason, "pose": self._pose(),
                    "goal": self.goal, "route": list(self.path), "velocity": self.last_velocity,
                    "sim_seconds": self.elapsed, "wall_contacts": self.wall_contacts,
                    "map_id": self.map_id, "map_source": "measured lidar rays",
                    "mapping": self.mapper.status(), "vision": self.perception.status(),
                    "perception": "learned RGB-D vision", "localization": self.mapper.status()["localization"],
                    "sensor_assumptions": ["stabilized planar lidar", "biased body velocity and IMU gyro odometry",
                                           "fixed camera mount and nominal base height"],
                    "posture": self.sim.mode, "action": dict(self.action) if self.action else None}

    def close(self):
        with self.lock:
            self.stop()
            self._render_call(self._close_owned)
            if self.mapping_path:
                self.mapper.save(self.mapping_path)
            self.perception.close(wait=False)

    def _close_owned(self):
        if self.live_renderer:
            self.live_renderer.close()
            self.live_renderer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None
        if self.overview_renderer:
            self.overview_renderer.close()
            self.overview_renderer = None
