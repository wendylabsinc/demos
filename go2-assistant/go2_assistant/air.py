"""ROS 2 backend for a Go2 Air exposed through Wendy's Go2 bridge.

The Air publishes an accumulated occupied-voxel map rather than raw LiDAR
scans. This backend keeps missing sectors unknown, plans only through derived
known-free cells, and uses the current voxel projection as an independent
motion guard. RGB recognition is intentionally non-spatial because the bridge
does not provide calibrated camera depth.
"""
from __future__ import annotations

from collections import deque
import math
import os
from pathlib import Path
import threading
import time
from typing import Callable

import numpy as np
from PIL import Image

from .mapping import OccupancyMapper
from .perception import RgbPerception


def _angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _stamp_ns(stamp):
    if (type(stamp.sec) is not int or type(stamp.nanosec) is not int
            or not 0 <= stamp.nanosec < 1_000_000_000):
        raise ValueError("Malformed ROS capture timestamp")
    result = stamp.sec * 1_000_000_000 + stamp.nanosec
    if result <= 0:
        raise ValueError("ROS capture timestamp must be positive")
    return result


def quaternion_state(x, y, z, w):
    values = np.asarray([x, y, z, w], dtype=np.float64)
    if not np.isfinite(values).all() or abs(np.linalg.norm(values) - 1.) > .01:
        raise ValueError("Odometry orientation must be a unit quaternion")
    values /= np.linalg.norm(values)
    x, y, z, w = values
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    # Express body points in a heading-relative frame with odometry vertical.
    level = np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]]) @ rotation
    return yaw, level


def point_cloud_xyz(message):
    """Decode a bounded PointCloud2-shaped object without importing ROS."""
    if (message.header.frame_id != "base_link" or type(message.width) is not int
            or type(message.height) is not int or not 0 < message.width * message.height <= 200_000
            or not 12 <= message.point_step <= 256
            or not message.width * message.point_step <= message.row_step
            or len(message.data) != message.row_step * message.height
            or len(message.data) > 16_000_000):
        raise ValueError("Invalid base_link PointCloud2 dimensions")
    fields = {}
    for field in message.fields:
        if field.name in {"x", "y", "z"}:
            size = 4 if field.datatype == 7 else 8 if field.datatype == 8 else 0
            if (field.name in fields or field.count != 1 or not size or field.offset < 0
                    or field.offset + size > message.point_step):
                raise ValueError("Invalid PointCloud2 XYZ layout")
            fields[field.name] = (field.offset, "f4" if size == 4 else "f8")
    if set(fields) != {"x", "y", "z"}:
        raise ValueError("PointCloud2 requires finite XYZ fields")
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({"names": list("xyz"),
                      "formats": [endian + fields[name][1] for name in "xyz"],
                      "offsets": [fields[name][0] for name in "xyz"],
                      "itemsize": message.point_step})
    data = np.ndarray((message.height, message.width), dtype=dtype, buffer=bytes(message.data),
                      strides=(message.row_step, message.point_step))
    points = np.column_stack([data[name].astype(np.float64).reshape(-1) for name in "xyz"])
    if not np.isfinite(points).all():
        raise ValueError("PointCloud2 contains non-finite XYZ values")
    return points


def rgb_image(message):
    if (message.header.frame_id != "camera_optical_frame" or message.encoding != "rgb8"
            or type(message.width) is not int or type(message.height) is not int
            or not 8 <= message.width <= 1920 or not 8 <= message.height <= 1080
            or not message.width * 3 <= message.step <= message.width * 3 + 4096
            or len(message.data) != message.step * message.height):
        raise ValueError("Expected a bounded camera_optical_frame rgb8 image")
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    return rows[:, :message.width * 3].reshape(message.height, message.width, 3).copy()


class CaptureClock:
    """Convert synchronized ROS wall-clock captures to local monotonic time."""

    def __init__(self, *, wall_clock_ns=time.time_ns, monotonic=time.monotonic, max_age=.75):
        self.wall_clock_ns = wall_clock_ns
        self.monotonic = monotonic
        self.max_age = float(max_age)
        self.offset = float(os.getenv("GO2_SENSOR_CLOCK_OFFSET_SECONDS", "0"))
        if not math.isfinite(self.offset):
            raise ValueError("GO2_SENSOR_CLOCK_OFFSET_SECONDS must be finite")

    def capture(self, source_ns):
        source_ns = int(source_ns)
        age = ((self.wall_clock_ns() + round(self.offset * 1e9)) - source_ns) / 1e9
        if age < -.05:
            raise ValueError("Sensor capture is in the future; check clock synchronization")
        if age > self.max_age:
            raise ValueError(f"Sensor capture is stale ({age:.3f}s)")
        return self.monotonic() - max(0., age)


class AirBackend:
    """Conservative voxel navigation and RGB-only perception for Go2 Air."""

    sensor_mapping = True
    spatial_perception = False
    is_simulator = False
    tick_interval = .05
    capabilities = {"explore": True, "find_object": True, "spin": True,
                    "spatial_people": False, "posture": False}
    available_views = ("map", "camera")

    def __init__(self, map_id, *, mapping_path=None, mapper=None, perception=None,
                 clock=time.monotonic, command_sink: Callable | None = None,
                 writer_count: Callable | None = None, start_ros=True):
        if not isinstance(map_id, str) or not map_id.strip() or len(map_id) > 128:
            raise ValueError("Air mode requires a bounded explicit map/session ID")
        self.map_id = map_id.strip()
        self.mapping_path = Path(mapping_path) if mapping_path else None
        if mapper is not None:
            self.mapper = mapper
        elif self.mapping_path and self.mapping_path.exists():
            self.mapper = OccupancyMapper.load(self.mapping_path)
            if self.mapper.map_id != self.map_id:
                raise ValueError("Saved voxel map does not match GO2_MAP_ID")
        else:
            self.mapper = OccupancyMapper(robot_radius=.4, map_id=self.map_id)
        self.perception = perception if perception is not None else RgbPerception()
        self.clock = clock
        self.lock = threading.RLock()
        self._command_sink = command_sink or (lambda command: None)
        self._writer_count = writer_count or (lambda: 1)
        self._ros = None
        self._odom = deque(maxlen=32)
        self._last_source = {}
        self._cloud_at = None
        self._camera_at = None
        self._scan_ranges = None
        self._scan_angles = None
        self._frame = None
        self._sensor_errors = {}
        self._armed = False
        self.state, self.reason = "idle", "Air backend is disarmed"
        self.path, self.goal = [], None
        self._route_revision = -1
        self.action = None
        self.last_command = (0., 0., 0.)
        if start_ros:
            self._ros = AirRosAdapter(self)

    def _advance(self, key, source_ns):
        source_ns = int(source_ns)
        if source_ns <= self._last_source.get(key, -1):
            raise ValueError(f"{key} capture did not advance")
        self._last_source[key] = source_ns

    def update_odometry(self, x, y, quaternion, source_ns, captured_at):
        with self.lock:
            self._advance("odom", source_ns)
            x, y = _finite(x, "odometry x"), _finite(y, "odometry y")
            captured_at = _finite(captured_at, "odometry capture")
            yaw, level = quaternion_state(*quaternion)
            if self._odom and math.hypot(x - self._odom[-1]["pose"][0], y - self._odom[-1]["pose"][1]) > 1.5:
                self._fault("Odometry jumped by more than 1.5 metres")
                return False
            self._odom.append({"source_ns": int(source_ns), "captured": captured_at,
                               "pose": (x, y, yaw), "level": level})
            self._sensor_errors.pop("odom", None)
            return True

    def _paired_odom(self, source_ns):
        if not self._odom:
            raise ValueError("Voxel map is waiting for odometry")
        sample = min(self._odom, key=lambda item: abs(item["source_ns"] - source_ns))
        if abs(sample["source_ns"] - source_ns) > 100_000_000:
            raise ValueError("Voxel map and odometry differ by more than 100 ms")
        return sample

    def update_voxels(self, points, source_ns, captured_at):
        with self.lock:
            self._advance("cloud", source_ns)
            captured_at = _finite(captured_at, "voxel capture")
            odom = self._paired_odom(int(source_ns))
            leveled = np.asarray(points, dtype=np.float64) @ odom["level"].T
            result = self.mapper.update_voxels(leveled, odom["pose"], timestamp=captured_at)
            self._scan_ranges = result["ranges"]
            self._scan_angles = np.linspace(-math.pi, math.pi, len(self._scan_ranges), endpoint=False)
            self._cloud_at = captured_at
            self._sensor_errors.pop("cloud", None)
            self._refresh_route()
            return True

    def update_camera(self, image, source_ns, captured_at):
        with self.lock:
            self._advance("camera", source_ns)
            captured_at = _finite(captured_at, "camera capture")
            if not self._odom:
                raise ValueError("Camera is waiting for odometry")
            pose = dict(zip(("x", "y", "yaw"), self._odom[-1]["pose"]))
            self._frame = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
            self._camera_at = captured_at
            self.perception.submit(self._frame, pose, captured_at)
            self._sensor_errors.pop("camera", None)
            return True

    def sensor_error(self, key, error):
        with self.lock:
            self._sensor_errors[key] = str(error)
            if key in {"odom", "cloud"} and self._armed:
                self._fault(f"{key} input failed: {error}")

    def _pose(self):
        if not self._odom:
            return None
        return dict(zip(("x", "y", "yaw"), self._odom[-1]["pose"]))

    def _readiness_error(self):
        now = self.clock()
        if not self._odom or not 0 <= now - self._odom[-1]["captured"] <= .5:
            return "Odometry is unavailable or stale"
        if self._cloud_at is None or not 0 <= now - self._cloud_at <= .5:
            return "Voxel map is unavailable or stale"
        if not self.mapper.localization.get("valid"):
            return "Voxel map localization is not ready"
        try:
            writers = int(self._writer_count())
        except Exception as error:
            return f"Cannot verify cmd_vel ownership: {error}"
        if writers != 1:
            return f"Expected one cmd_vel publisher, found {writers}"
        return None

    def arm(self, timeout=5.):
        deadline = self.clock() + max(0., float(timeout))
        while True:
            with self.lock:
                error = self._readiness_error()
                if error is None:
                    self._armed = True
                    self.state, self.reason = "idle", "Air backend armed; awaiting a task"
                    self._publish((0., 0., 0.))
                    return self.status() | {"accepted": True}
            if self.clock() >= deadline:
                with self.lock:
                    self._armed = False
                    self.reason = error
                    return self.status() | {"accepted": False}
            time.sleep(.05)

    def _fault(self, reason):
        self._armed = False
        self.path, self.goal, self.action = [], None, None
        self.state, self.reason = "fault", str(reason)
        self._publish((0., 0., 0.))

    def observe(self):
        with self.lock:
            vision = self.perception.poll()
            error = self._readiness_error()
            pose = self._pose()
            return {"map_id": self.map_id,
                    "timestamp": self._odom[-1]["captured"] if self._odom else self.clock(),
                    "pose": pose, "detections": vision["detections"] if error is None else [],
                    "room": vision["room"] if error is None else None, "vision": self.perception.status(),
                    "healthy": error is None, "reason": error or "",
                    "localization": dict(self.mapper.localization),
                    "map_source": "accumulated occupied voxels"}

    def map_snapshot(self):
        return self.mapper.snapshot()

    def set_search_target(self, label):
        self.perception.set_target(label)

    def search_points(self, room=None):
        return self.mapper.frontiers(limit=20)

    def navigate(self, x, y):
        with self.lock:
            if not self._armed:
                return {"accepted": False, "reason": "Air backend must be explicitly armed"}
            error = self._readiness_error()
            if error:
                self._fault(error)
                return {"accepted": False, "reason": error}
            x, y = _finite(x, "goal x"), _finite(y, "goal y")
            route = self.mapper.plan(x, y)
            if not route:
                return {"accepted": False, "reason": "No route exists through voxel-observed free space"}
            self.path, self.goal = route, (x, y)
            self._route_revision = self.mapper.revision
            self.action = None
            self.state, self.reason = "navigating", "Following a route through voxel-observed free space"
            return {"accepted": True}

    def _begin_turn(self, name, radians):
        if not self._armed:
            return {"accepted": False, "reason": "Air backend must be explicitly armed"}
        error = self._readiness_error()
        pose = self._pose()
        if error or pose is None or not self.mapper.free(pose["x"], pose["y"]):
            return {"accepted": False, "reason": error or "Turning footprint is not observed clear"}
        self.path, self.goal = [], None
        self.action = {"name": name, "remaining": float(radians), "last_yaw": pose["yaw"],
                       "started": self.clock()}
        self.state, self.reason = "navigating", "Turning in place"
        return {"accepted": True, "action": name}

    def scan(self):
        with self.lock:
            return self._begin_turn("scan", .82)

    def perform_action(self, action):
        with self.lock:
            if action == "spin":
                return self._begin_turn("spin", 2 * math.pi)
            return {"accepted": False, "reason": "Go2 Air posture actions are not enabled in this safe first pass"}

    def _refresh_route(self):
        if not self.goal or self._route_revision == self.mapper.revision:
            return
        route = self.mapper.plan(*self.goal)
        if not route:
            self.path = []
            self.state, self.reason = "blocked", "New voxel data closed the route"
            self._publish((0., 0., 0.))
        else:
            self.path = route
            self.reason = "Replanned around updated occupied voxels"
        self._route_revision = self.mapper.revision

    def _sector(self, limit):
        if self._scan_ranges is None:
            return 0., None
        selected = np.abs(self._scan_angles) <= limit
        finite = np.isfinite(self._scan_ranges[selected])
        values = self._scan_ranges[selected][finite]
        return float(np.count_nonzero(finite) / max(1, np.count_nonzero(selected))), \
            float(np.min(values)) if len(values) else None

    def _turn_clear(self):
        if self._scan_ranges is None:
            return False
        finite = np.isfinite(self._scan_ranges)
        return np.count_nonzero(finite) / len(finite) >= .6 and float(np.min(self._scan_ranges[finite])) >= .55

    def _motion(self):
        error = self._readiness_error()
        if error:
            self._fault(error)
            return (0., 0., 0.)
        pose = self._pose()
        if self.action:
            delta = abs(_angle(pose["yaw"] - self.action["last_yaw"]))
            self.action["remaining"] -= delta
            self.action["last_yaw"] = pose["yaw"]
            if self.action["remaining"] <= .06:
                self.action = None
                self.state, self.reason = "arrived", "Finished the bounded turn"
                return (0., 0., 0.)
            if self.clock() - self.action["started"] > 20. or not self._turn_clear():
                self._fault("Bounded turn timed out or lost voxel clearance")
                return (0., 0., 0.)
            return (0., 0., .4)
        self._refresh_route()
        if self.state != "navigating" or not self.path:
            return (0., 0., 0.)
        while self.path and math.dist((pose["x"], pose["y"]), self.path[0]) < .25:
            self.path.pop(0)
        if not self.path or math.dist((pose["x"], pose["y"]), self.goal) <= .35:
            self.path, self.goal = [], None
            self.state, self.reason = "arrived", "Reached the requested location within odometry tolerance"
            return (0., 0., 0.)
        target = self.path[0]
        heading = math.atan2(target[1] - pose["y"], target[0] - pose["x"])
        heading_error = _angle(heading - pose["yaw"])
        if abs(heading_error) > .2:
            if not self._turn_clear():
                self.state, self.reason = "blocked", "Voxel coverage cannot confirm turning clearance"
                return (0., 0., 0.)
            return (0., 0., max(-.4, min(.4, 1.2 * heading_error)))
        coverage, clearance = self._sector(math.radians(30))
        if coverage < .6 or clearance is None:
            self.state, self.reason = "blocked", "Forward voxel coverage is incomplete"
            return (0., 0., 0.)
        if clearance < .85:
            self.state, self.reason = "blocked", "Occupied voxel is inside the stopping clearance"
            return (0., 0., 0.)
        lookahead_distance = min(.7, math.dist((pose["x"], pose["y"]), target))
        lookahead = (pose["x"] + lookahead_distance * math.cos(heading),
                     pose["y"] + lookahead_distance * math.sin(heading))
        if not self.mapper.line_free((pose["x"], pose["y"]), lookahead):
            self.state, self.reason = "blocked", "Planned corridor is no longer voxel-observed clear"
            return (0., 0., 0.)
        return (.55, 0., max(-.3, min(.3, heading_error)))

    def _publish(self, command):
        self.last_command = tuple(float(value) for value in command)
        self._command_sink(self.last_command)

    def tick(self, dt=.05):
        with self.lock:
            if not self._armed:
                self._publish((0., 0., 0.))
                return self.status()
            self._publish(self._motion())
            return self.status()

    def stop(self):
        with self.lock:
            self.path, self.goal, self.action = [], None, None
            if self.state != "fault":
                self.state, self.reason = "idle", "Stopped"
            self._publish((0., 0., 0.))
            return {"accepted": True, "reason": self.reason}

    def status(self):
        with self.lock:
            coverage, clearance = self._sector(math.radians(30))
            return {"state": self.state, "reason": self.reason, "armed": self._armed,
                    "backend": "go2-air-ros2", "map_id": self.map_id, "pose": self._pose(),
                    "goal": self.goal, "route": list(self.path), "velocity": self.last_command,
                    "mapping": self.mapper.status(), "vision": self.perception.status(),
                    "perception": "learned RGB recognition without metric depth",
                    "spatial_perception": False, "front_coverage": coverage,
                    "front_clearance": clearance, "sensor_errors": dict(self._sensor_errors),
                    "capabilities": dict(self.capabilities)}

    def frame(self, view="robot"):
        with self.lock:
            if view not in {"robot", "camera", "overview"} or self._frame is None:
                raise RuntimeError("A fresh Go2 Air camera frame is not available")
            return self._frame.copy()

    def close(self):
        with self.lock:
            self.stop()
            if self.mapping_path and self.mapper.revision:
                self.mapper.save(self.mapping_path)
            self.perception.close(wait=False)
        if self._ros is not None:
            self._ros.close()
            self._ros = None


class AirRosAdapter:
    """Lazy ROS imports and callbacks around the transport-independent backend."""

    def __init__(self, backend):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image as RosImage, PointCloud2

        self.rclpy, self.Twist = rclpy, Twist
        self.backend = backend
        self.clock = CaptureClock()
        rclpy.init(args=None)
        self.node = rclpy.create_node("wendy_go2_assistant_air")
        sensors = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.node.create_publisher(Twist, "/cmd_vel", QoSProfile(depth=1))
        self.node.create_subscription(Odometry, "/odom", self._odom, sensors)
        self.node.create_subscription(PointCloud2, "/utlidar/cloud_base", self._cloud, sensors)
        self.node.create_subscription(RosImage, "/camera/color/image_raw", self._camera, sensors)
        backend._command_sink = self.publish
        backend._writer_count = self.writer_count
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,),
                                       name="go2-air-ros2", daemon=True)
        self.thread.start()

    def _captured(self, message):
        source = _stamp_ns(message.header.stamp)
        return source, self.clock.capture(source)

    def _odom(self, message):
        try:
            if message.header.frame_id != "odom" or message.child_frame_id != "base_link":
                raise ValueError("Expected odom to base_link")
            source, captured = self._captured(message)
            pose, q = message.pose.pose, message.pose.pose.orientation
            self.backend.update_odometry(pose.position.x, pose.position.y,
                                         (q.x, q.y, q.z, q.w), source, captured)
        except Exception as error:
            self.backend.sensor_error("odom", error)

    def _cloud(self, message):
        try:
            source, captured = self._captured(message)
            self.backend.update_voxels(point_cloud_xyz(message), source, captured)
        except Exception as error:
            self.backend.sensor_error("cloud", error)

    def _camera(self, message):
        try:
            source, captured = self._captured(message)
            self.backend.update_camera(rgb_image(message), source, captured)
        except Exception as error:
            self.backend.sensor_error("camera", error)

    def publish(self, command):
        message = self.Twist()
        message.linear.x, message.linear.y, message.angular.z = command
        self.publisher.publish(message)

    def writer_count(self):
        return len(self.node.get_publishers_info_by_topic("/cmd_vel"))

    def close(self):
        try:
            self.publish((0., 0., 0.))
        finally:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
            self.thread.join(timeout=2.)
