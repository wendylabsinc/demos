"""ROS 2 transport for a small Ackermann robot, with motion disabled at startup.

The driver below the ROS command topic must have its own hardware watchdog.
An application cannot guarantee a stop after a process crash or radio failure.
"""
from collections import deque
import json
import math
import threading
import time
import urllib.request
from urllib.parse import urlparse

from .types import Command, Observation, Pose


class HardwareFault(RuntimeError):
    """The vehicle has stopped and needs an explicit reset and start."""


def _number(value, name, minimum, maximum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


class Hardware:
    """Use fresh ROS measurements and issue bounded Ackermann commands.

    ``transport_factory`` and ``clock`` let tests exercise the same safety logic
    without installing ROS. Production uses the real rclpy transport by default.
    """

    def __init__(self, config, *, transport_factory=None, clock=time.monotonic,
                 watchdog=True):
        self.config = dict(config)
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._armed = False
        self._estop_latched = False
        self._fault_reason = None
        self._scan = None
        self._scan_error = None
        self._odom = deque(maxlen=100)
        self._odom_error = None
        self._people = None
        self._people_error = None
        self._estop = None
        self._last_step = -math.inf
        self._last_zero = -math.inf
        self._run_started = 0.0
        self._distance = 0.0
        self._distance_pose = None
        self._transport = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None

        geometry = config.get("geometry", {})
        if config.get("geometry_confirmed") is not True:
            raise ValueError("Measure the ROSMASTER chassis geometry and set geometry_confirmed")
        self.geometry = {
            name: _number(geometry.get(name), f"geometry.{name}", low, high)
            for name, low, high in (("wheelbase", .1, .8), ("length", .15, 1.2),
                                   ("width", .1, .8))
        }
        if self.geometry["length"] <= self.geometry["wheelbase"]:
            raise ValueError("geometry.length must exceed wheelbase")
        self.max_speed = _number(config.get("max_speed", .3), "max_speed", .01, .3)
        self.max_steering = _number(config.get("max_steering", .55), "max_steering", .01, .55)
        self._max_age = _number(config.get("sensor_max_age", .3), "sensor_max_age", .05, .3)
        self._sync_slop = _number(config.get("sync_slop", .1), "sync_slop", .001, .1)
        self._command_timeout = _number(config.get("command_timeout", .2), "command_timeout", .05, .3)
        self._max_run = _number(config.get("max_run_seconds", 120), "max_run_seconds", 1, 120)
        self._max_distance = _number(config.get("max_run_distance", 10), "max_run_distance", .1, 10)
        _number(config.get("driver_command_timeout"), "driver_command_timeout", .01, .3)
        if config.get("driver_watchdog_confirmed") is not True:
            raise ValueError("Confirm the base driver's independent command watchdog before using hardware")
        for key in ("scan_topic", "odom_topic", "command_topic", "estop_topic",
                    "scan_frame", "odom_frame", "base_frame"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"Set the measured ROS {key}")
        if config["scan_frame"] != config["base_frame"]:
            raise ValueError("Reproject lidar at the base-frame origin before feeding this adapter")
        self._require_people = bool(config.get("require_people", False))
        if self._require_people and not config.get("people_topic"):
            raise ValueError("require_people needs a people_topic")
        self.config.setdefault("motion_backend", "ros2_ackermann")
        if self.config["motion_backend"] not in ("ros2_ackermann", "rosmaster_http"):
            raise ValueError("motion_backend must be rosmaster_http or ros2_ackermann")
        if self.config["motion_backend"] == "rosmaster_http":
            full_steer = _number(config.get("steering_full_scale_rad"), "steering_full_scale_rad", .01, .55)
            api_speed = _number(config.get("motion_api_max_speed"), "motion_api_max_speed", .01, .3)
            if self.max_steering > full_steer or self.max_speed > api_speed:
                raise ValueError("Planner motion limits must fit the calibrated ROSMASTER driver limits")

        factory = transport_factory or _Ros2Transport
        self._transport = factory(self.config, self._on_scan, self._on_odom,
                                  self._on_people, self._on_estop, self._on_transport_fault)
        try:
            self._send_zero()
        except Exception:
            self._transport.close()
            raise
        if hasattr(self._transport, "start"):
            self._transport.start()
        if watchdog:
            self._watchdog_thread = threading.Thread(target=self._watchdog_loop,
                                                     name="rosmaster-stop-watchdog", daemon=True)
            self._watchdog_thread.start()

    def _stamp(self, stamp):
        seconds = float(stamp.sec) + float(stamp.nanosec) / 1e9
        if not math.isfinite(seconds) or seconds <= 0:
            raise HardwareFault("Sensor timestamp is missing or invalid")
        ros_age = self._transport.ros_now() - seconds
        if not math.isfinite(ros_age) or ros_age < -.05 or ros_age > self._max_age:
            raise HardwareFault("Sensor capture is stale or clocks are not synchronized")
        return seconds, self._clock() - max(0.0, ros_age)

    def _on_scan(self, msg):
        with self._lock:
            try:
                stamp, capture = self._stamp(msg.header.stamp)
                if msg.header.frame_id != self.config["scan_frame"]:
                    raise HardwareFault("Lidar frame differs from the calibrated base frame")
                values = [float(v) for v in msg.ranges]
                amin, amax, inc = float(msg.angle_min), float(msg.angle_max), float(msg.angle_increment)
                rmin, rmax = float(msg.range_min), float(msg.range_max)
                increment, scan_time = float(msg.time_increment), float(msg.scan_time)
                if not all(math.isfinite(v) for v in (amin, amax, inc, rmin, rmax, increment, scan_time)):
                    raise HardwareFault("Lidar geometry contains non-finite values")
                if len(values) < 90 or inc <= 0 or not 0 <= rmin < rmax <= 100:
                    raise HardwareFault("Lidar geometry or range limits are invalid")
                coverage = (len(values) - 1) * inc
                if coverage < 2 * math.pi - 2 * inc or coverage > 2 * math.pi + inc:
                    raise HardwareFault("A full 360-degree lidar scan is required")
                if abs(amax - amin - coverage) > inc * .1:
                    raise HardwareFault("Lidar angles do not match the number of rays")
                if increment < 0 or scan_time < 0 or max(scan_time, (len(values)-1)*increment) > self._max_age:
                    raise HardwareFault("Lidar scan duration is invalid")
                if any(not math.isfinite(v) or not rmin <= v <= rmax for v in values):
                    raise HardwareFault("Lidar has missing or invalid rays; do not infer free space")
                if self._scan and stamp <= self._scan["stamp"]:
                    raise HardwareFault("Lidar capture timestamp did not advance")
                self._scan = dict(stamp=stamp, pair_stamp=stamp + (len(values)-1)*increment/2,
                                  captured_at=capture, ranges=values, angle_min=amin,
                                  angle_increment=inc, range_max=rmax)
                self._scan_error = None
            except (HardwareFault, ValueError, TypeError, AttributeError) as exc:
                self._scan_error = str(exc)
                self._trip(f"Lidar fault: {exc}")

    def _on_odom(self, msg):
        with self._lock:
            try:
                stamp, capture = self._stamp(msg.header.stamp)
                if msg.header.frame_id != self.config["odom_frame"] or msg.child_frame_id != self.config["base_frame"]:
                    raise HardwareFault("Odometry frames differ from the configured frames")
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                vals = [float(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)]
                if not all(math.isfinite(v) for v in vals):
                    raise HardwareFault("Odometry contains non-finite values")
                if abs(sum(v*v for v in vals[3:])-1) > .02:
                    raise HardwareFault("Odometry quaternion is not normalized")
                # This app's bicycle model and occupancy map require planar motion.
                if abs(p.z) > .1 or abs(q.x) > .05 or abs(q.y) > .05:
                    raise HardwareFault("Odometry is not planar")
                if self._odom and stamp <= self._odom[-1]["stamp"]:
                    raise HardwareFault("Odometry capture timestamp did not advance")
                yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                pose = Pose(p.x, p.y, yaw)
                self._odom.append(dict(stamp=stamp, captured_at=capture, pose=pose))
                self._odom_error = None
                if self._armed and self._distance_pose is not None:
                    self._distance += math.hypot(pose.x-self._distance_pose.x,
                                                pose.y-self._distance_pose.y)
                    if self._distance >= self._max_distance:
                        self._trip("Maximum hardware travel distance reached")
                self._distance_pose = pose
            except (HardwareFault, ValueError, TypeError, AttributeError) as exc:
                self._odom_error = str(exc)
                self._trip(f"Odometry fault: {exc}")

    def _on_people(self, msg):
        with self._lock:
            try:
                payload = json.loads(msg.data)
                if not isinstance(payload, dict) or payload.get("frame_id") != self.config["base_frame"]:
                    raise HardwareFault("Person detections need the configured base frame")
                stamp = _number(payload.get("stamp"), "people.stamp", .001, 1e12)
                age = self._transport.ros_now()-stamp
                if not -.05 <= age <= self._max_age:
                    raise HardwareFault("Person detections are stale or clocks are not synchronized")
                raw = payload.get("people")
                if not isinstance(raw, list) or len(raw) > 100:
                    raise HardwareFault("Person detections need a bounded people array")
                detections = []
                identities = set()
                for person in raw:
                    if not isinstance(person, dict):
                        raise HardwareFault("Each person detection must be an object")
                    x = _number(person.get("x"), "people.x", -100, 100)
                    y = _number(person.get("y"), "people.y", -100, 100)
                    detection = dict(x=x, y=y)
                    identity = person.get("id")
                    if identity is not None:
                        integer_id = type(identity) is int and 0 <= identity <= 2**53-1
                        string_id = isinstance(identity, str) and 0 < len(identity) <= 128 and bool(identity.strip())
                        if not integer_id and not string_id:
                            raise HardwareFault("people.id must be a nonnegative integer or a bounded nonempty string")
                        if identity in identities:
                            raise HardwareFault("Person tracking IDs must be unique within a frame")
                        identities.add(identity)
                        detection["id"] = identity
                    if person.get("confidence") is not None:
                        detection["confidence"] = _number(person["confidence"], "people.confidence", 0, 1)
                    detections.append(detection)
                if self._people and stamp <= self._people["stamp"]:
                    raise HardwareFault("Person detection timestamp did not advance")
                self._people = dict(stamp=stamp, captured_at=self._clock()-max(0, age), people=detections)
                self._people_error = None
            except (HardwareFault, ValueError, TypeError, AttributeError) as exc:
                self._people_error = str(exc)
                if self._require_people:
                    self._trip(f"Person perception fault: {exc}")

    def _on_estop(self, msg):
        with self._lock:
            if not isinstance(msg.data, bool):
                self._trip("E-stop heartbeat is invalid")
                self._estop = None
                return
            self._estop = dict(active=msg.data, received_at=self._clock())
            if msg.data:
                self._estop_latched = True
                self._trip("Physical e-stop asserted")

    def _on_transport_fault(self, error):
        with self._lock:
            self._trip(f"ROS transport failed: {error}")

    def _observation(self):
        now = self._clock()
        if self._closed:
            raise HardwareFault("Hardware adapter is closed")
        if not self._estop or now-self._estop["received_at"] > self._max_age:
            raise HardwareFault("E-stop heartbeat is missing or stale")
        if self._estop["active"]:
            raise HardwareFault("Physical e-stop is active")
        if self._scan_error or not self._scan:
            raise HardwareFault(self._scan_error or "Lidar scan has not arrived")
        if now-self._scan["captured_at"] > self._max_age:
            raise HardwareFault("Lidar capture is stale")
        if self._odom_error or not self._odom:
            raise HardwareFault(self._odom_error or "Odometry has not arrived")
        if now-self._odom[-1]["captured_at"] > self._max_age:
            raise HardwareFault("Odometry capture is stale")
        paired = min(self._odom, key=lambda d: abs(d["stamp"]-self._scan["pair_stamp"]))
        if abs(paired["stamp"]-self._scan["pair_stamp"]) > self._sync_slop:
            raise HardwareFault("Lidar and odometry captures are not synchronized")
        if now-paired["captured_at"] > self._max_age:
            raise HardwareFault("Paired odometry is stale")
        people_fresh = self._people and now-self._people["captured_at"] <= self._max_age
        if self._require_people and (self._people_error or not people_fresh):
            raise HardwareFault(self._people_error or "Person perception heartbeat is missing or stale")
        pose = paired["pose"]
        people = []
        if people_fresh and not self._people_error:
            for p in self._people["people"]:
                c, s = math.cos(pose.yaw), math.sin(pose.yaw)
                people.append(dict(p, x=pose.x+p["x"]*c-p["y"]*s,
                                   y=pose.y+p["x"]*s+p["y"]*c,
                                   distance=math.hypot(p["x"], p["y"]),
                                   bearing=math.atan2(p["y"], p["x"])))
        return Observation(pose=pose, ranges=list(self._scan["ranges"]),
                           angle_min=self._scan["angle_min"],
                           angle_increment=self._scan["angle_increment"],
                           range_max=self._scan["range_max"],
                           captured_at=min(self._scan["captured_at"], paired["captured_at"]),
                           people=people, collision=False)

    def read(self):
        with self._lock:
            try:
                return self._observation()
            except HardwareFault as exc:
                self._trip(str(exc))
                raise

    def start(self):
        """Arm only after an explicit user Start and good sensor evidence."""
        with self._lock:
            self.read()
            if self._fault_reason or self._estop_latched:
                raise HardwareFault(self._fault_reason or "E-stop is latched; reset before Start")
            self._armed = True
            self._run_started = self._clock()
            self._last_step = self._clock()
            self._distance = 0.0
            self._distance_pose = self._odom[-1]["pose"]

    def reset(self):
        """Clear faults only while stopped, with a fresh released e-stop."""
        with self._lock:
            self.stop()
            self._observation()
            self._fault_reason = None
            self._estop_latched = False

    reset_estop = reset

    def stop(self):
        with self._lock:
            self._armed = False
            self._send_zero()

    def emergency_stop(self):
        with self._lock:
            self._estop_latched = True
            self._trip("Operator e-stop asserted")

    def _send_zero(self):
        if self._transport is not None:
            self._transport.publish(Command())
            self._last_zero = self._clock()

    def _trip(self, reason):
        self._armed = False
        self._fault_reason = str(reason)
        try:
            self._send_zero()
        except Exception:
            # A disconnected DDS/driver can only be stopped by its own watchdog.
            pass

    def step(self, command, dt=.05):
        del dt  # Hardware time comes from clocks and sensor timestamps.
        with self._lock:
            if command.speed == 0 and command.steering == 0:
                self._last_step = self._clock()
                self._send_zero()
                return
            try:
                if not self._armed or self._fault_reason or self._estop_latched:
                    raise HardwareFault(self._fault_reason or "Hardware is stopped; press Start explicitly")
                if not math.isfinite(command.speed) or not math.isfinite(command.steering):
                    raise HardwareFault("Motion command contains non-finite values")
                self._check_limits()
                self._observation()
                bounded = Command(max(-self.max_speed, min(self.max_speed, command.speed)),
                                  max(-self.max_steering, min(self.max_steering, command.steering)))
                self._transport.publish(bounded)
                self._last_step = self._clock()
            except Exception as exc:
                self._trip(str(exc))
                raise HardwareFault(str(exc)) from exc

    def _check_limits(self):
        if self._clock()-self._run_started >= self._max_run:
            raise HardwareFault("Maximum hardware run duration reached")
        if self._distance >= self._max_distance:
            raise HardwareFault("Maximum hardware travel distance reached")
        if self._clock()-self._last_step > self._command_timeout:
            raise HardwareFault("Controller command heartbeat expired")

    def _watchdog_tick(self):
        with self._lock:
            if self._closed:
                return
            if self._armed:
                try:
                    self._check_limits()
                    self._observation()
                except Exception as exc:
                    self._trip(str(exc))
            if not self._armed and self._clock()-self._last_zero >= .1:
                try:
                    self._send_zero()
                except Exception as exc:
                    self._trip(f"Stop command failed: {exc}")

    def _watchdog_loop(self):
        while not self._watchdog_stop.wait(.02):
            self._watchdog_tick()

    def scene(self):
        return dict(width=14, height=12, obstacles=[], people=[])

    def status(self):
        with self._lock:
            return dict(armed=self._armed, fault=self._fault_reason,
                        estop_latched=self._estop_latched, distance=self._distance)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._trip("Hardware adapter closed")
            self._closed = True
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=1)
        self._transport.close()


class _Ros2Transport:
    """Actual ROS subscriptions and publishers, isolated from safety state."""

    def __init__(self, config, on_scan, on_odom, on_people, on_estop, on_fault):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from ackermann_msgs.msg import AckermannDriveStamped
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan
            from std_msgs.msg import Bool, String
        except ImportError as exc:
            raise RuntimeError("Hardware requires ROS 2 rclpy and message packages; use Dockerfile.ROS2") from exc
        self._rclpy = rclpy
        self._message_type = AckermannDriveStamped
        self._motion = _RosmasterMotion(config) if config["motion_backend"] == "rosmaster_http" else None
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node("sassy_rosmaster_driver", context=self._context)
        self._base_frame = config["base_frame"]
        # BEST_EFFORT readers accept both reliable and best-effort sensor writers.
        sensors = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE)
        commands = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        self._publisher = None if self._motion else self._node.create_publisher(
            AckermannDriveStamped, config["command_topic"], commands)
        self._subscriptions = [
            self._node.create_subscription(LaserScan, config["scan_topic"], on_scan, sensors),
            self._node.create_subscription(Odometry, config["odom_topic"], on_odom, sensors),
            self._node.create_subscription(Bool, config["estop_topic"], on_estop, sensors),
        ]
        if config.get("people_topic"):
            self._subscriptions.append(self._node.create_subscription(
                String, config["people_topic"], on_people, sensors))
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._stopping = threading.Event()

        def spin():
            try:
                while not self._stopping.is_set() and self._context.ok():
                    self._executor.spin_once(timeout_sec=.05)
            except Exception as exc:
                if not self._stopping.is_set():
                    on_fault(exc)

        self._thread = threading.Thread(target=spin, name="rosmaster-ros2", daemon=True)

    def start(self):
        self._thread.start()

    def ros_now(self):
        return self._node.get_clock().now().nanoseconds / 1e9

    def publish(self, command):
        if self._motion:
            self._motion.publish(command)
            return
        msg = self._message_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.drive.speed = float(command.speed)
        msg.drive.steering_angle = float(command.steering)
        self._publisher.publish(msg)

    def close(self):
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        self._executor.shutdown(timeout_sec=1)
        self._node.destroy_node()
        self._context.shutdown()


class _RosmasterMotion:
    """Talk to Wendy's real RC-car motion service, which owns the STM32 link."""

    def __init__(self, config, *, opener=urllib.request.urlopen, clock=time.monotonic):
        self._opener = opener
        self._clock = clock
        self._url = str(config.get("motion_url", "")).rstrip("/")
        parsed = urlparse(self._url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("motion_url needs a plain http(s) motion-service URL")
        if config.get("steering_calibration_confirmed") is not True:
            raise ValueError("Calibrate ROSMASTER steering and set steering_calibration_confirmed")
        self._speed = _number(config.get("motion_api_max_speed"), "motion_api_max_speed", .01, .3)
        self._steering = _number(config.get("steering_full_scale_rad"), "steering_full_scale_rad", .01, .55)
        self._sign = config.get("rosmaster_steer_sign", -1)
        if isinstance(self._sign, bool) or self._sign not in (-1, 1):
            raise ValueError("rosmaster_steer_sign must be -1 or 1")
        self._last_health = -math.inf

    def _request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self._url+path, data=data,
                                         headers={"Content-Type": "application/json"})
        with self._opener(request, timeout=.1) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise HardwareFault("ROSMASTER motion response is too large")
            result = json.loads(raw)
        if not isinstance(result, dict):
            raise HardwareFault("ROSMASTER motion response is invalid")
        return result

    def publish(self, command):
        if not math.isfinite(command.speed) or not math.isfinite(command.steering):
            raise HardwareFault("ROSMASTER command must be finite")
        if command.speed == 0 and command.steering == 0:
            if self._request("/stop", {}).get("ok") is not True:
                raise HardwareFault("ROSMASTER motion service rejected Stop")
            return
        if self._clock()-self._last_health > 1:
            health = self._request("/health")
            if health.get("connected") is not True:
                raise HardwareFault("ROSMASTER STM32 driver board is disconnected")
            actual_speed = _number(health.get("max_speed"), "motion health.max_speed", .01, .3)
            if abs(actual_speed-self._speed) > 1e-6:
                raise HardwareFault("ROSMASTER MAX_SPEED differs from the calibrated configuration")
            self._last_health = self._clock()
        payload = dict(throttle=max(-1., min(1., command.speed/self._speed)),
                       steer=max(-1., min(1., self._sign*command.steering/self._steering)))
        if self._request("/drive", payload).get("ok") is not True:
            raise HardwareFault("ROSMASTER motion service rejected Drive")
