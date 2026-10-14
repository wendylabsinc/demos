"""ROS sensor transports for the Go2 Air bridge and legacy Go2 EDU.

No motion occurs at startup. Every motion request needs fresh pose, obstacle and
camera observations. The Air uses the Wendy Go2 bridge's ROS 2 contract; the
legacy hardware profile retains its separate Unitree SDK motion worker.
"""
import json
import math
import os
import subprocess
import sys
import threading
import time


def _enabled(name):
    value = os.getenv(name, "").strip().lower()
    if value in ("", "0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    raise ValueError(f"{name} must be true or false")


class SafetyGate:
    def __init__(self, max_age=.6):
        self.max_age = max_age
        self.armed = False
        self.timestamps = {}
        self.fault = 'Motion disabled at startup'

    def fresh(self, now):
        return all(0 <= now-self.timestamps.get(k, -math.inf) <= self.max_age
                   for k in ('pose', 'scan', 'camera'))

    def arm(self, now):
        if not self.fresh(now):
            raise RuntimeError('Fresh pose, obstacle data and camera are required')
        self.armed = True
        self.fault = ''

    def check(self, now):
        if not self.fresh(now):
            self.armed = False
            self.fault = 'Stale or missing sensor. Explicit Start required after recovery'
        return self.armed


class ROSTransport:
    def __init__(self, profile='ros-sim'):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        self.profile = profile
        self.bridge_contract = profile in ('air', 'ros-sim')
        self.rclpy = rclpy
        self.Twist = Twist
        self.lock = threading.RLock()
        self.gate = SafetyGate(float(os.getenv('SENSOR_MAX_AGE', '.6')))
        self.pose = (0., 0., 0.)
        self.scan = ()
        self.scan_coverage_flags = ()
        self.scan_error = None
        self.allow_scan_gaps = _enabled('GO2_ALLOW_SCAN_GAPS') if self.bridge_contract else False
        self.latest_odom = None
        self.depth = self.rgb = self.intrinsics = None
        self.preview = None
        self.stamps = {}
        self.capture_id = 0
        self.running = True
        self.last_command = 0.
        self.sdk = None
        rclpy.init()
        from rclpy.parameter import Parameter
        self.node = rclpy.create_node('go2_showcase', parameter_overrides=[
            Parameter('use_sim_time', Parameter.Type.BOOL, False)])
        command_topic = os.getenv('CMD_TOPIC', '/cmd_vel' if self.bridge_contract else '/go2/cmd_vel')
        self.publisher = self.node.create_publisher(Twist, command_topic, 10)
        self.topic_names = {'CMD_TOPIC': command_topic}
        self.subs = []

        if self.bridge_contract:
            from sensor_msgs.msg import PointCloud2
            topics = [(Odometry, 'ODOM_TOPIC', '/utlidar/robot_odom', self._odom),
                      (PointCloud2, 'CLOUD_TOPIC', '/utlidar/cloud_base', self._cloud),
                      (Image, 'RGB_TOPIC', '/camera/color/image_raw', self._rgb)]
        else:
            from sensor_msgs.msg import CameraInfo, LaserScan
            topics = [(Odometry, 'ODOM_TOPIC', '/go2/odom', self._odom),
                      (LaserScan, 'SCAN_TOPIC', '/go2/scan', self._scan),
                      (Image, 'RGB_TOPIC', '/go2/front_camera/image_raw', self._rgb),
                      (Image, 'DEPTH_TOPIC', '/go2/front_camera/depth/image_raw', self._depth),
                      (CameraInfo, 'CAMERA_INFO_TOPIC', '/go2/front_camera/camera_info', self._info)]
        for kind, key, default, callback in topics:
            topic = os.getenv(key, default)
            self.topic_names[key] = topic
            self.subs.append(self.node.create_subscription(kind, topic, callback, qos_profile_sensor_data))

        if profile == 'hardware':
            if not os.getenv('UNITREE_INTERFACE'):
                raise RuntimeError('Set UNITREE_INTERFACE to the confirmed EDU robot network interface')
            if not all(os.getenv(key) for key in ('ODOM_TOPIC', 'SCAN_TOPIC', 'RGB_TOPIC')):
                raise RuntimeError('EDU hardware requires measured ROS pose, scan and RGB topic settings')
            self.sdk = subprocess.Popen([sys.executable, '-m', 'showcase.transport', '--sdk-worker'],
                                        stdin=subprocess.PIPE, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _capture(self, key, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        now_ros = self.node.get_clock().now().nanoseconds / 1e9
        if not math.isfinite(stamp) or not -.1 <= now_ros-stamp <= self.gate.max_age:
            self.gate.timestamps.pop(key, None)
            return False
        self.gate.timestamps[key] = time.monotonic()
        self.stamps[key] = stamp
        return True

    def _odom(self, msg):
        with self.lock:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            vals = (p.x, p.y, q.x, q.y, q.z, q.w)
            norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
            frames_invalid = (msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link')
            if ((self.bridge_contract and frames_invalid)
                    or not all(math.isfinite(v) for v in vals) or not .9 < norm < 1.1):
                self.gate.timestamps.pop('pose', None)
                self.latest_odom = None
                return
            if self._capture('pose', msg):
                self.pose = (p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y),
                                                  1-2*(q.y*q.y+q.z*q.z)))
                self.latest_odom = msg

    def _scan(self, msg):
        """Legacy LaserScan input used only by the EDU profile."""
        with self.lock:
            if not self._capture('scan', msg):
                return
            rays = tuple((msg.angle_min+i*msg.angle_increment, float(r))
                         for i, r in enumerate(msg.ranges))
            frontal = [r for a, r in rays if abs(math.atan2(math.sin(a), math.cos(a))) < .45]
            valid = lambda r: ((math.isfinite(r) and msg.range_min <= r <= msg.range_max)
                               or r == math.inf)
            if len(frontal) < 8 or sum(valid(r) for r in frontal) < .9*len(frontal):
                self.gate.timestamps.pop('scan', None)
                return
            self.scan = rays
            self.scan_coverage_flags = tuple(valid(r) for _, r in rays)
            self.scan_error = None

    def _cloud(self, msg):
        """Convert the Air bridge's occupied voxel cloud without inventing free space."""
        from .pointcloud import project_cloud
        with self.lock:
            if not self._capture('scan', msg):
                return
            if self.latest_odom is None:
                self.gate.timestamps.pop('scan', None)
                self.scan_error = 'waiting for matching odometry'
                return
            try:
                projected = project_cloud(msg, self.latest_odom)
            except (ValueError, TypeError, AttributeError, OverflowError) as exc:
                self.gate.timestamps.pop('scan', None)
                self.scan = ()
                self.scan_coverage_flags = ()
                self.scan_error = str(exc)
                return
            self.scan_coverage_flags = projected.coverage
            self.scan = tuple((projected.angle_min+i*projected.angle_increment, distance)
                              for i, distance in enumerate(projected.ranges)
                              if math.isfinite(distance))
            self.scan_error = None

    def _rgb(self, msg):
        from .perception import decode_image
        with self.lock:
            if self._capture('camera', msg):
                try:
                    self.rgb = decode_image(msg)
                except (ValueError, TypeError):
                    self.gate.timestamps.pop('camera', None)
                    return
                self.preview = self.rgb
                self.capture_id += 1

    def _depth(self, msg):
        from .perception import decode_image
        with self.lock:
            if self._capture('depth', msg):
                self.depth = decode_image(msg)

    def _info(self, msg):
        with self.lock:
            self.intrinsics = (float(msg.k[0]), float(msg.k[2]))

    def _run(self):
        while self.running:
            try:
                self.rclpy.spin_once(self.node, timeout_sec=.03)
                with self.lock:
                    now = time.monotonic()
                    if not self.gate.check(now) or now-self.last_command > .35:
                        self._write(0, 0, 0)
            except Exception as exc:
                with self.lock:
                    self.gate.armed = False
                    self.gate.fault = str(exc)
                    self._write(0, 0, 0)

    def read(self):
        from .models import Observation
        with self.lock:
            now = time.monotonic()
            x, y, yaw = self.pose
            voxels = {}
            for a, r in sorted(self.scan, key=lambda ray: ray[1]):
                if math.isfinite(r) and .1 < r < 5:
                    px, py = x+r*math.cos(yaw+a), y+r*math.sin(yaw+a)
                    voxels.setdefault((round(px/.2), round(py/.2)), (px, py, .12))
            obstacles = tuple(list(voxels.values())[:200])
            sectors = tuple((math.atan2(math.sin(a), math.cos(a)), min(12., r))
                            for a, r in self.scan if r > 0 and not math.isnan(r))
            coverage = self.scan_coverage_flags
            if self.bridge_contract:
                front_indices = [i for i in range(len(coverage))
                                 if abs(math.atan2(math.sin(-math.pi+i*2*math.pi/len(coverage)),
                                                  math.cos(-math.pi+i*2*math.pi/len(coverage)))) < .45]
                front_complete = bool(front_indices) and all(coverage[i] for i in front_indices)
                front = [r for a, r in sectors
                         if abs(math.atan2(math.sin(a), math.cos(a))) < .45]
                front_clearance = min(front) if front and (front_complete or self.allow_scan_gaps) else None
                scan_complete = bool(coverage) and all(coverage)
                coverage_ratio = sum(coverage)/len(coverage) if coverage else 0.
            else:
                front = [r for a, r in sectors
                         if abs(math.atan2(math.sin(a), math.cos(a))) < .45]
                front_clearance = min(front) if front else None
                front_complete = scan_complete = True
                coverage_ratio = 1.
            return Observation(x=x, y=y, yaw=yaw, fresh=self.gate.fresh(now),
                               scan_fresh=now-self.gate.timestamps.get('scan', -math.inf) < self.gate.max_age,
                               obstacles=obstacles, detections=(), sectors=sectors, gesture=None,
                               front_clearance=front_clearance,
                               front_coverage_complete=front_complete,
                               scan_coverage_complete=scan_complete,
                               scan_coverage=coverage_ratio,
                               scan_gaps_allowed=self.allow_scan_gaps)

    def arm(self):
        with self.lock:
            self.gate.arm(time.monotonic())

    def _write(self, vx, vy, wz):
        if self.sdk:
            if self.sdk.poll() is not None:
                self.gate.armed = False
                raise RuntimeError('Unitree SDK worker exited')
            self.sdk.stdin.write(json.dumps({'sent_at': time.monotonic(),
                                             'velocity': [vx, vy, wz]})+'\n')
            self.sdk.stdin.flush()
        else:
            msg = self.Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
            self.publisher.publish(msg)

    def send(self, command):
        with self.lock:
            vals = (command.vx, command.vy, command.yaw_rate)
            if not all(math.isfinite(v) for v in vals):
                self.stop()
                return False
            if not self.gate.check(time.monotonic()):
                self._write(0, 0, 0)
                return False
            obs = self.read()
            vx = max(0., min(.55, command.vx))
            # No strafing or reverse. The bridge's Move path has no obstacle avoidance.
            wz = max(-.6, min(.6, command.yaw_rate))
            if obs.front_clearance is None or obs.front_clearance < .65:
                vx = 0.
            if wz and (not obs.scan_coverage_complete and not obs.scan_gaps_allowed
                       or any(r < .45 for _, r in obs.sectors)):
                wz = 0.
            self.last_command = time.monotonic()
            self._write(vx, 0., wz)
            return True

    def stop(self):
        with self.lock:
            self.gate.armed = False
            self._write(0, 0, 0)

    def status(self):
        with self.lock:
            now = time.monotonic()
            coverage = self.scan_coverage_flags
            return {'profile': self.profile, 'armed': self.gate.armed, 'fault': self.gate.fault,
                    'ready': self.gate.fresh(now), 'topics': self.topic_names,
                    'sensor_ages': {k: round(now-v, 3) for k, v in self.gate.timestamps.items()},
                    'vision': 'Raw RGB camera frame sent to configured vision model',
                    'capture_id': self.capture_id, 'scan_error': self.scan_error,
                    'scan_coverage': round(sum(coverage)/len(coverage), 3) if coverage else 0,
                    'scan_gaps_allowed': self.allow_scan_gaps}

    def sensor_snapshot(self):
        """Only measured navigation data; no object labels or fixture positions."""
        with self.lock:
            obs = self.read()
            sectors = {}
            for bearing, distance in obs.sectors:
                index = min(17, int((bearing+math.pi)/(2*math.pi)*18))
                sectors[index] = min(sectors.get(index, 12.), distance)
            return {'pose': {'x': obs.x, 'y': obs.y, 'yaw': obs.yaw},
                    'front_clearance': obs.front_clearance,
                    'front_coverage_complete': obs.front_coverage_complete,
                    'scan_coverage_complete': obs.scan_coverage_complete,
                    'scan_coverage': obs.scan_coverage,
                    'scan_gaps_allowed': obs.scan_gaps_allowed,
                    'scan_sectors': [[-math.pi+(index+.5)*2*math.pi/18, distance]
                                     for index, distance in sorted(sectors.items())],
                    'healthy': obs.fresh, 'fresh': obs.fresh, 'scan_fresh': obs.scan_fresh,
                    'capture_id': self.capture_id, 'sensor_ages': self.status()['sensor_ages']}

    def camera_jpeg(self):
        return self.frame_jpeg()

    def frame_jpeg(self):
        import cv2
        with self.lock:
            if self.preview is None:
                return b''
            ok, encoded = cv2.imencode('.jpg', self.preview[:, :, ::-1])
            return encoded.tobytes() if ok else b''

    def close(self):
        self.stop()
        self.running = False
        self.thread.join(timeout=2)
        if self.sdk:
            self.sdk.stdin.close()
            try:
                self.sdk.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.sdk.terminate()
        self.node.destroy_node()
        self.rclpy.shutdown()


def build_transport(profile='ros-sim'):
    if profile not in ('ros-sim', 'air', 'hardware'):
        raise ValueError('ROS transport profile must be ros-sim, air or hardware')
    return ROSTransport(profile)


def sdk_worker():
    """Legacy EDU SDK process with an independent .35 second input watchdog."""
    import select
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    ChannelFactoryInitialize(int(os.getenv('ROS_DOMAIN_ID', '0')), os.environ['UNITREE_INTERFACE'])
    client = SportClient()
    client.SetTimeout(.25)
    client.Init()
    client.StopMove()
    try:
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], .35)
            if not readable:
                client.StopMove()
                continue
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            age = time.monotonic() - float(request['sent_at'])
            if not 0 <= age <= .35:
                client.StopMove()
                continue
            vx, vy, wz = request['velocity']
            if not all(math.isfinite(v) for v in (vx, vy, wz)):
                client.StopMove()
                continue
            if vx == vy == wz == 0:
                client.StopMove()
            else:
                result = client.Move(max(0, min(.55, vx)), 0., max(-.6, min(.6, wz)))
                if result not in (None, 0):
                    raise RuntimeError('Unitree Move failed with status '+str(result))
    finally:
        client.StopMove()


if __name__ == '__main__' and '--sdk-worker' in sys.argv:
    sdk_worker()
