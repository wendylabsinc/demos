import math
import struct
import unittest
from showcase.transport import SafetyGate, ROSTransport
from showcase.models import Command, Observation
from showcase.pointcloud import project_cloud
import threading
from types import SimpleNamespace


def cloud(points, *, endian=False, padding=0, rows=1):
    width = len(points) // rows
    data = b''.join(b''.join(struct.pack(('>' if endian else '<') + 'fff', *point)
                    + b'\0' * 20 for point in points[row*width:(row+1)*width])
                    + b'\0' * padding for row in range(rows))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=10, nanosec=0), frame_id='base_link'),
        width=width, height=rows, point_step=32, row_step=width*32+padding,
        is_bigendian=endian, data=data,
        fields=[SimpleNamespace(name=name, offset=index*4, count=1, datatype=7)
                for index, name in enumerate('xyz')])


def circle(radius=3):
    return [(radius*math.cos(-math.pi+i*math.pi/36),
             radius*math.sin(-math.pi+i*math.pi/36), .08) for i in range(72)]


def odometry(stamp=None):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp or SimpleNamespace(sec=10, nanosec=0), frame_id='odom'),
        child_frame_id='base_link',
        pose=SimpleNamespace(pose=SimpleNamespace(orientation=SimpleNamespace(x=0, y=0, z=0, w=1))))


class SafetyTests(unittest.TestCase):
    def test_startup_requires_three_fresh_inputs(self):
        gate = SafetyGate()
        with self.assertRaises(RuntimeError):
            gate.arm(1)
        gate.timestamps = {'pose': 1, 'scan': 1, 'camera': 1}
        self.assertFalse(gate.check(1))
        gate.arm(1)
        self.assertTrue(gate.check(1.2))

    def test_stale_sensor_disarms_until_operator_restarts(self):
        gate = SafetyGate()
        gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), 1)
        gate.arm(1)
        self.assertFalse(gate.check(2))
        gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), 2)
        self.assertFalse(gate.check(2))
        gate.arm(2)
        self.assertTrue(gate.check(2))

    def test_future_or_invalid_sensor_is_not_fresh(self):
        for value in (math.nan, math.inf, 3):
            gate = SafetyGate()
            gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), value)
            self.assertFalse(gate.fresh(2))

    def test_motor_output_disabled_and_bounded(self):
        import time
        transport = ROSTransport.__new__(ROSTransport)
        transport.lock = threading.RLock()
        transport.gate = SafetyGate()
        transport.gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), time.monotonic())
        outputs = []
        transport._publish = lambda *values: outputs.append(values)
        transport.read = lambda: Observation(fresh=True, scan_fresh=True, front_clearance=2., sectors=((0., 2.),))
        self.assertFalse(transport.send(Command(vx=.55)))
        self.assertEqual(outputs[-1], (0, 0, 0))
        transport.arm()
        self.assertTrue(transport.send(SimpleNamespace(vx=5, vy=5, yaw_rate=5)))
        self.assertEqual(outputs[-1], (.55, 0., .6))
        transport.read = lambda: Observation(fresh=True, scan_fresh=True, front_clearance=.2, sectors=((0., .2),))
        transport.send(Command(vx=.55, yaw_rate=.6))
        self.assertEqual(outputs[-1], (0., 0., 0.))
        transport.send(SimpleNamespace(vx=math.nan, vy=0, yaw_rate=0))
        self.assertFalse(transport.gate.armed)
        self.assertEqual(outputs[-1], (0, 0, 0))

    def test_air_cloud_projects_padded_and_big_endian_points(self):
        for endian, rows, padding in ((False, 1, 0), (True, 2, 8)):
            with self.subTest(endian=endian, rows=rows, padding=padding):
                scan = project_cloud(cloud(circle(), endian=endian, rows=rows, padding=padding),
                                     odometry())
                self.assertEqual(len(scan.ranges), 72)
                self.assertTrue(all(abs(value-3) < 1e-5 for value in scan.ranges))
                self.assertTrue(all(scan.coverage))

    def test_air_cloud_keeps_unknown_sectors_and_rejects_floor(self):
        points = circle()
        points[36] = (.4, 0, .1)
        points[0] = (-1, 0, -.31)
        points[1] = (math.nan, 0, 0)
        scan = project_cloud(cloud(points), odometry())
        self.assertAlmostEqual(scan.ranges[36], .4, places=5)
        self.assertTrue(math.isinf(scan.ranges[0]))
        self.assertTrue(math.isinf(scan.ranges[1]))

    def test_air_cloud_rejects_bad_layout_or_unpaired_odometry(self):
        message = cloud(circle())
        message.data = message.data[:-1]
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            project_cloud(message, odometry())
        with self.assertRaisesRegex(ValueError, 'matching odom'):
            project_cloud(cloud(circle()), odometry(SimpleNamespace(sec=9, nanosec=0)))

    def test_air_transport_does_not_treat_unknown_front_as_clear(self):
        import time
        transport = ROSTransport.__new__(ROSTransport)
        transport.lock = threading.RLock()
        transport.gate = SafetyGate()
        transport.gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), time.monotonic())
        transport.pose = (0., 0., 0.)
        transport.scan = tuple((-math.pi+i*math.pi/36, 3.) for i in range(72) if i != 36)
        transport.scan_coverage_flags = tuple(i != 36 for i in range(72))
        transport.bridge_contract = True
        transport.allow_scan_gaps = False
        observation = transport.read()
        self.assertIsNone(observation.front_clearance)
        self.assertFalse(observation.front_coverage_complete)
        self.assertFalse(observation.scan_coverage_complete)
        transport.allow_scan_gaps = True
        observation = transport.read()
        self.assertAlmostEqual(observation.front_clearance, 3.)
        self.assertTrue(observation.scan_gaps_allowed)


class IdleStopTests(unittest.TestCase):
    """Zero commands are for a robot that could be moving, not a flood while idle."""

    def make(self, armed=False):
        import time
        transport = ROSTransport.__new__(ROSTransport)
        transport.lock = threading.RLock()
        transport.gate = SafetyGate()
        transport.gate.armed = armed
        transport.gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), time.monotonic())
        transport.published = []
        transport._publish = lambda *values: transport.published.append(values)
        transport.read = lambda: Observation(fresh=True, scan_fresh=True, front_clearance=2., sectors=((0., 2.),))
        return transport

    def at(self, seconds):
        from unittest import mock
        return mock.patch('showcase.transport.time.monotonic', return_value=seconds)

    def test_unconnected_robot_gets_a_slow_heartbeat_not_a_stream(self):
        transport = self.make()
        transport.gate.timestamps = {}
        for tick in range(400):  # 20 s at the 20 Hz control rate
            with self.at(100 + tick/20):
                transport.send(Command())
        self.assertLessEqual(len(transport.published), 21)
        self.assertGreaterEqual(len(transport.published), 19)
        self.assertTrue(all(values == (0, 0, 0) for values in transport.published))

    def test_stops_follow_motion_at_full_rate_then_back_off(self):
        transport = self.make(armed=True)
        with self.at(100.):
            transport._write(.4, 0., 0.)
        for tick in range(1, 20):  # one second of 20 Hz stops after the last motion
            with self.at(100 + tick/20):
                transport._write(0, 0, 0)
        self.assertEqual(len(transport.published), 1 + 19)
        count = len(transport.published)
        for tick in range(21, 80):  # then idle: a stop at most once a second
            with self.at(100 + tick/20):
                transport._write(0, 0, 0)
        self.assertLessEqual(len(transport.published) - count, 3)

    def test_explicit_stop_and_gate_loss_always_publish(self):
        transport = self.make(armed=True)
        with self.at(50.):
            transport.stop()
        with self.at(50.01):
            transport.stop()
        self.assertEqual(transport.published, [(0, 0, 0), (0, 0, 0)])
        transport.published.clear()
        transport.gate.armed = True
        transport.gate.timestamps = dict.fromkeys(('pose', 'scan', 'camera'), 10.)
        with self.at(50.02):  # throttled idle heartbeat just went out; loss must still stop now
            self.assertFalse(transport._check_gate(50.02))
        self.assertEqual(transport.published, [(0, 0, 0)])
        self.assertFalse(transport.gate.armed)
        with self.at(50.03):  # already disarmed: no repeat
            transport._check_gate(50.03)
        self.assertEqual(len(transport.published), 1)


if __name__ == '__main__':
    unittest.main()
