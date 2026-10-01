import math
import unittest
from showcase.transport import SafetyGate, ROSTransport
from showcase.models import Command, Observation
import threading
from types import SimpleNamespace


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
        transport._write = lambda *values: outputs.append(values)
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


if __name__ == '__main__':
    unittest.main()
