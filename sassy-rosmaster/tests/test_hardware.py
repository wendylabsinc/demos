import copy
import json
import math
from types import SimpleNamespace as N
import unittest

from sassy_rosmaster.hardware import Hardware, HardwareFault, _RosmasterMotion
from sassy_rosmaster.types import Command


def configuration():
    return dict(geometry={"wheelbase": .32, "length": .48, "width": .26},
                geometry_confirmed=True, max_speed=.3, max_steering=.55,
                driver_watchdog_confirmed=True, driver_command_timeout=.225,
                scan_topic="/scan_base", odom_topic="/odom", command_topic="/ackermann_cmd",
                estop_topic="/sassy_rosmaster/estop", scan_frame="base_link",
                odom_frame="odom", base_frame="base_link")


class FakeClock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


class FakeTransport:
    def __init__(self, cfg, on_scan, on_odom, on_people, on_estop, on_fault):
        self.scan, self.odom, self.people, self.estop, self.fault = on_scan, on_odom, on_people, on_estop, on_fault
        self.commands = []
        self.clock = None
        self.fail_publish = False
        self.closed = False

    def ros_now(self):
        return self.clock()+1000

    def publish(self, command):
        if self.fail_publish:
            raise RuntimeError("DDS disconnected")
        self.commands.append(command)

    def close(self):
        self.closed = True


def header(stamp, frame="base_link"):
    return N(stamp=N(sec=int(stamp), nanosec=int(round((stamp-int(stamp))*1e9))), frame_id=frame)


def scan(stamp):
    return N(header=header(stamp), ranges=[5.]*360, angle_min=-math.pi,
             angle_max=math.pi-2*math.pi/360, angle_increment=2*math.pi/360,
             range_min=.05, range_max=6., time_increment=0., scan_time=.05)


def odom(stamp, x=0., y=0., yaw=0.):
    return N(header=header(stamp, "odom"), child_frame_id="base_link",
             pose=N(pose=N(position=N(x=x, y=y, z=0.),
                           orientation=N(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2)))))


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.hw = Hardware(configuration(), transport_factory=FakeTransport,
                           clock=self.clock, watchdog=False)
        self.t = self.hw._transport
        self.t.clock = self.clock

    def tearDown(self):
        self.hw.close()

    def fresh(self, *, age=0., x=0., y=0., yaw=0.):
        stamp = self.clock()+1000-age
        self.t.estop(N(data=False))
        self.t.odom(odom(stamp, x, y, yaw))
        self.t.scan(scan(stamp))

    def test_startup_disarmed_and_motion_requires_explicit_start(self):
        self.fresh()
        self.assertEqual(self.t.commands, [Command()])
        with self.assertRaisesRegex(HardwareFault, "press Start"):
            self.hw.step(Command(.2, .1))
        self.assertEqual(self.t.commands[-1], Command())
        self.hw.reset()
        self.hw.start()
        self.hw.step(Command(.2, .1))
        self.assertEqual(self.t.commands[-1], Command(.2, .1))

    def test_motion_limits_are_enforced(self):
        self.fresh()
        self.hw.start()
        self.hw.step(Command(99, -99))
        self.assertEqual(self.t.commands[-1], Command(.3, -.55))

    def test_capture_age_is_preserved_after_delayed_delivery(self):
        self.fresh(age=.2)
        self.assertAlmostEqual(self.hw.read().captured_at, self.clock()-.2)
        self.hw.start()
        self.clock.now += .11
        with self.assertRaisesRegex(HardwareFault, "stale"):
            self.hw.read()
        self.assertFalse(self.hw.status()["armed"])
        self.assertEqual(self.t.commands[-1], Command())

    def test_full_scan_and_finite_rays_required(self):
        for mutate in (lambda m: setattr(m, "ranges", m.ranges[:90]),
                       lambda m: m.ranges.__setitem__(4, math.nan),
                       lambda m: m.ranges.__setitem__(4, math.inf),
                       lambda m: m.ranges.__setitem__(4, .001),
                       lambda m: setattr(m, "angle_increment", math.nan),
                       lambda m: setattr(m.header, "frame_id", "laser")):
            with self.subTest(mutate=mutate):
                self.clock.now += .01
                self.fresh()
                self.hw.reset()
                self.hw.start()
                message = scan(self.clock()+1000+.001)
                mutate(message)
                self.t.scan(message)
                self.assertFalse(self.hw.status()["armed"])
                self.assertEqual(self.t.commands[-1], Command())
                with self.assertRaises(HardwareFault):
                    self.hw.read()

    def test_stale_source_and_future_clock_are_rejected(self):
        for delta in (-.31, .2):
            with self.subTest(delta=delta):
                self.t.estop(N(data=False))
                self.t.scan(scan(self.clock()+1000+delta))
                with self.assertRaisesRegex(HardwareFault, "clocks|stale"):
                    self.hw.read()

    def test_scan_and_odometry_must_be_paired(self):
        self.t.estop(N(data=False))
        self.t.scan(scan(self.clock()+1000))
        self.t.odom(odom(self.clock()+1000-.2))
        with self.assertRaisesRegex(HardwareFault, "synchronized"):
            self.hw.start()

    def test_invalid_odometry_stops_immediately(self):
        self.fresh()
        self.hw.start()
        message = odom(self.clock()+1000+.001)
        message.pose.pose.orientation.w = 0
        self.t.odom(message)
        self.assertFalse(self.hw.status()["armed"])
        self.assertEqual(self.t.commands[-1], Command())
        with self.assertRaisesRegex(HardwareFault, "normalized"):
            self.hw.read()

    def test_estop_latches_until_reset_then_separate_start(self):
        self.fresh()
        self.hw.start()
        self.t.estop(N(data=True))
        self.assertEqual(self.t.commands[-1], Command())
        with self.assertRaises(HardwareFault):
            self.hw.reset()
        self.t.estop(N(data=False))
        with self.assertRaisesRegex(HardwareFault, "e-stop"):
            self.hw.start()
        self.hw.reset()
        self.assertFalse(self.hw.status()["armed"])
        self.hw.start()
        self.assertTrue(self.hw.status()["armed"])

    def test_controller_freeze_stops_without_step_or_read(self):
        self.fresh()
        self.hw.start()
        self.hw.step(Command(.2, 0))
        self.clock.now += .21
        self.hw._watchdog_tick()
        self.assertFalse(self.hw.status()["armed"])
        self.assertIn("heartbeat expired", self.hw.status()["fault"])
        self.assertEqual(self.t.commands[-1], Command())

    def test_estop_heartbeat_expiry_stops_even_with_sensors(self):
        self.fresh()
        self.hw.start()
        self.clock.now += .31
        self.t.odom(odom(self.clock()+1000))
        self.t.scan(scan(self.clock()+1000))
        with self.assertRaisesRegex(HardwareFault, "E-stop heartbeat"):
            self.hw.read()

    def test_stale_odometry_stops_even_with_fresh_scan_and_estop(self):
        self.fresh()
        self.hw.start()
        self.clock.now += .31
        self.t.estop(N(data=False))
        self.t.scan(scan(self.clock()+1000))
        with self.assertRaisesRegex(HardwareFault, "Odometry capture is stale"):
            self.hw.read()

    def test_distance_and_duration_bounds_stop(self):
        self.fresh()
        self.hw.start()
        self.clock.now += .05
        self.t.odom(odom(self.clock()+1000, x=10.1))
        self.assertIn("distance", self.hw.status()["fault"])
        self.assertEqual(self.t.commands[-1], Command())
        self.clock.now += .05
        self.fresh(x=10.1)
        self.hw.reset()
        self.hw.start()
        self.clock.now += 120
        self.hw._watchdog_tick()
        self.assertIn("duration", self.hw.status()["fault"])

    def test_people_bridge_transforms_measured_body_coordinates(self):
        self.fresh(x=2, y=3, yaw=math.pi/2)
        self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000,
                        frame_id="base_link", people=[dict(x=1, y=0)]))))
        person = self.hw.read().people[0]
        self.assertAlmostEqual(person["x"], 2)
        self.assertAlmostEqual(person["y"], 4)
        self.assertAlmostEqual(person["distance"], 1)
        self.assertAlmostEqual(person["bearing"], 0)
        self.assertNotIn("id", person)
        self.assertNotIn("confidence", person)

    def test_people_bridge_preserves_tracking_identity_when_detection_order_changes(self):
        self.fresh(x=2, y=3, yaw=math.pi/2)
        first = dict(x=1, y=0, id=7, confidence=.93)
        second = dict(x=2, y=.5, id="track-B", confidence=.81)
        self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000,
                       frame_id="base_link", people=[first, second, dict(x=3, y=0, id=None, confidence=None)]))))
        people = self.hw.read().people
        self.assertEqual([(p["id"], p["confidence"]) for p in people[:2]], [(7, .93), ("track-B", .81)])
        self.assertAlmostEqual(people[0]["x"], 2)
        self.assertAlmostEqual(people[0]["y"], 4)
        self.assertNotIn("id", people[2])
        self.assertNotIn("confidence", people[2])
        # A fresh frame may reorder boxes. The hardware adapter must not invent
        # array-index IDs, or a pending remark could address the wrong person.
        self.clock.now += .05
        self.fresh(x=2, y=3, yaw=math.pi/2)
        self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000,
                       frame_id="base_link", people=[second, first]))))
        reordered = self.hw.read().people
        self.assertEqual([p["id"] for p in reordered], ["track-B", 7])
        self.assertEqual([p["confidence"] for p in reordered], [.81, .93])
        reordered[0]["id"] = "changed-by-consumer"
        self.assertEqual(self.hw.read().people[0]["id"], "track-B")

    def test_required_people_bridge_rejects_invalid_tracking_metadata(self):
        self.hw._require_people = True
        invalid = [dict(id=True), dict(id=-1), dict(id=1.5), dict(id=2**53),
                   dict(id=""), dict(id=" "*2), dict(id="x"*129), dict(id={}),
                   dict(confidence=True), dict(confidence=-.01), dict(confidence=1.01),
                   dict(confidence=math.nan), dict(confidence=math.inf)]
        frames = [[dict(x=1, y=0, **metadata)] for metadata in invalid]
        frames.append([dict(x=1, y=0, id=7), dict(x=2, y=0, id=7)])
        for people in frames:
            with self.subTest(people=people):
                self.clock.now += .05
                self.fresh()
                self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000,
                               frame_id="base_link", people=[]))))
                self.hw.reset()
                self.hw.start()
                self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000+.001,
                               frame_id="base_link", people=people))))
                self.assertFalse(self.hw.status()["armed"])
                self.assertEqual(self.t.commands[-1], Command())
                with self.assertRaises(HardwareFault):
                    self.hw.read()

    def test_required_people_bridge_failure_stops(self):
        self.hw._require_people = True
        self.fresh()
        with self.assertRaisesRegex(HardwareFault, "perception heartbeat"):
            self.hw.read()
        self.t.people(N(data=json.dumps(dict(stamp=self.clock()+1000,
                       frame_id="base_link", people=[]))))
        self.hw.reset()
        self.hw.start()
        self.t.people(N(data="malformed"))
        self.assertFalse(self.hw.status()["armed"])
        self.assertEqual(self.t.commands[-1], Command())

    def test_transport_failure_disarms_and_requires_reset(self):
        self.fresh()
        self.hw.start()
        self.t.fail_publish = True
        with self.assertRaisesRegex(HardwareFault, "DDS disconnected"):
            self.hw.step(Command(.2, 0))
        self.assertFalse(self.hw.status()["armed"])
        self.t.fail_publish = False
        with self.assertRaisesRegex(HardwareFault, "DDS disconnected"):
            self.hw.start()

    def test_no_motion_without_measured_geometry_and_driver_watchdog(self):
        for key in ("geometry_confirmed", "driver_watchdog_confirmed"):
            cfg = copy.deepcopy(configuration())
            cfg[key] = False
            with self.assertRaises(ValueError):
                Hardware(cfg, transport_factory=FakeTransport, watchdog=False)

    def test_close_sends_zero_and_closes_transport(self):
        self.fresh()
        self.hw.start()
        self.hw.step(Command(.2, .1))
        self.hw.close()
        self.assertEqual(self.t.commands[-1], Command())
        self.assertTrue(self.t.closed)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return json.dumps(self.payload).encode()


class RosmasterMotionTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.health = dict(connected=True, max_speed=.3)

        def opener(req, timeout):
            self.requests.append((req.full_url, None if req.data is None else json.loads(req.data), timeout))
            return FakeResponse(self.health if req.full_url.endswith("/health") else dict(ok=True))

        cfg = dict(motion_url="http://127.0.0.1:3201", motion_api_max_speed=.3,
                   steering_full_scale_rad=.5, steering_calibration_confirmed=True,
                   rosmaster_steer_sign=-1)
        self.motion = _RosmasterMotion(cfg, opener=opener)

    def test_real_r2_api_uses_pwm_steering_sign_and_speed_scaling(self):
        self.motion.publish(Command(.15, .25))
        self.assertEqual(self.requests[-1], ("http://127.0.0.1:3201/drive", dict(throttle=.5, steer=-.5), .1))
        self.motion.publish(Command())
        self.assertEqual(self.requests[-1][0], "http://127.0.0.1:3201/stop")

    def test_disconnected_board_or_changed_driver_scale_blocks_motion(self):
        for health in (dict(connected=False, max_speed=.3), dict(connected=True, max_speed=.6)):
            self.health = health
            with self.assertRaises((HardwareFault, ValueError)):
                self.motion.publish(Command(.15, 0))
            self.assertTrue(self.requests[-1][0].endswith("/health"))


if __name__ == "__main__":
    unittest.main()
