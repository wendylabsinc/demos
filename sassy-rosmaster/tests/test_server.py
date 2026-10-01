"""Exercise the HTTP controls and rover lifecycle with isolated fake backends."""

import asyncio
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from sassy_rosmaster import server
from sassy_rosmaster.types import Command, Observation, Pose


class FakeAudio:
    def __init__(self):
        self.events = []

    def emit(self, kind, text):
        self.events.append((kind, text))

    def close(self):
        pass


class FakeJob:
    def __init__(self):
        self.requests, self.results = [], []
        self.accept = True

    def request(self, context):
        if not self.accept:
            return False
        self.requests.append(context)
        return True

    def poll(self, *, allow_result=True):
        result, self.results = self.results, []
        return result if allow_result else []

    def cancel_pending(self):
        self.results = []

    def status(self):
        return dict(state='unverified', requested_model='test-model', error=None)

    def close(self):
        pass


class FakeSpeech(FakeJob):
    def __init__(self):
        super().__init__()
        self.planner = FakeJob()
        self.planner.accept = False

    def explorer(self):
        return self.planner


class FakePerception:
    def __init__(self, source):
        self.backend = source.simulation
        self.visible = True
        self.stale = False
        self.people_override = None

    def start(self):
        pass

    def snapshot(self):
        people = [dict(x=self.backend.pose.x+1.2, y=self.backend.pose.y)] if self.visible and self.backend.person else []
        if self.people_override is not None:
            people = self.people_override
        return dict(ready=True, state='observed', model='test-camera',
                    captured_at=time.monotonic()-(2. if self.stale else 0.),
                    frame_id='camera', people=people, jpeg=None)

    def close(self):
        pass


class FakeBackend:
    """Only physical I/O is replaced. The production rover and planner run."""

    geometry = {"length": .48, "width": .26, "wheelbase": .32}
    max_speed = .18
    max_steering = .2

    def __init__(self):
        self.pose = Pose(0., 0., 0.)
        self.commands = []
        self.stale = False
        self.collision = False
        self.person = False
        self.armed = False
        self.estop = False
        self.resets = 0

    def read(self):
        captured = time.monotonic() - (1. if self.stale else 0.)
        people = [dict(distance=1.2, bearing=0., x=self.pose.x + 1.2, y=self.pose.y)] if self.person else []
        return Observation(self.pose, [6.] * 180, captured_at=captured,
                           people=people, collision=self.collision)

    def step(self, command, dt=.05):
        self.commands.append(command)
        self.pose = Pose(self.pose.x + command.speed * math.cos(self.pose.yaw) * dt,
                         self.pose.y + command.speed * math.sin(self.pose.yaw) * dt,
                         self.pose.yaw + command.speed * math.tan(command.steering) / .32 * dt)

    def reset(self):
        self.pose = Pose(0., 0., 0.)
        self.collision = self.person = False
        self.resets += 1

    def scene(self):
        return dict(width=14., height=12., obstacles=[], people=[],
                    car=dict(x=self.pose.x, y=self.pose.y, yaw=self.pose.yaw, **self.geometry))

    def start(self):
        if self.estop:
            raise RuntimeError("E-stop is latched")
        self.armed = True

    def stop(self):
        self.armed = False
        self.commands.append(Command())

    def emergency_stop(self):
        self.estop = True
        self.stop()

    def reset_estop(self):
        self.estop = False
        self.resets += 1

    def add_person(self, *, ahead=False):
        self.person = True
        return True

    def clear_people(self):
        self.person = False

    def close(self):
        self.stop()


def request(method, path, body=None):
    """Call the real ASGI app without a listener, HTTP client or background loop."""
    encoded = b"" if body is None else json.dumps(body).encode()
    scope = dict(type="http", asgi={"version": "3.0", "spec_version": "2.3"},
                 http_version="1.1", method=method, scheme="http", path=path,
                 raw_path=path.encode(), query_string=b"", root_path="",
                 headers=[(b"host", b"localhost"), (b"content-type", b"application/json")],
                 server=("localhost", 8096), client=("127.0.0.1", 12345))
    received = False
    result = {"body": b""}

    async def receive():
        nonlocal received
        if not received:
            received = True
            return dict(type="http.request", body=encoded, more_body=False)
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = {key.decode(): value.decode() for key, value in message["headers"]}
        elif message["type"] == "http.response.body":
            result["body"] += message.get("body", b"")

    asyncio.run(server.app(scope, receive, send))
    result["json"] = json.loads(result["body"])
    return result


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(server, "RobotAudio", FakeAudio))
        self.stack.enter_context(patch.object(server, "Speech", FakeSpeech))
        self.stack.enter_context(patch('sassy_rosmaster.perception.CameraPerception', FakePerception))
        self.backend = FakeBackend()
        self.rover = None

    def rover_for(self, mode="simulation", configuration=None):
        config = configuration if configuration is not None else dict(max_speed=.18, max_steering=.2)
        config_path = Path(self.directory) / "hardware.json"
        config_path.write_text(json.dumps(config))
        self.stack.enter_context(patch.dict("os.environ", dict(ROVER_MODE=mode,
                                      ROVER_DATA=self.directory, ROVER_CONFIG=str(config_path))))
        target = "sassy_rosmaster.simulation.Simulation" if mode == "simulation" else "sassy_rosmaster.hardware.Hardware"
        self.stack.enter_context(patch(target, return_value=self.backend))
        self.rover = server.Rover()
        self.addCleanup(self.rover.close)
        previous = getattr(server.app.state, "rover", None)
        server.app.state.rover = self.rover
        self.addCleanup(setattr, server.app.state, "rover", previous)
        return self.rover

    def control(self, action, **extra):
        return request("POST", "/api/control", dict(action=action, **extra))

    def test_startup_health_and_map_are_coherent(self):
        rover = self.rover_for()
        health = request("GET", "/health")
        self.assertEqual(health["status"], 200)
        self.assertFalse(health["json"]["running"])
        self.assertFalse(health["json"]["sensor_ready"])
        rover.tick()
        self.assertTrue(request("GET", "/health")["json"]["sensor_ready"])
        state = request("GET", "/api/state")["json"]
        grid = state["map"]
        self.assertEqual(len(grid["cells"]), grid["width"] * grid["height"])
        self.assertTrue(set(grid["cells"]).issubset({-1, 0, 100}))
        self.assertEqual(len(state["scan"]), 180)
        self.assertEqual(state["scene"]["car"]["x"], state["pose"]["x"])
        self.assertTrue(all(command == Command() for command in self.backend.commands))
        download = request("GET", "/api/map")
        self.assertEqual(download["json"], grid)
        self.assertIn("rosmaster-map.json", download["headers"]["content-disposition"])

    def test_http_controls_and_request_bounds(self):
        rover = self.rover_for()
        self.assertTrue(self.control("start")["json"]["running"])
        rover.tick()
        self.assertGreater(rover.command.speed, 0.)
        self.assertFalse(self.control("stop")["json"]["running"])
        self.assertEqual(self.backend.commands[-1], Command())
        self.assertEqual(self.control("sass", value=3)["json"]["sass"], 3)
        self.assertEqual(self.control("invented_action")["status"], 409)
        for field, value in (("duration", 0), ("duration", 121), ("value", 0), ("value", 4)):
            with self.subTest(field=field, value=value):
                self.assertEqual(self.control("start", **{field: value})["status"], 422)

    def test_stale_sensor_fault_stops_and_requires_reset(self):
        rover = self.rover_for()
        self.backend.stale = True
        self.assertEqual(self.control("start")["status"], 409)
        self.assertFalse(rover.running)
        self.backend.stale = False
        self.assertEqual(self.control("start")["status"], 200)
        rover.tick()
        self.backend.stale = True
        rover.tick()
        self.assertFalse(rover.running)
        self.assertIn("stale", rover.fault.lower())
        self.assertEqual(self.backend.commands[-1], Command())
        self.backend.stale = False
        self.assertEqual(self.control("start")["status"], 409)
        self.assertEqual(self.control("reset")["status"], 200)
        self.assertEqual(self.control("start")["status"], 200)

    def test_bumper_fault_halts_and_counts_once(self):
        rover = self.rover_for()
        self.control("start")
        self.backend.collision = True
        rover.tick()
        rover.tick()
        self.assertFalse(rover.running)
        self.assertEqual(rover.collisions, 1)
        self.assertIn("Bumper", rover.fault)
        self.assertEqual(self.backend.commands[-1], Command())

    def test_reset_refreshes_pose_and_personality_for_new_scenario(self):
        rover = self.rover_for()
        self.backend.pose = Pose(2., 1., .4)
        rover.tick()
        rover.personality.level = 3
        self.backend.person = True
        self.assertTrue(rover.personality.update(self.backend.read(), time.monotonic())[1])
        response = self.control("reset")
        self.assertEqual(response["status"], 200)
        state = response["json"]
        self.assertFalse(state["running"])
        self.assertEqual(state["pose"], dict(x=0., y=0., yaw=0.))
        self.assertEqual(state["pose"]["x"], state["scene"]["car"]["x"])
        self.assertEqual(state["map"]["coverage"], 0.)
        self.assertEqual(state["sass"], 3)
        self.backend.person = True
        self.assertTrue(rover.personality.update(self.backend.read(), time.monotonic())[1],
                        "Reset should allow a new pedestrian scenario to honk immediately")

    def test_hardware_estop_latches_until_reset_and_explicit_start(self):
        rover = self.rover_for("hardware")
        self.control("start")
        self.assertTrue(self.backend.armed)
        response = self.control("estop")
        self.assertFalse(response["json"]["running"])
        self.assertTrue(self.backend.estop)
        self.assertEqual(self.control("start")["status"], 409)
        self.assertEqual(self.control("person")["status"], 409)
        self.assertEqual(self.control("reset")["status"], 200)
        self.assertFalse(self.backend.estop)
        self.assertFalse(self.backend.armed)
        self.assertFalse(rover.running)
        self.assertEqual(self.control("start")["status"], 200)
        self.assertTrue(self.backend.armed)

    def test_hardware_timed_run_stops_without_automatic_restart(self):
        rover = self.rover_for("hardware")
        with patch("sassy_rosmaster.server.time.monotonic", return_value=100.):
            self.assertEqual(self.control("start", duration=2)["status"], 200)
            self.assertEqual(rover.deadline, 102.)
        with patch("sassy_rosmaster.server.time.monotonic", return_value=102.1):
            rover.tick()
        self.assertFalse(rover.running)
        self.assertFalse(self.backend.armed)
        self.assertEqual(self.backend.commands[-1], Command())
        self.assertIn("Timed run", rover.navigator.status)
        self.assertIsNone(rover.fault)

    def test_hardware_navigation_uses_validated_backend_limits(self):
        rover = self.rover_for("hardware", configuration={})
        self.assertEqual(rover.navigator.max_speed, self.backend.max_speed)
        self.assertEqual(rover.navigator.max_steering, self.backend.max_steering)
        self.assertEqual(rover.navigator.wheelbase, self.backend.geometry["wheelbase"])

    def test_hidden_person_labels_do_not_drive_camera_encounters(self):
        rover = self.rover_for()
        self.backend.person = True
        rover.perception.visible = False
        self.control('start')
        rover.tick()
        self.assertEqual(rover.observation.people, [])
        self.assertFalse(rover.speech.requests)
        rover.perception.visible = True
        rover.tick()
        self.assertTrue(rover.speech.requests)
        self.assertEqual(rover.speech.requests[0]['nearest_person_distance'], 1.2)
        self.assertIn('horn', [event['kind'] for event in rover.events])
        self.assertNotIn('speech', [event['kind'] for event in rover.events])
        rover.speech.results = [dict(kind='speech', text='A generated encounter remark.')]
        rover.tick()
        self.assertIn('A generated encounter remark.', [event['text'] for event in rover.events])

    def test_stale_camera_and_stopped_model_results_are_discarded(self):
        rover = self.rover_for()
        self.backend.person = True
        rover.perception.stale = True
        self.control('start')
        rover.tick()
        self.assertEqual(rover.observation.people, [])
        self.assertFalse(rover.speech.requests)
        self.control('stop')
        rover.speech.results = [dict(text='Late model result.')]
        rover.tick()
        self.assertNotIn('Late model result.', [event['text'] for event in rover.events])
        self.assertEqual(rover.command, Command())

    def test_speech_addresses_nearest_front_blocker_not_closer_rear_person(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.5, y=0.), dict(id=8, x=-.4, y=0.)]
        self.control('start')
        rover.tick()
        self.assertEqual(rover.speech.requests[0]['nearest_person_distance'], 1.5)
        self.assertEqual(len(rover.speech.requests[0]['people']), 1)
        self.assertEqual(rover.speech_target, {'id': 7})

    def test_changed_or_remote_speech_target_discards_delayed_remark(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.2, y=0.)]
        self.control('start')
        rover.tick()
        self.assertEqual(rover.speech_target, {'id': 7})
        rover.speech.results = [dict(text='A delayed remark for person seven.')]
        rover.perception.people_override = [dict(id=7, x=-1., y=0.), dict(id=8, x=1.3, y=0.)]
        rover.tick()
        self.assertNotIn('A delayed remark for person seven.', [event['text'] for event in rover.events])
        self.assertIsNone(rover.speech_target)

    def test_backing_away_keeps_speech_relevant_to_same_visible_person(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.2, y=0.)]
        self.control('start')
        rover.tick()
        rover.perception.people_override = [dict(id=7, x=2.6, y=0.)]
        rover.speech.results = [dict(text='A generated remark for the visible person.')]
        rover.tick()
        self.assertIn('A generated remark for the visible person.', [event['text'] for event in rover.events])

    def test_brief_detection_gap_holds_reply_until_same_person_is_seen_again(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.2, y=0.)]
        self.control('start')
        rover.tick()
        rover.speech.results = [dict(text='Current model reply for person seven.')]
        rover.perception.people_override = []
        rover.tick()
        self.assertNotIn('Current model reply for person seven.', [event['text'] for event in rover.events])
        self.assertTrue(rover.speech.results)
        rover.perception.people_override = [dict(id=7, x=1.4, y=0.)]
        rover.tick()
        self.assertIn('Current model reply for person seven.', [event['text'] for event in rover.events])

    def test_detection_gap_expiry_discards_reply_without_inventing_presence(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.2, y=0.)]
        self.control('start')
        rover.tick()
        rover.speech.results = [dict(text='Reply for a person no longer observed.')]
        rover.perception.people_override = []
        rover.speech_last_seen -= 1.
        rover.tick()
        self.assertIsNone(rover.speech_target)
        self.assertEqual(rover.speech.results, [])
        self.assertNotIn('Reply for a person no longer observed.', [event['text'] for event in rover.events])

    def test_clear_people_invalidates_pending_speech_and_previous_camera_frame(self):
        rover = self.rover_for()
        rover.perception.people_override = [dict(id=7, x=1.2, y=0.)]
        self.control('start')
        rover.tick()
        self.assertTrue(rover.observation.people)
        rover.speech.results = [dict(text='Stale scene remark.')]
        self.control('clear_people')
        self.assertEqual(rover.state()['observed_people'], [])
        self.assertFalse(rover.camera_status()['ready'])
        self.assertEqual(rover.speech.results, [])

    def test_no_id_person_bridge_uses_current_front_blocker_predicate(self):
        rover = self.rover_for('hardware')
        self.backend.person = True
        self.control('start')
        rover.tick()
        self.assertEqual(rover.speech_target, {'id': None})
        rover.speech.results = [dict(text='A current generated request to move.')]
        rover.tick()
        self.assertIn('A current generated request to move.', [event['text'] for event in rover.events])
        rover.speech_target = {'id': None}
        rover.speech.results = [dict(text='A stale request after the blocker left.')]
        self.backend.person = False
        rover.tick()
        self.assertNotIn('A stale request after the blocker left.', [event['text'] for event in rover.events])
        self.assertIsNone(rover.speech_target)

    def test_stop_reset_and_sensor_fault_clear_speech_target(self):
        rover = self.rover_for()
        for action in ('stop', 'reset'):
            with self.subTest(action=action):
                rover.speech_target = {'id': 7}
                rover.speech.results = [dict(text='An invalidated pending remark.')]
                self.control(action)
                self.assertIsNone(rover.speech_target)
                self.assertEqual(rover.speech.results, [])
        self.control('start')
        rover.speech_target = {'id': 7}
        self.backend.stale = True
        rover.tick()
        self.assertIsNone(rover.speech_target)
        self.assertFalse(rover.running)

    def test_shutdown_closes_every_resource_when_physical_stop_and_camera_close_fail(self):
        rover = self.rover_for()
        rover.mode = 'hardware'
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(self.backend, 'stop', side_effect=RuntimeError('physical stop unavailable')))
            camera = mocks.enter_context(patch.object(rover.perception, 'close', side_effect=RuntimeError('camera close failed')))
            explorer = mocks.enter_context(patch.object(rover.explorer, 'close'))
            speech = mocks.enter_context(patch.object(rover.speech, 'close'))
            backend = mocks.enter_context(patch.object(self.backend, 'close'))
            audio = mocks.enter_context(patch.object(rover.audio, 'close'))
            with self.assertLogs(server.log, level='ERROR') as logs:
                rover.close()
            self.assertTrue(any('physical stop unavailable' in line for line in logs.output))
            for resource in (camera, explorer, speech, backend, audio):
                resource.assert_called_once()
            self.assertFalse(rover.running)
            self.assertIsNone(rover.speech_target)
            rover.close()
            backend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
