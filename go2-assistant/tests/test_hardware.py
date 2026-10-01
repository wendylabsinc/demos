"""Hardware adapter tests use measured-input fixtures and recorded HTTP calls."""

import copy
import unittest
from urllib.parse import parse_qs, urlparse

from go2_assistant.hardware import HardwareBackend


class FakeInstallation:
    def __init__(self):
        self.wall = 1_790_000_000.0
        self.monotonic = 100.0
        self.calls = []
        self.observation = {
            "map_id": "commissioned-home-v3", "frame_id": "map",
            "timestamp": self.wall, "healthy": True, "reason": "",
            "localization": {"valid": True, "calibrated": True},
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "detections": [{"label": "person", "track_id": "owner-1",
                            "x": 2.0, "y": 0.0, "confidence": .95,
                            "timestamp": self.wall - .2}],
        }
        self.navigator_health = {
            "ready": True, "reason": "", "lidar_mount": {"measured_on_this_robot": True},
            "geofence": {"enabled": True},
            "pose": {"x": 0.0, "y": 0.0, "frame": "map", "age_s": .1, "usable": True},
        }
        self.motion_health = {
            "status": "ok", "forward_guard": {
                "installed": True, "have_scan": True, "age_s": .1,
                "stale_after_s": .5, "mount": {"measured_on_this_robot": True},
            },
        }
        self.navigator_state = {"mode": "idle", "explore": {
            "goals_succeeded": 2, "goals_failed": 1, "goal_active": False,
        }}
        self.pause_error = None
        self.stop_error = None
        self.pause_acknowledged = True
        self.pause_readback = True
        self.goal_accepted = True
        self.goal = None

    def transport(self, method, url, timeout):
        self.calls.append((method, url, timeout))
        parsed = urlparse(url)
        if parsed.netloc == "perception:8127":
            return copy.deepcopy(self.observation)
        if parsed.netloc == "navigator:3202":
            if parsed.path == "/health":
                return copy.deepcopy(self.navigator_health)
            if parsed.path == "/state":
                return copy.deepcopy(self.navigator_state)
            if parsed.path == "/pause":
                if self.pause_error:
                    raise self.pause_error
                if self.pause_acknowledged and self.pause_readback:
                    self.navigator_state["mode"] = "paused"
                return {"mode": "paused" if self.pause_acknowledged else "goal"}
            if parsed.path == "/goal":
                args = parse_qs(parsed.query)
                self.goal = (float(args["x"][0]), float(args["y"][0]))
                if self.goal_accepted:
                    self.navigator_state["mode"] = "goal"
                return {"accepted": self.goal_accepted, "reason": "outside geofence"}
        if parsed.netloc == "motion:3201":
            if parsed.path == "/health":
                return copy.deepcopy(self.motion_health)
            if parsed.path == "/stop":
                if self.stop_error:
                    raise self.stop_error
                return {"stopped": True}
        raise AssertionError(f"Unexpected HTTP call: {method} {url}")

    def backend(self):
        return HardwareBackend(
            "http://perception:8127/observations", "commissioned-home-v3",
            "http://navigator:3202", "http://motion:3201", transport=self.transport,
            clock=lambda: self.monotonic, wall_clock=lambda: self.wall,
        )

    def paths(self):
        return [(method, urlparse(url).netloc, urlparse(url).path) for method, url, _ in self.calls]


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.installation = FakeInstallation()
        self.backend = self.installation.backend()

    def arm_and_navigate(self):
        self.assertTrue(self.backend.arm()["accepted"])
        self.assertTrue(self.backend.navigate(1.0, 0.0)["accepted"])

    def test_hardware_starts_disarmed_and_never_moves_on_observation_or_arm(self):
        self.assertFalse(self.backend.status()["armed"])
        self.assertFalse(self.backend.navigate(1, 0)["accepted"])
        self.assertEqual(self.installation.calls, [])
        self.assertTrue(self.backend.observe()["healthy"])
        self.assertTrue(self.backend.arm()["accepted"])
        self.assertTrue(all(method == "GET" for method, _, _ in self.installation.calls))
        self.assertEqual(self.backend.status()["state"], "idle")

    def test_normalized_times_preserve_capture_age_and_drop_old_detections(self):
        self.installation.observation["timestamp"] -= .3
        old = dict(self.installation.observation["detections"][0])
        old["timestamp"] -= 4
        self.installation.observation["detections"].append(old)
        observation = self.backend.observe()
        self.assertAlmostEqual(observation["timestamp"], 99.7, places=5)
        self.assertEqual(len(observation["detections"]), 1)
        self.assertAlmostEqual(observation["detections"][0]["timestamp"], 99.8, places=5)

    def test_replans_confirm_pause_and_stop_before_each_goal(self):
        self.arm_and_navigate()
        self.installation.calls.clear()
        self.assertTrue(self.backend.navigate(1.5, .5)["accepted"])
        paths = self.installation.paths()
        pause = paths.index(("POST", "navigator:3202", "/pause"))
        stop = paths.index(("POST", "motion:3201", "/stop"))
        goal = paths.index(("POST", "navigator:3202", "/goal"))
        readback = paths.index(("GET", "navigator:3202", "/state"))
        self.assertLess(pause, readback)
        self.assertLess(readback, goal)
        self.assertLess(stop, goal)
        self.assertEqual(self.installation.goal, (1.5, .5))

    def test_failed_pause_still_stops_motion_and_never_sends_new_goal(self):
        self.arm_and_navigate()
        self.installation.calls.clear()
        self.installation.pause_error = OSError("navigator unreachable")
        result = self.backend.navigate(2, 0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["state"], "fault")
        self.assertFalse(result["armed"])
        self.assertIn(("POST", "motion:3201", "/stop"), self.installation.paths())
        self.assertNotIn(("POST", "navigator:3202", "/goal"), self.installation.paths())

    def test_pause_ack_without_matching_readback_cannot_replace_goal(self):
        self.arm_and_navigate()
        self.installation.calls.clear()
        self.installation.pause_readback = False
        result = self.backend.navigate(2, 0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["state"], "fault")
        self.assertNotIn(("POST", "navigator:3202", "/goal"), self.installation.paths())

    def test_stale_pose_faults_disarms_and_attempts_both_stops(self):
        self.arm_and_navigate()
        self.installation.calls.clear()
        self.installation.wall += 1.0
        self.installation.monotonic += 1.0
        state = self.backend.tick()
        self.assertEqual(state["state"], "fault")
        self.assertFalse(state["armed"])
        self.assertIn("stale", state["reason"])
        self.assertIn(("POST", "navigator:3202", "/pause"), self.installation.paths())
        self.assertIn(("POST", "motion:3201", "/stop"), self.installation.paths())

    def test_map_identity_change_stops_current_task(self):
        self.arm_and_navigate()
        self.installation.observation["map_id"] = "new-origin"
        self.assertFalse(self.backend.observe()["healthy"])
        self.assertEqual(self.backend.status()["state"], "fault")
        self.assertFalse(self.backend.status()["armed"])

    def test_missing_map_or_uncommissioned_localization_cannot_arm(self):
        for mutation in (lambda obs: obs.pop("map_id"),
                         lambda obs: obs["localization"].update(calibrated=False),
                         lambda obs: obs["localization"].update(valid=False),
                         lambda obs: obs.update(frame_id="odom"),
                         lambda obs: obs["pose"].update(x=float("nan")),
                         lambda obs: obs.update(timestamp=self.installation.wall + 1)):
            with self.subTest(mutation=mutation):
                installation = FakeInstallation()
                mutation(installation.observation)
                backend = installation.backend()
                self.assertFalse(backend.arm()["accepted"])
                self.assertTrue(all(method == "GET" for method, _, _ in installation.calls))

    def test_uncommissioned_lidar_or_stale_motion_guard_cannot_arm(self):
        self.installation.navigator_health["lidar_mount"]["measured_on_this_robot"] = False
        self.assertFalse(self.backend.arm()["accepted"])
        self.installation.navigator_health["lidar_mount"]["measured_on_this_robot"] = True
        self.installation.motion_health["forward_guard"]["age_s"] = 2.0
        self.assertFalse(self.backend.arm()["accepted"])

    def test_stop_failure_retains_fault_and_disarms(self):
        self.arm_and_navigate()
        self.installation.stop_error = OSError("motion unreachable")
        result = self.backend.stop()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["state"], "fault")
        self.assertFalse(result["armed"])

    def test_idle_without_measured_arrival_is_blocked(self):
        self.arm_and_navigate()
        self.installation.navigator_state["mode"] = "idle"
        self.installation.navigator_state["explore"]["goals_succeeded"] += 1
        self.assertEqual(self.backend.tick()["state"], "blocked")

    def test_measured_arrival_requires_success_when_result_counters_exist(self):
        self.arm_and_navigate()
        self.installation.observation["pose"]["x"] = .95
        self.installation.navigator_health["pose"]["x"] = .95
        self.installation.navigator_state["mode"] = "idle"
        self.assertEqual(self.backend.tick()["state"], "blocked")

    def test_failed_goal_cannot_claim_arrival_even_if_pose_is_close(self):
        self.arm_and_navigate()
        self.installation.observation["pose"]["x"] = .95
        self.installation.navigator_health["pose"]["x"] = .95
        self.installation.navigator_state["mode"] = "idle"
        self.installation.navigator_state["explore"]["goals_failed"] += 1
        self.assertEqual(self.backend.tick()["state"], "blocked")

    def test_arrival_needs_measured_pose_and_success_then_stops(self):
        self.arm_and_navigate()
        self.installation.calls.clear()
        self.installation.observation["pose"]["x"] = .95
        self.installation.navigator_health["pose"]["x"] = .95
        self.installation.navigator_state["mode"] = "idle"
        self.installation.navigator_state["explore"]["goals_succeeded"] += 1
        self.assertEqual(self.backend.tick()["state"], "arrived")
        self.assertIn(("POST", "motion:3201", "/stop"), self.installation.paths())

    def test_goal_rejection_faults_and_disarms(self):
        self.assertTrue(self.backend.arm()["accepted"])
        self.installation.goal_accepted = False
        result = self.backend.navigate(20, 0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["state"], "fault")
        self.assertIn("outside geofence", result["reason"])

    def test_tick_never_integrates_pose_from_elapsed_time(self):
        self.arm_and_navigate()
        self.backend.tick(1000)
        self.assertEqual(self.backend.observe()["pose"], {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.assertEqual(self.backend.status()["state"], "navigating")
        self.assertIsNone(self.backend.frame())

    def test_every_http_call_has_a_short_timeout(self):
        self.arm_and_navigate()
        self.backend.tick()
        self.assertTrue(all(0 < timeout <= .4 for _, _, timeout in self.installation.calls))


if __name__ == "__main__":
    unittest.main()
