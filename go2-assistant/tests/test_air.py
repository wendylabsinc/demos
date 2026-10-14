"""Go2 Air tests use synthetic ROS-shaped inputs and never contact hardware."""
from types import SimpleNamespace
import math
import time

import numpy as np
import pytest

from go2_assistant.air import AirBackend, CaptureClock, point_cloud_xyz, quaternion_state, rgb_image
from go2_assistant.agent import tool_definitions
from go2_assistant.core import Assistant
from go2_assistant.mapping import OccupancyMapper, voxel_scan
from go2_assistant.memory import Memory
from go2_assistant.perception import ImageDetection, ModelObservation, RgbPerception


class Vision:
    def __init__(self):
        self.result = {"detections": [], "room": None}
        self.target = None

    def poll(self):
        return {"state": "ready", "provider": "test", "model": "test", "available": True,
                "pending": False, "error": None, "capture_age_s": 0., "captured_at": 0.,
                "capture_pose": {"x": 0., "y": 0., "yaw": 0.}, "latency_s": 0.,
                "vocabulary": "fixed learned classes", "spatial": False, **self.result}

    def status(self):
        return {key: value for key, value in self.poll().items() if key not in {"detections", "room"}}

    def set_target(self, target):
        self.target = target

    def submit(self, image, pose, captured_at):
        return True

    def close(self, wait=False):
        pass


def room_voxels(pose=(0., 0., 0.), obstacle=False):
    values = np.arange(-3., 3.001, .05)
    world = np.vstack((
        np.column_stack((values, np.full_like(values, -3.), np.zeros_like(values))),
        np.column_stack((values, np.full_like(values, 3.), np.zeros_like(values))),
        np.column_stack((np.full_like(values, -3.), values, np.zeros_like(values))),
        np.column_stack((np.full_like(values, 3.), values, np.zeros_like(values))),
    ))
    if obstacle:
        wall = np.arange(-.8, .801, .05)
        world = np.vstack((world, np.column_stack((np.full_like(wall, .7), wall, np.zeros_like(wall)))))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    relative = world[:, :2] - np.asarray(pose[:2])
    body = relative @ np.array(((c, -s), (s, c)))
    return np.column_stack((body, world[:, 2]))


def ready_backend(clock_value=None, *, writers=lambda: 1):
    clock_value = clock_value or [100.]
    commands = []
    backend = AirBackend("air-test", mapper=OccupancyMapper(robot_radius=.22, map_id="air-test"),
                         perception=Vision(), clock=lambda: clock_value[0],
                         command_sink=commands.append, writer_count=writers, start_ros=False)
    backend.update_odometry(0., 0., (0., 0., 0., 1.), 1_000_000_000, clock_value[0])
    backend.update_voxels(room_voxels(), 1_000_000_000, clock_value[0])
    return backend, commands, clock_value


def test_voxel_projection_keeps_missing_sectors_unknown():
    angles = np.linspace(-.3, .3, 20)
    points = np.column_stack((2 * np.cos(angles), 2 * np.sin(angles), np.zeros(len(angles))))
    ranges, scan_angles = voxel_scan(points)
    assert len(ranges) == len(scan_angles) == 72
    assert 0 < np.count_nonzero(np.isfinite(ranges)) < 72
    assert np.isnan(ranges[np.argmin(abs(scan_angles - math.pi / 2))])
    assert np.nanmin(ranges) == pytest.approx(2.)


def test_voxel_mapper_builds_routes_without_raw_scan_claims():
    mapper = OccupancyMapper(robot_radius=.22, map_id="air")
    result = mapper.update_voxels(room_voxels(), (0., 0., 0.), timestamp=1.)
    assert result["localization"]["method"] == "voxel_map_odometry"
    assert mapper.status()["source"] == "accumulated occupied voxels"
    assert mapper.plan(1., 0.)
    assert not mapper.free(3., 0.)


def test_air_starts_disarmed_and_drives_only_after_explicit_arm():
    backend, commands, _ = ready_backend()
    assert not backend.status()["armed"]
    assert backend.navigate(1., 0.)["accepted"] is False
    backend.tick()
    assert commands[-1] == (0., 0., 0.)
    assert backend.arm(timeout=0)["accepted"] is True
    assert backend.navigate(1., 0.)["accepted"] is True
    backend.tick()
    assert commands[-1][0] == .55
    assert commands[-1][1:] == (0., 0.)


def test_stale_voxels_latch_fault_and_zero_motion():
    backend, commands, clock = ready_backend()
    assert backend.arm(timeout=0)["accepted"]
    assert backend.navigate(1., 0.)["accepted"]
    backend.tick()
    clock[0] += .6
    backend.tick()
    assert backend.status()["state"] == "fault"
    assert backend.status()["armed"] is False
    assert commands[-1] == (0., 0., 0.)


def test_competing_cmd_vel_writer_prevents_arming():
    backend, commands, _ = ready_backend(writers=lambda: 2)
    result = backend.arm(timeout=0)
    assert result["accepted"] is False
    assert "publisher" in result["reason"]
    assert not any(command[0] for command in commands)


def test_updated_voxels_close_route_before_new_motion():
    backend, commands, clock = ready_backend()
    assert backend.arm(timeout=0)["accepted"]
    assert backend.navigate(1., 0.)["accepted"]
    backend.tick()
    assert commands[-1][0] == .55
    clock[0] += .1
    backend.update_odometry(0., 0., (0., 0., 0., 1.), 1_100_000_000, clock[0])
    backend.update_voxels(room_voxels(obstacle=True), 1_100_000_000, clock[0])
    backend.tick()
    assert backend.status()["state"] == "blocked"
    assert backend.status()["route"] == []
    assert commands[-1] == (0., 0., 0.)


def test_rgb_only_find_reports_visibility_without_inventing_position():
    backend, _, clock = ready_backend()
    backend.perception.result["detections"] = [{
        "label": "cup", "recognized_label": "cup", "track_id": "visual:cup:1",
        "bbox": [.4, .4, .6, .7], "confidence": .95, "timestamp": clock[0],
        "capture_pose": {"x": 0., "y": 0., "yaw": 0.}, "source": "vision:test",
        "spatial": False, "matches_target": True,
    }]
    assistant = Assistant(backend, Memory(":memory:", "air-test"), clock=lambda: clock[0])
    result = assistant.command("find_object", {"object": "cup"})
    assert result["result"]["state"] == "succeeded"
    assert "can see cup" in result["reply"]
    assert "Its map position" not in result["reply"]
    assert assistant.memory.objects() == []
    follow = assistant.command("follow_person")
    assert follow["error"] and "distance" in follow["reply"]


def test_rgb_perception_publishes_boxes_without_spatial_coordinates():
    class Detector:
        provider, model = "test", "rgb-test"
        def detect(self, image, target):
            return ModelObservation((ImageDetection("cup", (.2, .3, .6, .8), .9, True),), None)

    now = [10.]
    perception = RgbPerception(detector=Detector(), clock=lambda: now[0])
    try:
        assert perception.submit(np.zeros((16, 16, 3), dtype=np.uint8),
                                 {"x": 1., "y": 2., "yaw": .2}, now[0])
        deadline = time.monotonic() + 1
        result = perception.poll()
        while result["pending"] and time.monotonic() < deadline:
            time.sleep(.001)
            result = perception.poll()
        assert result["available"] and result["spatial"] is False
        assert result["detections"][0]["label"] == "cup"
        assert result["detections"][0]["spatial"] is False
        assert "x" not in result["detections"][0]
    finally:
        perception.close(wait=True)


def test_capture_clock_rejects_old_and_future_sources(monkeypatch):
    monkeypatch.delenv("GO2_SENSOR_CLOCK_OFFSET_SECONDS", raising=False)
    clock = CaptureClock(wall_clock_ns=lambda: 10_000_000_000,
                         monotonic=lambda: 50., max_age=.75)
    assert clock.capture(9_800_000_000) == pytest.approx(49.8)
    with pytest.raises(ValueError, match="stale"):
        clock.capture(9_000_000_000)
    with pytest.raises(ValueError, match="future"):
        clock.capture(10_100_000_000)


def point_cloud_message(points):
    fields = [SimpleNamespace(name=name, offset=index * 4, datatype=7, count=1)
              for index, name in enumerate("xyz")]
    data = np.asarray(points, dtype="<f4").tobytes()
    return SimpleNamespace(header=SimpleNamespace(frame_id="base_link"), width=len(points), height=1,
                           point_step=12, row_step=12 * len(points), data=data,
                           fields=fields, is_bigendian=False)


def test_ros_shaped_cloud_and_rgb_decoders_validate_frames_and_stride():
    points = room_voxels()[:20]
    np.testing.assert_allclose(point_cloud_xyz(point_cloud_message(points)), points, atol=1e-6)
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 24)
    image = SimpleNamespace(header=SimpleNamespace(frame_id="camera_optical_frame"),
                            encoding="rgb8", width=8, height=8, step=24,
                            data=pixels.tobytes())
    assert rgb_image(image).shape == (8, 8, 3)
    image.header.frame_id = "camera_link"
    with pytest.raises(ValueError):
        rgb_image(image)


def test_air_tool_schema_omits_unavailable_spatial_and_posture_actions():
    tools = {item["function"]["name"]: item["function"]
             for item in tool_definitions(AirBackend.capabilities)}
    assert "follow_person" not in tools and "come_here" not in tools
    assert tools["perform_action"]["parameters"]["properties"]["action"]["enum"] == ["spin"]
    assert {"explore", "find_object", "go_to_room"} <= set(tools)


def test_quaternion_leveling_preserves_yaw_and_removes_roll_pitch():
    yaw, level = quaternion_state(0., 0., math.sin(.25), math.cos(.25))
    assert yaw == pytest.approx(.5)
    np.testing.assert_allclose(level, np.eye(3), atol=1e-9)
