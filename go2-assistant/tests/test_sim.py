"""Sensor-derived navigation, renderer ownership and actual torque actions."""
import math
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import pytest

from go2_assistant import sim as sim_module
from go2_assistant.sim import ASSETS, LIDAR_ANGLES, LIDAR_MAX, SimBackend


class EmptyVision:
    """Test camera sink: no semantic labels or fixture coordinates supplied."""
    def __init__(self):
        self.exposures = []
        self.target = None

    def submit(self, rgb, depth, camera, captured_at):
        self.exposures.append((rgb.copy(), depth.copy(), camera, captured_at))
        return True

    def poll(self):
        return {"detections": [], "room": None, "state": "ready"}

    def status(self):
        return {"state": "ready", "pending": False, "model": "test-empty"}

    def set_target(self, target):
        self.target = target

    def close(self, wait=False):
        pass


needs_assets = pytest.mark.skipif(not (ASSETS / "policy/policy.onnx").is_file(), reason="fetch pinned assets first")


@needs_assets
def test_measured_map_unknown_space_torque_navigation_and_rendering(tmp_path):
    vision = EmptyVision()
    backend = SimBackend(perception=vision, mapping_path=tmp_path / "measured-map.npz")
    try:
        observation = backend.observe()
        assert observation["healthy"] and observation["detections"] == []
        assert observation["room"] is None
        assert observation["map_source"] == "measured lidar rays"
        assert "person" not in backend.status()
        assert "supplied" not in str(backend.status())
        assert "exact" not in str(backend.status())
        snapshot = backend.map_snapshot()
        assert snapshot["observed_cells"] > 500
        assert snapshot["observed_cells"] < snapshot["size_cells"] ** 2 / 3
        assert not backend.mapper.free(8., 8.)
        with pytest.raises(ValueError, match="measured free space"):
            backend.navigate(8., 8.)
        # The same camera exposure goes to recognition, with only estimated pose
        # and fixed optical calibration, never an exact world-camera transform.
        image, depth, camera, stamp = vision.exposures[-1]
        assert image.shape == (360, 640, 3) and depth.shape == (360, 640)
        assert camera["capture_pose"] == observation["pose"]
        assert stamp == backend.capture_time
        assert camera["world_from_camera"].shape == (4, 4)
        image_before = backend.frame("robot")
        timestamp = backend.capture_time
        count = backend._render_counts["robot"]
        backend.set_person(2.3, -1.3)
        assert backend.frame("robot").tobytes() == image_before.tobytes()
        backend._advance_person(.2)
        mujoco.mj_forward(backend.sim.model, backend.sim.data)
        first, second = backend.frame("robot"), backend.frame("robot")
        assert image_before.tobytes() != first.tobytes()
        assert first.tobytes() == second.tobytes()
        assert backend._render_counts["robot"] == count + 1
        assert backend.capture_time == timestamp
        with ThreadPoolExecutor(max_workers=2) as clients:
            with pytest.raises(RuntimeError, match="thread that created"):
                clients.submit(backend.frame, "overview").result(timeout=5)
            assert clients.submit(backend.observe).result(timeout=5)["healthy"]

        frontiers = backend.search_points()
        assert frontiers and backend.search_points("unseen kitchen") == frontiers
        goal = min(frontiers, key=lambda p: math.dist((0, 0), p))
        assert backend.mapper.free(*goal)
        joints = backend.sim.data.qpos[backend.sim.qaddr].copy()
        actual_start = backend.sim.data.qpos[:2].copy()  # Test-only physical evidence.
        backend.navigate(*goal)
        movement = 0.
        for _ in range(250):
            backend.tick(.1)
            movement = max(movement, float(np.linalg.norm(backend.sim.data.qpos[backend.sim.qaddr] - joints)))
            if backend.state in {"arrived", "fault", "blocked"}:
                break
        assert backend.state == "arrived", backend.status()
        assert math.dist(tuple(backend.sim.data.qpos[:2]), tuple(actual_start)) > .3
        pose = backend.observe()["pose"]
        assert math.dist((pose["x"], pose["y"]), goal) < .3
        assert movement > .15 and backend.wall_contacts == 0
        backend.stop()
        assert backend.last_velocity == (0., 0., 0.)
        backend.sensor_fault = True
        backend.tick(.1)
        assert backend.state == "fault" and not backend.observe()["healthy"]
        assert backend.last_velocity == (0., 0., 0.)
    finally:
        backend.close()
    assert (tmp_path / "measured-map.npz").is_file()


@needs_assets
def test_changed_scene_wall_changes_lidar_map_and_route(monkeypatch):
    original = sim_module._scene_model

    def moved_wall(assets):
        model = original(assets)
        # Change the world before the backend takes its first sensor scan.
        # Closing a formerly open area must remove routes from the learned map.
        model.geom_pos[model.geom("home_wall_1").id, 0] = 1.0
        return model

    monkeypatch.setattr(sim_module, "_scene_model", moved_wall)
    backend = SimBackend(perception=EmptyVision())
    try:
        forward = int(np.argmin(np.abs(LIDAR_ANGLES)))
        assert backend._scan_ranges[forward] < .8
        assert backend._scan_ranges[forward] != LIDAR_MAX
        assert not backend.mapper.free(1.4, 0.)
        with pytest.raises(ValueError, match="measured free space"):
            backend.navigate(1.4, 0.)
        assert backend.observe()["detections"] == []
        assert backend.mapper.status()["occupied_cells"] > 0
    finally:
        backend.close()


@needs_assets
def test_bounded_spin_incremental_scan_rest_stand_and_cancellation():
    backend = SimBackend(perception=EmptyVision())
    try:
        start = backend.observe()["pose"]
        assert backend.perform_action("spin")["accepted"]
        total_yaw, previous_yaw = 0., start["yaw"]
        for _ in range(201):
            backend.tick(.1)
            yaw = backend.observe()["pose"]["yaw"]
            total_yaw += (yaw - previous_yaw + math.pi) % (2 * math.pi) - math.pi
            previous_yaw = yaw
            if backend.state in {"arrived", "fault"}:
                break
        assert backend.state == "arrived", backend.status()
        assert 5.9 < total_yaw < 6.7
        assert backend.elapsed < 20.5
        pose = backend.observe()["pose"]
        assert math.dist((pose["x"], pose["y"]), (start["x"], start["y"])) < .2
        assert backend.wall_contacts == 0 and backend.last_velocity == (0., 0., 0.)
        assert not backend.spin(turns=2.)["accepted"]
        backend.scan()
        assert backend.scan_remaining == pytest.approx(.8)
        for _ in range(50):
            backend.tick(.1)
            if backend.state in {"arrived", "fault"}:
                break
        assert backend.state == "arrived", backend.status()
        assert backend.perform_action("lie_down")["accepted"]
        for _ in range(81):
            backend.tick(.1)
            if backend.state in {"arrived", "fault"}:
                break
        assert backend.state == "arrived" and backend.status()["posture"] == "lying", backend.status()
        assert backend.sim.data.qpos[2] < .14
        backend.stop()
        backend.tick(.5)
        assert backend.sim.mode == "lying" and backend.observe()["healthy"]
        with pytest.raises(RuntimeError, match="stand up"):
            backend.navigate(0., 0.)
        assert backend.perform_action("stand_up")["accepted"]
        for _ in range(81):
            backend.tick(.1)
            if backend.state in {"arrived", "fault"}:
                break
        assert backend.state == "arrived" and backend.status()["posture"] == "standing", backend.status()
        assert backend.sim.data.qpos[2] > .22
        assert backend.perform_action("spin")["accepted"]
        backend.tick(.5)
        backend.stop()
        assert backend.action is None and backend.last_velocity == (0., 0., 0.)
        backend.tick(1.)
        assert backend.state == "idle" and np.linalg.norm(backend.sim.command) == 0.
        assert backend.perform_action("lie_down")["accepted"]
        backend.tick(.3)
        backend.stop()
        assert "settling safely" in backend.reason
        for _ in range(40):
            backend.tick(.1)
        assert backend.state == "idle" and backend.sim.mode == "lying", backend.status()
        assert backend.wall_contacts == 0 and backend.observe()["healthy"]
    finally:
        backend.close()


def test_live_route_revalidation_stops_or_replans_on_new_ranges():
    class ChangedMap:
        revision = 2
        alternate = []
        def pose(self):
            return {"x":0.,"y":0.,"yaw":0.}
        def line_free(self, start, end):
            return False
        def plan(self, x, y):
            return list(self.alternate)

    class Motion:
        owner = "test"
        stopped = False
        def stop(self, token):
            self.stopped = True

    backend = SimBackend.__new__(SimBackend)
    backend.mapper, backend.sim, backend.token = ChangedMap(), Motion(), "test"
    backend.path, backend.goal = [(1.,0.)], (1.,0.)
    backend._route_revision = 1
    backend.last_velocity = (.55,0.,0.)
    backend.state, backend.reason = "navigating", "test"
    backend._refresh_route()
    assert backend.state == "blocked" and backend.path == []
    assert backend.sim.stopped and backend.last_velocity == (0.,0.,0.)
    backend.path = [(1.,0.)]
    backend.mapper.revision = 3
    backend.mapper.alternate = [(0.,.8),(1.,.8),(1.,0.)]
    backend._refresh_route()
    assert backend.path == backend.mapper.alternate
    assert "Replanned" in backend.reason



def test_operator_person_moves_continuously_and_stays_still_without_target():
    import threading
    from types import SimpleNamespace
    backend = SimBackend.__new__(SimBackend)
    backend.lock = threading.RLock()
    backend.person_id = 0
    backend.sim = SimpleNamespace(data=SimpleNamespace(mocap_pos=np.array([[2.,-.8,0.]])))
    backend._visual_revision = 0
    backend._person_target = None
    backend._scene_floor_free = lambda x,y: abs(x)<4.3 and abs(y)<3.8
    backend._advance_person(.2)
    assert backend.get_person() == {"x":2., "y":-.8}
    backend.set_person(2.35,-.8)
    assert backend.get_person() == {"x":2., "y":-.8}  # Request never teleports.
    backend._advance_person(.2)
    assert backend.get_person()["x"] == pytest.approx(2.1)
    assert backend.get_person()["y"] == pytest.approx(-.8)
    assert backend._person_target is not None
    backend._advance_person(.5)
    assert backend.get_person()["x"] == pytest.approx(2.35)
    assert backend._person_target is None
    final = backend.get_person()
    backend._advance_person(1.)
    assert backend.get_person() == final
    with pytest.raises(ValueError):
        backend.set_person(float("nan"),0.)
    with pytest.raises(ValueError):
        backend.set_person(8.,0.)
    backend._scene_floor_free = lambda x,y: not 2.45<x<2.65
    with pytest.raises(ValueError,match="clear straight floor corridor"):
        backend.set_person(3.,-.8)
    assert backend.get_person() == final and backend._person_target is None
