"""Verify actual render buffers, geometry rays and evolving numerical fields."""
import copy
import threading
import cv2
import mujoco
import numpy as np
import pytest
from yard_drone.environment import CELL, YardEnvironment
from yard_drone.model import build_model
from yard_drone.navigation import unproject
from yard_drone.sensors import SensorRig, ranges, scene_state


def scene(position=(0, -6.3, 4.5)):
    model = build_model()
    data = mujoco.MjData(model)
    # Known calibration pose in a fixture. Application controls never assign it.
    data.qpos[:3] = position
    mujoco.mj_forward(model, data)
    return model, data, YardEnvironment(model, data)


def test_real_rgb_depth_capture_and_pose_changes():
    model, data, environment = scene()
    with SensorRig(model) as rig:
        clean = rig.capture(copy.copy(data), scene_state(model))
        assert clean["image"].shape == (240, 320, 3)
        assert clean["image"].dtype == np.uint8
        assert clean["depth"].shape == (240, 320)
        assert abs(float(clean["depth"][120, 160]) - 4.425) < .015
        assert np.allclose(clean["cameraPosition"], [0, -6.3, 4.425])
        environment.inject("fire", [0, -6.3])
        fire = rig.capture(copy.copy(data), scene_state(model))
        difference = np.abs(fire["image"].astype(float) - clean["image"])
        assert np.count_nonzero(difference.max(axis=2) > 30) > 300
        environment.inject("oil", [1, -6.0])
        oil = rig.capture(copy.copy(data), scene_state(model))
        assert np.count_nonzero(np.abs(oil["image"].astype(float) - fire["image"]).max(axis=2) > 20) > 100
        moved = copy.copy(data)
        moved.qpos[0] += .4
        moved_frame = rig.capture(moved, scene_state(model))
        assert np.linalg.norm(np.array(moved_frame["cameraPosition"]) - clean["cameraPosition"]) > .39
        assert np.mean(np.abs(moved_frame["image"].astype(float) - oil["image"])) > 1


def test_tracking_camera_observes_physical_worker_and_thread_ownership():
    model, data, _ = scene((0., -3., 4.5))
    with SensorRig(model) as rig:
        frame = rig.capture(copy.copy(data), scene_state(model), "tracking")
        image = frame["image"]
        yellow = (image[..., 0] > 130) & (image[..., 1] > 130) & (image[..., 2] < 110)
        assert np.count_nonzero(yellow) > 10
        failures = []
        def wrong_thread():
            try:
                rig.capture(copy.copy(data))
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join(timeout=2)
        assert len(failures) == 1 and "thread" in failures[0]


def test_unlabeled_ranges_respect_obstacle_occlusion():
    model, data, _ = scene((5., 0., 1.0))
    scan = ranges(model, data)
    assert len(scan) == 97
    assert abs(scan[0]["distance"] - 1.0) < .001
    assert abs(scan[-1]["distance"] - .97) < .001
    assert not any("label" in ray or "object" in ray or "geom" in ray for ray in scan)


def test_oil_conserves_volume_and_fire_consumes_fuel():
    model, data, environment = scene()
    environment.inject("oil", [-5., 4.])
    initial_oil = environment.oil.copy()
    volume = float(environment.oil.sum() * CELL ** 2)
    environment.inject("fire", [4., -4.])
    initial_fuel = float(environment.fuel.sum())
    initial_temperature = environment.temperature.copy()
    for _ in range(100):
        environment.step(.04)
    assert abs(float(environment.oil.sum() * CELL ** 2) - volume) < 1e-8
    assert np.max(np.abs(environment.oil - initial_oil)) > .001
    assert environment.fuel.sum() < initial_fuel and environment.burned_fuel_mass > 0
    assert np.max(np.abs(environment.temperature - initial_temperature)) > 20
    assert environment.temperature.max() > 510 and environment.smoke.sum() > 0
    snapshot = environment.snapshot()["hazardFields"]
    assert snapshot["time"] >= 3.99 and len(snapshot["cells"]) > 3
    for point in ([99, 0], [float("nan"), 0], [7, 0]):
        with pytest.raises(ValueError):
            environment.inject("oil", point)
    environment.clear()
    assert environment.oil.sum() == 0 and environment.smoke.sum() == 0


def test_worker_moves_by_finite_forces_and_is_stationary_by_default():
    model, data, environment = scene((0., -6.3, .12))
    initial = environment.worker.position.copy()
    for _ in range(100):
        environment.step(.002)
        mujoco.mj_step(model, data)
    assert np.linalg.norm(environment.worker.position - initial) < 1e-5
    before = data.qpos.copy()
    assert environment.move_worker(.7, .5)["accepted"]
    assert np.array_equal(data.qpos, before)
    for _ in range(2000):
        environment.step(.002)
        assert np.linalg.norm(data.qfrc_applied[environment.worker.dof_ids]) <= 75 * .9 + 1e-5
        mujoco.mj_step(model, data)
    assert np.linalg.norm(environment.worker.position[:2] - [2.7, .5]) < .035
    assert abs(environment.worker.position[2] - .9) < 1e-6
    assert not environment.move_worker(float("nan"), 0)["accepted"]


def test_worker_badges_decode_from_actual_camera_pixels():
    model, data, _ = scene((2., 0., 4.5))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    with SensorRig(model) as rig:
        roof = rig.capture(copy.copy(data), scene_state(model))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(roof["image"], cv2.COLOR_RGB2GRAY))
        assert ids is not None and 7 in ids.ravel()
        west = copy.copy(data)
        west.qpos[0] = -.5
        front = rig.capture(west, scene_state(model), "tracking")
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(front["image"], cv2.COLOR_RGB2GRAY))
        assert ids is not None and 7 in ids.ravel()


def test_far_worker_tracking_uses_more_real_pixels_and_correct_depth_calibration():
    # This roughly seven-metre acquisition geometry produced aliased 8px tags
    # in the VM's old320px camera. The extra samples come from rendering the
    # real scene, rather than resizing an already aliased frame.
    model, data, _ = scene((-4.76, 2.26, 4.5))
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    with SensorRig(model) as rig:
        survey = rig.capture(copy.copy(data), scene_state(model))
        assert survey['image'].shape == (240, 320, 3)
        for worker_y in (0., 3.):
            data.qpos[8] = worker_y
            mujoco.mj_forward(model, data)
            frame = rig.capture(copy.copy(data), scene_state(model), 'tracking')
            assert frame['image'].shape == (480, 640, 3)
            assert frame['depth'].shape == (480, 640)
            assert frame['intrinsics']['width'] == 640 and frame['intrinsics']['height'] == 480
            assert frame['intrinsics']['fx'] == pytest.approx(240.)
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame['image'], cv2.COLOR_RGB2GRAY))
            assert ids is not None and 7 in ids.ravel()
            quad = corners[np.flatnonzero(ids.ravel() == 7)[0]].reshape(4, 2)
            uv = (quad.min(axis=0) + quad.max(axis=0)) / 2
            observed = unproject(frame, *uv)
            badge_centres = np.array([[1.639, worker_y, 1.15], [2., worker_y, 1.691]])
            assert np.linalg.norm(badge_centres - observed, axis=1).min() < .12
        data.qpos[8] = 0.
        mujoco.mj_forward(model, data)
        # Alternating GL contexts must preserve the unchanged survey pixels.
        repeated = rig.capture(copy.copy(data), scene_state(model))
        assert np.max(np.abs(repeated['image'].astype(int) - survey['image'])) <= 1
