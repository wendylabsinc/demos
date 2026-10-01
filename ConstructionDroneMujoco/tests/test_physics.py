"""Integrate physical actions separately from language and visual inference."""
import time
import mujoco
import numpy as np
from yard_drone.model import GRAVITY, MASS, PHYSICS_HZ
from yard_drone.navigation import Navigator
from yard_drone.simulation import MAX_ACCELERATION, MAX_SPEED, Simulation
from yard_drone.yard import HOME, ZONES


def test_hover_rotor_lag_and_gravity(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert sim.model.nq == 9 and sim.model.nv == 8 and sim.model.nu == 4
    assert np.isclose(sim.model.opt.timestep, 1 / PHYSICS_HZ)
    before = sim.data.qpos.copy()
    assert sim.action({"action": "takeoff"})["accepted"]
    assert np.array_equal(sim.data.qpos, before)
    sim.run_headless(12)
    frame = sim.snapshot()
    assert abs(frame["position"][2] - 4.5) < .015
    assert np.linalg.norm(frame["velocity"]) < .02
    assert abs(sum(frame["rotors"]) - MASS * GRAVITY) < .05
    assert sim.obstacle_contacts == 0
    height = sim.data.qpos[2]
    sim.action({"action": "motors_off"})
    sim.run_headless(.002)
    assert sim.data.act.sum() > 0  # Finite motor response remains after cutoff.
    assert sim.data.ctrl.sum() == 0
    sim.run_headless(.248)
    assert .20 < height - sim.data.qpos[2] < .30
    assert sim.data.qvel[2] < -2.0


def test_grounded_land_and_physics_fault(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    assert sim.action({"action": "land"})["accepted"]
    sim.run_headless(2)
    assert sim.data.qpos[2] < .20 and sim.motors == "off"
    sim.error = "Physics loop failed"
    assert sim.snapshot()["ready"] is False
    assert sim.snapshot()["error"] == "Physics loop failed"
    assert not sim.action({"action": "takeoff"})["accepted"]


def test_zone_navigation_limits_and_landing(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    before = sim.data.qpos.copy()
    assert sim.action({"action": "goto", "zone": "A"})["accepted"]
    assert np.array_equal(before, sim.data.qpos)
    speeds, accelerations, tracking = [], [], []
    for index in range(30 * PHYSICS_HZ):
        sim.step()
        if index % 50 == 0:
            speeds.append(np.linalg.norm(sim.reference_velocity))
            accelerations.append(np.linalg.norm(sim.reference_acceleration))
            tracking.append(sim.error_now)
    assert np.linalg.norm(sim.data.qpos[:3] - ZONES["A"]["position"]) < .03
    assert max(speeds) <= MAX_SPEED + .001
    assert max(accelerations) <= MAX_ACCELERATION + .001
    assert max(tracking) < .08 and sim.obstacle_contacts == 0
    assert sim.action({"action": "land"})["accepted"]
    sim.run_headless(30)
    assert np.linalg.norm(sim.data.qpos[:3] - HOME) < .03
    assert sim.mode == "idle" and sim.motors == "off"
    assert sim.data.act.sum() < .001 and sim.obstacle_contacts == 0


def test_follow_requires_camera_and_stale_observation_holds(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    assert not sim.action({"action": "follow"})["accepted"]
    assert "camera" in sim.action({"action": "follow"})["message"].lower()
    sim.action({"action": "takeoff"})
    sim.run_headless(12)
    hover = sim.data.qpos[:3].copy()
    # Exercise stale-observation handling, with search coverage exhausted.
    # The controller must not fall back to the physical worker's hidden pose.
    sim._camera_time = float(sim.data.time)
    sim._camera_wall = time.monotonic()
    assert sim.action({"action": "follow"})["accepted"]
    sim.navigator.covered[:] = True
    sim._person = {"position": [5., 4., .9], "time": float(sim.data.time) - 3., "confidence": .9}
    assert sim.move_worker(.8, .5)["accepted"]
    sim.run_headless(8)
    assert np.linalg.norm(sim.data.qpos[:3] - hover) < .03
    assert sim._person is None and "holding" in sim.last_command
    assert sim.environment.worker.position[0] > 2.5


def test_navigation_routes_around_measured_obstacle():
    nav = Navigator()
    nav.seen[:] = True
    rays = [{"origin": [-3., y, 4.5], "direction": [1., 0., 0.], "distance": 3.}
            for y in np.arange(-3., 3.1, .5)]
    nav.observe_ranges(rays)
    path = nav.plan(np.array([-3., 0., 4.5]), np.array([3., 0., 4.5]))
    assert any(abs(point[1]) > 3. for point in path[:-1])
    blocked = nav.blocked(4.5)
    for start, end in zip(nav.path, nav.path[1:]):
        for alpha in np.linspace(0, 1, 80):
            p = np.array(start) * (1 - alpha) + np.array(end) * alpha
            assert not blocked[nav.cell(p)]


def test_new_obstacle_invalidates_path_and_visible_clearance_recovers():
    nav = Navigator()
    assert nav.path_safe([-3, 0, 4.5], [[3, 0, 4.5]])
    nav.observe_ranges([{"origin": [-2, 0, 4.5], "direction": [1, 0, 0], "distance": 2.}])
    assert not nav.path_safe([-3, 0, 4.5], [[3, 0, 4.5]])
    # A later unobstructed ground return directly observes the same cell. It
    # removes the old obstacle instead of retaining a historical maximum.
    nav.observe_ranges([{"origin": [0, 0, 4.5], "direction": [0, 0, -1], "distance": 4.5}])
    assert nav.height[nav.cell([0, 0])] == 0
    assert nav.path_safe([-3, 0, 4.5], [[3, 0, 4.5]])
    assert nav.path_safe([0, 0, 4.5], [[0, 0, .12]])
    assert not nav.path_safe([0, 0, 4.5], [[20, 0, 4.5]])


def test_mission_restart_discards_old_capture_and_can_capture_again(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    sim._camera_time = 0.
    assert sim.action({"action": "baseline"})["accepted"]
    old_id = sim.missions.mission["id"]
    sim._capture_busy = True
    sim._capture_request = {"missionId": old_id}
    assert sim.action({"action": "baseline"})["accepted"]
    new_id = sim.missions.mission["id"]
    assert new_id != old_id and not sim._capture_busy
    assert sim._capture_request is None
    assert not sim.missions.record({"missionId": old_id})
    # Feed the navigation event handler a stable-arrival measurement. Flight
    # dynamics have separate integration tests; this checks the capture race.
    sim.trajectory = None
    sim._navigation(sim._scan_goal.copy(), np.zeros(3))
    sim.data.time += .7
    sim._navigation(sim._scan_goal.copy(), np.zeros(3))
    assert sim._capture_busy
    assert sim._capture_request["missionId"] == new_id
