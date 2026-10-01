"""Real MuJoCo sensors, dynamic worlds, and controller replays."""

import math
import unittest

import mujoco
import numpy as np

from sassy_rosmaster.navigation import Navigator, validate_observation
from sassy_rosmaster.simulation import ARENA_WIDTH, LENGTH, Simulation
from sassy_rosmaster.types import Command

DT = .05


def replay(seconds, seed=7, population=3):
    simulation = Simulation(seed=seed, population=population)
    navigator = Navigator(seed=seed)
    previous = simulation.read().pose
    travel = 0.
    actor_start = {item['id']: (item['x'], item['y']) for item in simulation.scene()['people']}
    actor_travel = {key: 0. for key in actor_start}
    for index in range(round(seconds / DT)):
        observation = simulation.read()
        observation.captured_at = index * DT
        validate_observation(observation, now=index*DT)
        travel += math.hypot(observation.pose.x-previous.x, observation.pose.y-previous.y)
        previous = observation.pose
        # No simulator identities are available to the production planner.
        if observation.people:
            raise AssertionError('Simulation leaked semantic pedestrian labels')
        navigator.observe(observation, DT)
        simulation.step(navigator.command(observation, DT), DT)
        for item in simulation.scene()['people']:
            key = item['id']
            old = actor_start[key]
            actor_travel[key] += math.hypot(item['x']-old[0], item['y']-old[1])
            actor_start[key] = item['x'], item['y']
    return simulation, navigator, travel, actor_travel


class SimulationTests(unittest.TestCase):
    def test_lidar_has_no_semantic_labels_and_camera_snapshot_is_independent(self):
        simulation = Simulation(seed=7)
        observation = simulation.read()
        validate_observation(observation)
        self.assertEqual(len(observation.ranges), 180)
        self.assertAlmostEqual(observation.angle_increment*180, math.tau)
        self.assertTrue(all(math.isfinite(d) and 0 < d <= 6 for d in observation.ranges))
        self.assertEqual(observation.people, [])
        model, data, pose, captured_at = simulation.camera_snapshot()
        self.assertIs(model, simulation.model)
        self.assertAlmostEqual(pose.x, observation.pose.x)
        self.assertAlmostEqual(pose.y, observation.pose.y)
        self.assertAlmostEqual(pose.yaw, observation.pose.yaw)
        self.assertGreaterEqual(captured_at, observation.captured_at)
        self.assertEqual(model.camera('front_rgbd').id, 0)
        optical_forward = -data.cam_xmat[0].reshape(3,3)[:,2]
        self.assertGreater(optical_forward[2], .2, 'The low camera must look up toward people')
        before = data.qpos.copy()
        simulation.step(Command(.3, .1), .1)
        np.testing.assert_array_equal(data.qpos, before)
        self.assertFalse(np.array_equal(simulation.data.qpos, before))

    def test_explicit_seed_reproduces_world_and_reset_samples_fresh_world(self):
        first, second = Simulation(seed=91), Simulation(seed=91)
        self.assertEqual(first.scene(), second.scene())
        old_layout = first.scene()['obstacles']
        old_model = first.model
        first.reset()
        second.reset()
        self.assertEqual(first.scene(), second.scene())
        self.assertNotEqual(first.scene()['obstacles'], old_layout)
        self.assertIsNot(first.model, old_model)
        # Entropy is the runtime default, not a built-in replay seed.
        self.assertIsNone(Simulation(population=0).seed)
        self.assertNotEqual(Simulation(population=0).scene()['scenario_id'],
                            Simulation(population=0).scene()['scenario_id'])

    def test_real_ray_cast_measures_actor_surface_without_identifying_it(self):
        simulation = Simulation(seed=7, population=0)
        simulation.data.qpos[:3] = (-5., -3.5, 0.)
        mujoco.mj_forward(simulation.model, simulation.data)
        empty_range = simulation.read().ranges[90]
        self.assertTrue(simulation.add_person())
        # A controlled physical pose isolates the sensor assertion from the
        # autonomous actor's random starting position and current intention.
        address = simulation._person_qpos[0]
        simulation.data.qpos[address:address+3] = (-3.4, -3.5, 0.)
        mujoco.mj_forward(simulation.model, simulation.data)
        observation = simulation.read()
        self.assertAlmostEqual(observation.ranges[90], 1.6-.23, places=5)
        self.assertLess(observation.ranges[90], empty_range)
        self.assertEqual(observation.people, [])

    def test_physical_wall_blocks_drive_and_reverse_recovers(self):
        simulation = Simulation(seed=7, population=0)
        # Procedural objects leave an open perimeter, independently of seed.
        simulation.data.qpos[:3] = (5., -4.5, 0.)
        mujoco.mj_forward(simulation.model, simulation.data)
        for _ in range(200):
            simulation.step(Command(.65, 0.), DT)
        stopped = simulation.read().pose
        self.assertGreater(stopped.x, 6.5)
        self.assertLessEqual(stopped.x+LENGTH/2, ARENA_WIDTH/2+.005)
        self.assertGreaterEqual(simulation.collision_count, 1)
        self.assertLess(simulation.max_penetration, .01)
        for _ in range(80):
            simulation.step(Command(-.3, 0.), DT)
        self.assertLess(simulation.read().pose.x, stopped.x-.5)
        self.assertFalse(simulation.read().collision)

    def test_autonomous_pedestrians_move_without_external_routes_or_contacts(self):
        simulation = Simulation(seed=19)
        initial = {p['id']: (p['x'], p['y']) for p in simulation.scene()['people']}
        travels = dict.fromkeys(initial, 0.)
        for _ in range(600):
            simulation.step(Command())
            for person in simulation.scene()['people']:
                key = person['id']
                old = initial[key]
                travels[key] += math.hypot(person['x']-old[0], person['y']-old[1])
                initial[key] = person['x'], person['y']
        self.assertTrue(all(distance > 2. for distance in travels.values()))
        self.assertEqual(simulation.collision_count, 0)
        self.assertEqual(simulation.actor_contact_count, 0)
        self.assertTrue(simulation.add_person())
        self.assertEqual(len(simulation.scene()['people']), 4)
        simulation.clear_people()
        self.assertEqual(simulation.scene()['people'], [])

    def test_two_minute_procedural_exploration_with_moving_people_has_no_contacts(self):
        simulation, navigator, travel, people_travel = replay(120, seed=7)
        self.assertEqual(simulation.collision_count, 0)
        self.assertEqual(simulation.actor_contact_count, 0)
        self.assertGreater(travel, 15.)
        self.assertGreater(len(navigator.visits), 50)
        self.assertGreater(navigator.map.export()['coverage'], 75.)
        self.assertTrue(all(distance > 5. for distance in people_travel.values()))

    def test_distinct_random_world_explores_with_dynamic_obstacles(self):
        simulation, navigator, travel, people_travel = replay(60, seed=19)
        self.assertEqual(simulation.collision_count, 0)
        self.assertEqual(simulation.actor_contact_count, 0)
        self.assertGreater(travel, 10.)
        self.assertGreater(len(navigator.visits), 30)
        self.assertTrue(all(distance > 3. for distance in people_travel.values()))

    def test_ahead_scenario_is_visible_and_autonomous_and_rejects_occluded_spawn(self):
        simulation = Simulation(seed=7, population=0)
        self.assertTrue(simulation.add_person(ahead=True))
        person = simulation.scene()['people'][0]
        initial = person['x'], person['y']
        observation = simulation.read()
        dx, dy = person['x']-observation.pose.x, person['y']-observation.pose.y
        bearing = math.atan2(dy, dx)-observation.pose.yaw
        self.assertLess(abs(math.atan2(math.sin(bearing), math.cos(bearing))), .2)
        self.assertGreater(math.hypot(dx, dy), 2.2)
        self.assertLess(observation.ranges[90], 2.5, 'Actual lidar must see the spawned actor')
        self.assertEqual(observation.people, [])
        navigator = Navigator(seed=7)
        rover_travel = actor_travel = 0.
        previous = observation.pose
        for index in range(600):
            observation = simulation.read()
            observation.captured_at = index*DT
            rover_travel += math.hypot(observation.pose.x-previous.x,
                                       observation.pose.y-previous.y)
            previous = observation.pose
            navigator.observe(observation, DT)
            simulation.step(navigator.command(observation, DT))
            person = simulation.scene()['people'][0]
            actor_travel += math.hypot(person['x']-initial[0], person['y']-initial[1])
            initial = person['x'], person['y']
        self.assertGreater(rover_travel, 5.)
        self.assertGreater(actor_travel, 2.)
        self.assertEqual(simulation.collision_count, 0)
        self.assertEqual(simulation.actor_contact_count, 0)

        # Endpoints beyond this table are geometrically empty, but an actor
        # spawned there would be occluded from the current camera and lidar.
        blocked = Simulation(seed=7, population=0)
        table = next(item for item in blocked.obstacles if item['kind']=='box')
        blocked.data.qpos[:3] = (table['x']-table['w']/2-.65, table['y'], 0.)
        mujoco.mj_forward(blocked.model, blocked.data)
        self.assertAlmostEqual(blocked.read().ranges[90], .65)
        self.assertFalse(blocked.add_person(ahead=True), 'A hidden spawn must be rejected')
        self.assertEqual(blocked.scene()['people'], [])


if __name__ == '__main__':
    unittest.main()
