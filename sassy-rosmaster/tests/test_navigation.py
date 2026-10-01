import math
import time
import unittest
from sassy_rosmaster.types import Pose, Observation
from sassy_rosmaster.navigation import validate_observation, OccupancyMap, Personality, Navigator
from sassy_rosmaster.types import Command


def observation():
    return Observation(Pose(0,0,0),[6.]*180,captured_at=time.monotonic())

class SensorAndMapTests(unittest.TestCase):
    def test_sensor_faults_fail_closed(self):
        validate_observation(observation())
        for mutate in (lambda o:setattr(o,'captured_at',o.captured_at-1),
                       lambda o:o.ranges.__setitem__(0,float('nan')),
                       lambda o:o.ranges.__setitem__(0,0),
                       lambda o:setattr(o,'angle_increment',math.pi/180)):
            obs=observation()
            mutate(obs)
            with self.assertRaises(ValueError):
                validate_observation(obs)

    def test_no_return_does_not_invent_walls(self):
        grid=OccupancyMap()
        grid.update(observation())
        self.assertGreater(grid.export()['coverage'],50)
        self.assertNotIn(100,grid.export()['cells'])

    def test_person_honk_is_rate_limited_and_allows_detour(self):
        obs=observation()
        obs.people=[dict(distance=1.2,bearing=0)]
        personality=Personality()
        hold,events=personality.update(obs,100)
        self.assertFalse(hold)
        self.assertEqual([e[0] for e in events],['horn'])
        self.assertEqual(personality.update(obs,101)[1],[])
        self.assertFalse(personality.update(obs,103)[0])

    def test_model_goal_is_rechecked_against_current_sensor_obstacles(self):
        obs=observation()
        navigator=Navigator(seed=7)
        navigator.observe(obs,.05)
        candidate=navigator.goal_candidates(obs)[0]
        self.assertTrue(navigator.accept_goal(candidate,obs))
        self.assertEqual(navigator.goal,(candidate['x'],candidate['y']))
        angle=math.atan2(candidate['y'],candidate['x'])
        beam=round((angle-obs.angle_min)/obs.angle_increment)%180
        obs.ranges[beam]=.7
        navigator.observe(obs,.05)
        self.assertFalse(navigator.accept_goal(candidate,obs))
        navigator.yielding=True
        self.assertFalse(navigator.accept_goal(dict(x=1.2,y=0),observation()))

    def test_inflated_margin_escape_is_reverse_and_increases_clearance(self):
        obs=observation()
        obs.ranges[90]=.35
        navigator=Navigator()
        navigator.observe(obs,.05)
        command=navigator.command(obs)
        self.assertLess(command.speed,0)
        self.assertGreaterEqual(command.speed,-.14)
        previous=.35
        for x,y,_ in navigator._arc(obs,command.speed,command.steering,24):
            clearance=math.hypot(x-.35,y)
            self.assertGreaterEqual(clearance,previous-1e-6)
            previous=clearance
        self.assertGreater(previous,navigator.body_radius)

    def test_escape_requires_rear_space_and_no_hard_overlap(self):
        for distance,rear,collision in ((.29,6.,False),(.35,.6,False),(.35,6.,True)):
            with self.subTest(distance=distance,rear=rear,collision=collision):
                obs=observation()
                obs.ranges[90]=distance
                obs.ranges[0]=rear
                obs.collision=collision
                navigator=Navigator()
                navigator.observe(obs,.05)
                self.assertEqual(navigator.command(obs),Command())

    def test_yielding_keeps_a_rear_goal_until_person_is_clear(self):
        obs=observation()
        obs.people=[dict(distance=1.2,bearing=0)]
        navigator=Navigator()
        navigator.observe(obs,.05)
        navigator.yielding=True
        navigator.goal=(-.4,0.)
        command=navigator.command(obs)
        self.assertLess(navigator.goal[0],-1.)
        self.assertLess(command.speed,0.)
        # A rear goal cannot be replaced with a forward goal when backing space
        # is no longer sufficient and the pedestrian still needs room.
        obs.ranges[0]=1.
        navigator.observe(obs,.05)
        navigator.goal=(-.4,0.)
        self.assertEqual(navigator.command(obs),Command())

if __name__=='__main__':
    unittest.main()
