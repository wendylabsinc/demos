"""Reproduce discovered startup-clearance and noisy pre-close guard stops."""
import json
from pathlib import Path
import unittest
from sim_runtime import run

ROOT=Path(__file__).parent


class CampaignFailureTests(unittest.TestCase):
    def setup_input(self):
        return json.loads((ROOT/'offline/validation/position-noise-20261008/setup.json').read_text())

    def test_initial_wrist_clearance_rejected_before_any_commands(self):
        setup=self.setup_input();setup['can_center_world_m'][0]-=.015;setup['can_center_world_m'][1]+=.005
        result=run(setup=setup,identity_mode='color_depth',noise_seed=2002,planner_mode='tuned')
        self.assertFalse(result['lift_verified'])
        self.assertIn('Planned finger trajectory intersects',result['error'])
        self.assertEqual(result['command_count'],0)
        collisions=[e['collision'] for e in result['events'] if e['phase']=='finger_trajectory_collision']
        self.assertTrue(collisions)
        self.assertEqual(set(collisions[0]['bodies']),{'can','left_wrist_yaw_link'})
        self.assertGreater(collisions[0]['distance_m'],0.)
        self.assertLess(collisions[0]['distance_m'],.01)

    def assert_preclose_stop(self,seed,policy):
        result=run(setup=self.setup_input(),identity_mode='color_depth',noise_seed=seed,noise_scale=2.,planner_mode=policy)
        self.assertFalse(result['lift_verified'])
        self.assertIn('moved during final reach; refusing to close',result['error'])
        self.assertEqual(result['failure_phase'],'pre-close')
        self.assertGreater(result['commands_at_failure'],0)
        self.assertEqual(result['stop_hold_commands'],1)
        self.assertNotIn('close',result['event_counts'])
        self.assertNotIn('lift',result['event_counts'])
        self.assertFalse(result['physical_pickup_verified'])
        self.assertEqual(result['perception']['yolo_calls'],0)

    def test_legacy_double_noise_stop_does_not_close_or_lift(self):
        self.assert_preclose_stop(3004,'legacy')

    def test_tuned_double_noise_stop_does_not_close_or_lift(self):
        self.assert_preclose_stop(3009,'tuned')


if __name__=='__main__':unittest.main()
