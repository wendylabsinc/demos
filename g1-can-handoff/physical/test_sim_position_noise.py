"""MuJoCo RGB-D + production tracker must reproduce recorded estimate drift."""
import json
import os
from pathlib import Path
import unittest
import numpy as np
from sim_runtime import run

FIXTURE = Path(__file__).parent / 'offline/validation/position-noise-20261008/setup.json'


class NoisyPerceptionSimulationTests(unittest.TestCase):
    def test_real_seeded_scene_stops_at_recorded_reach_drift(self):
        setup = json.loads(FIXTURE.read_text())
        output = None
        directory = os.environ.get('HANDOFF_SIM_ARTIFACT_DIR')
        if directory:
            folder = Path(directory) / 'recorded-position-drift'
            folder.mkdir(parents=True,exist_ok=True)
            output = folder / 'result.json'
        result = run(output=output,setup=setup,identity_mode='color_depth',
                     noise_trace='approach-drift',capture_animation=bool(directory))
        self.assertEqual(result.get('error'),'Can or table moved during approach; replanning required')
        self.assertEqual(result['failure_phase'],'reach')
        self.assertFalse(result['lift_verified'])
        self.assertFalse(result['physical_pickup_verified'])
        self.assertNotIn('reach',result['event_counts'])
        self.assertNotIn('close',result['event_counts'])
        self.assertEqual(result['stop_hold_commands'],1)
        self.assertEqual(result['perception']['yolo_calls'],0)
        cameras = [e for e in result['events'] if e['phase']=='camera']
        baseline = [e['vision'] for e in cameras if e['phase_name']=='planning'][-1]
        reach = cameras[-1]['vision']
        shift = np.linalg.norm(np.array(reach['center_xy_m'])-baseline['center_xy_m'])
        self.assertAlmostEqual(shift,.011616626484265183,delta=.0001)
        # No contact or physical-can movement is required to produce this stop.
        np.testing.assert_allclose(result['truth_can_position_m'][:2],setup['can_center_world_m'][:2],atol=.0001)
        self.assertEqual(result['simulation_contact_audit']['pregrasp_contact_steps'],0)
        self.assertIn('simulation_position_noise.py',result['source_sha256'])
        self.assertIn('g1_can_handoff/position_noise_profile.json',result['source_sha256'])


if __name__ == '__main__': unittest.main()
