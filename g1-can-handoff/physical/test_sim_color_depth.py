"""Production color/depth acquisition and tracking in the contact simulator."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from sim_runtime import run


class ColorDepthSimulationTests(unittest.TestCase):
    def simulate(self, name, **kwargs):
        directory = os.environ.get('HANDOFF_SIM_ARTIFACT_DIR')
        output = None
        if directory:
            destination = Path(directory) / name
            destination.mkdir(parents=True, exist_ok=True)
            output = destination / 'result.json'
        with patch('yolo_client.YoloClient.detect', side_effect=AssertionError('Unexpected YOLO dependency')):
            return run(output=output, height=.762, identity_mode='color_depth', position_noise='none',
                       capture_animation=bool(directory), **kwargs)

    def test_current_color_depth_path_lifts_and_holds_without_yolo(self):
        result = self.simulate('color-depth-pickup')
        self.assertTrue(result['lift_verified'], result.get('error'))
        self.assertEqual(result['event_counts']['lift_observation'], 5)
        self.assertEqual(result['perception']['yolo_calls'], 0)
        self.assertIn('visual_depth', result['perception']['tracking_modes'])
        self.assertFalse(result['physical_pickup_verified'])

    def test_stale_alignment_frame_stops_before_alignment_commands(self):
        result = self.simulate('stale-alignment', fault=dict(phase='align', kind='stale', age_s=.6))
        self.assertFalse(result['lift_verified'])
        self.assertIn('stale', result['error'])
        self.assertEqual(result['failure_phase'], 'align')
        self.assertEqual(result['fault']['command_count_at_injection'], result['commands_at_failure'])
        self.assertEqual(result['stop_hold_commands'], 1)
        self.assertNotIn('align', result['event_counts'])

    def test_head_camera_color_depth_lifts_without_yolo(self):
        result = self.simulate('head-camera-pickup', camera='head')
        self.assertTrue(result['lift_verified'], result.get('error'))
        self.assertEqual(result['event_counts']['lift_observation'], 5)
        self.assertEqual(result['perception']['yolo_calls'], 0)
        self.assertFalse(result['physical_pickup_verified'])

    def test_lost_can_stops_before_alignment_commands(self):
        result = self.simulate('lost-can', fault=dict(phase='align', kind='missing_target'))
        self.assertFalse(result['lift_verified'])
        self.assertIn('absent', result['error'])
        self.assertEqual(result['failure_phase'], 'align')
        self.assertEqual(result['fault']['command_count_at_injection'], result['commands_at_failure'])
        self.assertEqual(result['stop_hold_commands'], 1)
        self.assertNotIn('align', result['event_counts'])


if __name__ == '__main__':
    unittest.main()
