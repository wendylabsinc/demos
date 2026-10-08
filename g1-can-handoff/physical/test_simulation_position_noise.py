"""Recorded drift reproduces production checkpoint errors without hardware."""
from dataclasses import replace
import unittest
from unittest.mock import Mock
import numpy as np
from g1_can_handoff.perception import Observation
from pickup import execute
from simulation_position_noise import PositionNoise


class PositionNoiseTests(unittest.TestCase):
    def setUp(self):
        self.observation = Observation(np.array([.31, .155, .823]), .762, 123., .003, 100)
        self.vision = dict(center_xy_m=self.observation.center[:2].tolist(), top_quantile_z_m=.88)

    def test_real_approach_drift_reproduces_stop_before_reach_command(self):
        noise = PositionNoise(trace='approach-drift')
        calls = []
        def observe(phase, require_table, **kwargs):
            calls.append(phase)
            return noise.apply(self.observation, self.vision, phase, len(calls), **kwargs)
        baseline, vision = observe('planning', True)
        observe('plan-refresh', True)
        # Simulation has a second, owned preflight; this recording already held
        # ownership, so its recorded preflight is reused rather than skipped.
        baseline, vision = observe('planning', True)
        observe('plan-refresh', True)
        plant = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Can or table moved during approach'):
            execute(plant, (baseline, vision, np.zeros((4,3)), np.eye(3)), observe, lambda *a, **kw: None)
        self.assertEqual(calls[-1], 'reach')
        self.assertEqual(plant.move.call_count, 2)  # approach and align only
        self.assertEqual(plant.hand.call_count, 1)  # open, never close
        np.testing.assert_array_equal(self.observation.center, [.31, .155, .823])
        self.assertAlmostEqual(np.linalg.norm(noise.last_offset), .011616626484265183)

    def test_handover_bias_persists_in_later_phase_estimates(self):
        noise = PositionNoise(trace='handover')
        phases = ['planning','plan-refresh','planning','plan-refresh','pre-open','pre-stage','approach','align','reach']
        observations = [noise.apply(self.observation, self.vision, phase, i + 1)[0] for i, phase in enumerate(phases)]
        self.assertAlmostEqual(np.linalg.norm(observations[2].center - observations[0].center), .01040298684909448)
        self.assertLess(np.linalg.norm(observations[-1].center - observations[2].center), .01)
        self.assertGreater(observations[-1].center[0] - observations[0].center[0], .006)

    def test_seed_repeats_entire_correlated_trace_and_metadata(self):
        first, second = PositionNoise(seed=7), PositionNoise(seed=7)
        self.assertEqual(first.description, second.description)
        for i, phase in enumerate(['planning','align','reach','pre-lift','lift-verification']):
            a, av = first.apply(self.observation, self.vision, phase, i+1)
            b, bv = second.apply(self.observation, self.vision, phase, i+1)
            np.testing.assert_array_equal(a.center, b.center)
            self.assertEqual(av, bv)
            np.testing.assert_array_equal(a.center[:2], av['center_xy_m'])
            self.assertEqual(a.timestamp, self.observation.timestamp)
            self.assertEqual(a.table_height, self.observation.table_height)
            self.assertEqual(a.fit_error, self.observation.fit_error)
            self.assertEqual(a.center[2], self.observation.center[2])
            self.assertEqual(av['top_quantile_z_m'], self.vision['top_quantile_z_m'])
        self.assertTrue(av['simulation_position_noise']['extrapolated'])
        self.assertEqual(av['simulation_position_noise']['source_frame_id'], first.trace['samples'][-1]['frame_id'])

    def test_none_and_zero_scale_retain_ideal_estimate(self):
        for noise in (PositionNoise('none'), PositionNoise(scale=0)):
            for i, phase in enumerate(['planning','align','reach']):
                result, vision = noise.apply(self.observation,self.vision,phase,i+1)
                np.testing.assert_array_equal(result.center,self.observation.center)
                self.assertEqual(vision['center_xy_m'], self.vision['center_xy_m'])

    def test_large_noisy_estimate_cannot_bypass_postgrasp_tracking_gate(self):
        noise = PositionNoise(trace='approach-drift',scale=5)
        noise.apply(self.observation,self.vision,'reach',1)
        with self.assertRaisesRegex(ValueError,'outside the checked pickup path'):
            noise.apply(self.observation,self.vision,'pre-lift',2,expected_center_xy=self.observation.center[:2])

    def test_repeated_frame_does_not_resample_or_advance_phase(self):
        noise = PositionNoise(trace='handover')
        first = noise.apply(self.observation,self.vision,'planning',1)
        second = noise.apply(self.observation,self.vision,'planning',1)
        np.testing.assert_array_equal(first[0].center,second[0].center)
        self.assertEqual(first[1],second[1])
        with self.assertRaisesRegex(ValueError,'out-of-order'):
            noise.apply(self.observation,self.vision,'align',0)
        with self.assertRaisesRegex(ValueError,'Repeated'):
            noise.apply(self.observation,self.vision,'align',1)

    def test_invalid_noise_settings_and_estimates_are_rejected(self):
        for kwargs in ({'seed':-1},{'seed':True},{'seed':1.5},{'scale':float('nan')},
                       {'scale':float('inf')},{'scale':-1},{'trace':'invented'}):
            with self.assertRaises(ValueError): PositionNoise(**kwargs)
        with self.assertRaises(ValueError): PositionNoise('invented')
        with self.assertRaisesRegex(ValueError,'Invalid simulated'):
            PositionNoise().apply(replace(self.observation,center=np.array([np.nan,0,0])),self.vision,'planning',1)


if __name__ == '__main__': unittest.main()
