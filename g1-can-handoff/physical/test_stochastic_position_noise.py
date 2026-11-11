"""Fresh distributions, chronological validation and physical guard boundaries."""
import copy
import gzip
import json
from pathlib import Path
import unittest
import numpy as np
from g1_can_handoff.perception import Observation
from stochastic_position_noise import StochasticPositionNoise
from offline.fit_stochastic_noise import fit,segments
from offline.validate_stochastic_noise import metrics

BASE=Path(__file__).parent
PROFILE=BASE/'g1_can_handoff/stochastic_noise_profile.json'
CORPUS=BASE/'offline/validation/stochastic-noise-20261008/corpus.json.gz'


class StochasticNoiseTests(unittest.TestCase):
    def test_seeds_generate_new_sequences_and_same_seed_repeats(self):
        signatures=[]
        for seed in range(32):
            pair=[StochasticPositionNoise(seed=seed),StochasticPositionNoise(seed=seed)]
            curve=[]
            for frame,t in enumerate((0.,.2,1.,4.,10.),1):
                output=[]
                for model in pair:
                    model.advance(t,moving=frame>2)
                    output.append(model.offset('observe',frame))
                np.testing.assert_array_equal(output[0][0],output[1][0])
                self.assertEqual(output[0][1],output[1][1])
                curve.append(tuple(output[0][0]))
            signatures.append(tuple(curve))
        self.assertEqual(len(set(signatures)),32)

    def test_noise_does_not_move_truth_or_change_exposure_and_geometry_quality(self):
        model=StochasticPositionNoise(seed=1)
        observation=Observation(np.array([.31,.15,.823]),.762,10.,.003,100)
        vision={'center_xy_m':[.31,.15],'top_quantile_z_m':.88}
        model.advance(2.)
        noisy,updated=model.apply(observation,vision,'planning',1)
        np.testing.assert_array_equal(observation.center,[.31,.15,.823])
        self.assertEqual(noisy.timestamp,10.)
        self.assertEqual(noisy.table_height,.762)
        self.assertEqual(noisy.fit_error,.003)
        self.assertEqual(noisy.center[2],.823)
        np.testing.assert_array_equal(noisy.center[:2],updated['center_xy_m'])
        self.assertEqual(updated['top_quantile_z_m'],.88)

    def test_repeated_frame_reuses_jitter_and_nonadvancing_ids_are_rejected(self):
        model=StochasticPositionNoise()
        a=model.offset('planning',1);b=model.offset('planning',1)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1],b[1])
        with self.assertRaisesRegex(ValueError,'out-of-order'):model.offset('align',1)
        with self.assertRaisesRegex(ValueError,'monotonically'):model.advance(-1.)

    def test_conditional_regimes_change_from_arm_motion_and_occlusion_proxy(self):
        model=StochasticPositionNoise()
        for i,(moving,occluded,expected) in enumerate(((False,False,'stationary'),(True,False,'moving'),(True,True,'occluded'),(False,False,'stationary')),1):
            model.advance(float(i),moving=moving,occluded=occluded)
            self.assertEqual(model.offset('camera',i)[1]['regime'],expected)

    def test_fitted_jumps_persist_across_subsequent_observations(self):
        profile=json.loads(PROFILE.read_text())
        profile['session_scale_log_sigma']=0.
        for parameter in profile['regimes'].values():
            parameter.update(jitter_cov_m2=[[0,0],[0,0]],drift_cov_m2=[[0,0],[0,0]],jump_rate_hz=20.,jump_vectors_m=[[.003,.004]])
        model=StochasticPositionNoise(seed=1,profile=profile)
        model.advance(1.)
        first=model.offset('align',1)[0];second=model.offset('reach',2)[0]
        self.assertGreater(len(model.jump_events),0)
        np.testing.assert_array_equal(first,second)
        np.testing.assert_allclose(first,[.003*len(model.jump_events),.004*len(model.jump_events)])

    def test_ou_and_jitter_variance_matches_transition_distribution(self):
        profile=json.loads(PROFILE.read_text());profile['session_scale_log_sigma']=0.
        for parameter in profile['regimes'].values():
            parameter.update(jitter_cov_m2=[[1e-6,0],[0,1e-6]],drift_cov_m2=[[4e-6,0],[0,4e-6]],
                             drift_tau_s=2.,tau_at_search_boundary=False,jump_rate_hz=0.,jump_vectors_m=[])
        shifts=[]
        for seed in range(1500):
            model=StochasticPositionNoise(seed=seed,profile=profile)
            a=model.offset('a',1)[0];model.advance(1.);b=model.offset('b',2)[0]
            shifts.append(b-a)
        expected=2e-6+4e-6*(1-np.exp(-1))
        np.testing.assert_allclose(np.var(shifts,axis=0),[expected,expected],rtol=.12)

    def test_large_noise_still_hits_existing_40mm_tracking_gate(self):
        model=StochasticPositionNoise(seed=3,scale=1000)
        observation=Observation(np.array([.31,.15,.823]),.762,10.,.003,100)
        with self.assertRaisesRegex(ValueError,'outside the checked pickup path'):
            model.apply(observation,{},'pre-lift',1,expected_center_xy=observation.center[:2])

    def test_data_audit_keeps_unverified_stationarity_and_excludes_postgrasp(self):
        corpus=json.loads(gzip.decompress(CORPUS.read_bytes()))
        self.assertGreater(len(corpus['source_audit']),30)
        ids=[s['id'] for s in corpus['sessions']]
        self.assertEqual(len(ids),len(set(ids)))
        for session in corpus['sessions']:
            samples=session['samples']
            self.assertTrue(all(r['can_stationarity']=='pregrasp_candidate_unverified' for r in samples))
            self.assertTrue(all(a['t']<=b['t'] for a,b in zip(samples,samples[1:])))
            self.assertTrue(all(r['phase'] not in ('pre-lift','lift-verification') for r in samples))

    def test_validation_gate_retains_displacement_across_long_unobserved_gap(self):
        samples=[dict(t=0.,xy=[0.,0.],phase='planning',regime='stationary'),
                 dict(t=.2,xy=[.001,0.],phase='plan-refresh',regime='stationary'),
                 dict(t=15.,xy=[.012,0.],phase='reach',regime='stationary')]
        group=list(segments([dict(samples=samples)]))
        result=metrics(group,.003,[samples])
        self.assertEqual(result['phase_boundary_stop_fraction'],1.)
        self.assertEqual(result['large_step_count'],0)  # frequency cannot be inferred across the gap

    def test_invalid_profile_and_settings_are_rejected(self):
        for kwargs in ({'seed':True},{'scale':float('inf')},{'scale':-1},{'session_scale':0}):
            with self.assertRaises(ValueError):StochasticPositionNoise(**kwargs)
        for mutate in ('covariance','rate'):
            profile=json.loads(PROFILE.read_text())
            if mutate=='covariance':profile['regimes']['moving']['jitter_cov_m2']=[[-1,0],[0,1]]
            else:profile['regimes']['moving']['jump_rate_hz']=-1
            with self.assertRaises(ValueError):StochasticPositionNoise(profile=profile)


if __name__=='__main__':unittest.main()
