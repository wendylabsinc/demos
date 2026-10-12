"""Candidate correction tests; the production displacement guard stays intact."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from offline.scene_reference import estimate_scene_reference, transform_point

class SceneReferenceCandidateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(727)
        self.reference=np.c_[rng.uniform(.25,.75,120),rng.uniform(-.35,.35,120),np.full(120,.762)]
        self.r=Rotation.from_euler('xyz',[.002,-.001,.017]).as_matrix()
        self.t=np.array([.003,-.001,.0005])
        self.source=(self.reference-self.t)@self.r

    def test_common_scene_drift_is_removed_with_outliers(self):
        source=self.source.copy();source[-20:]+=.05
        transform,metrics=estimate_scene_reference(source,self.reference)
        self.assertGreaterEqual(metrics['inliers'],100)
        can=np.array([.299644,.153844,.823])
        measured=(can-self.t)@self.r
        np.testing.assert_allclose(transform_point(measured,transform),can,atol=1e-12)

    def test_independent_can_displacement_survives_scene_correction(self):
        transform,_=estimate_scene_reference(self.source,self.reference)
        initial=np.array([.299644,.153844,.823]);moved=initial+np.array([.025,0,0])
        measured=(moved-self.t)@self.r
        self.assertAlmostEqual(np.linalg.norm(transform_point(measured,transform)[:2]-initial[:2]),.025,places=12)

    def test_invalid_ambiguous_and_narrow_landmarks_are_rejected(self):
        for a,b in [(self.source[:29],self.reference[:29]),
                    (self.source*np.nan,self.reference),
                    (np.tile([.3,.1,.762],(120,1)),np.tile([.3,.1,.762],(120,1)))]:
            with self.assertRaises(ValueError):estimate_scene_reference(a,b)
        bad=self.source+np.random.default_rng(46).normal(0,.04,self.source.shape)
        with self.assertRaisesRegex(ValueError,'ambiguous'):estimate_scene_reference(bad,self.reference)

    def test_excessive_scene_correction_and_improper_transform_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'bounded'):estimate_scene_reference(self.reference+[.05,0,0],self.reference)
        bad=np.eye(4);bad[0,0]=-1
        with self.assertRaisesRegex(ValueError,'proper rigid'):transform_point([.3,.1,.8],bad)

class RecordedSceneReferenceTests(unittest.TestCase):
    def test_saved_false_stop_is_corrected_but_real_displacement_still_stops(self):
        from pathlib import Path
        from types import SimpleNamespace
        import os, tempfile, threading
        from unittest.mock import patch
        os.environ.setdefault('HANDOFF_DATA',tempfile.mkdtemp(prefix='scene-replay-'))
        from runtime import Runtime
        path=Path(__file__).parent/'offline/validation/act-hardware/scene-reference-fixture.npz'
        with np.load(path,allow_pickle=False) as f:
            baseline=f['baseline']
            def publish(center):
                runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock()
                runtime.tracking_error=None;runtime.tracking_observation=None
                runtime.tracking_baseline=SimpleNamespace(center=baseline.copy())
                with patch('runtime.time.monotonic',return_value=100.1):
                    runtime.publish_tracking_observation(SimpleNamespace(center=center,timestamp=100.),{},True)
                np.testing.assert_array_equal(runtime.tracking_baseline.center,baseline)
            with self.assertRaisesRegex(RuntimeError,'outside the checked pickup path'):
                publish(f['can_54815'])
            for fid in [54813,54814,54815]:
                transform,_=estimate_scene_reference(f['source_'+str(fid)],f['reference_'+str(fid)])
                corrected=transform_point(f['can_'+str(fid)],transform)
                publish(corrected)
                self.assertLess(np.linalg.norm(corrected[:2]-baseline[:2]),.003)
                # An independent25mm target displacement is not used to fit
                # background alignment and cannot be canceled by that fit.
                moved=f['can_'+str(fid)]+transform[:3,:3].T@np.array([.025,0,0])
                with self.assertRaisesRegex(RuntimeError,'outside the checked pickup path'):
                    publish(transform_point(moved,transform))
