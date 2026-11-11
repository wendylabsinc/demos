"""Captured partial-arc regression; no hardware or freshness bypass."""
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import least_squares
from camera_input import fit_cylinder,cylinder_jacobian


class CylinderFitTests(unittest.TestCase):
    def test_captured_arc_avoids_mirrored_local_minimum(self):
        with np.load(Path(__file__).parent/'fixtures/cylinder-local-minimum.npz') as f:
            points=f['points'][:,:2]
        old=least_squares(lambda c:np.linalg.norm(points-c,axis=1)-.033,
                          points.mean(0),loss='soft_l1',f_scale=.002)
        self.assertGreater(np.sqrt(np.mean(old.fun**2)),.006)
        repaired=fit_cylinder(points,.033)
        self.assertLess(np.sqrt(np.mean(repaired.fun**2)),.003)
        np.testing.assert_allclose(repaired.x,[.32900333,.13378391],atol=1e-6)
        self.assertLess(repaired.nfev,old.nfev)

    def test_analytic_derivative_matches_finite_difference(self):
        points=np.random.default_rng(3).uniform(-1,1,(80,2));center=np.array([.3,.1])
        numerical=np.column_stack([(np.linalg.norm(points-center-np.eye(2)[j]*1e-6,axis=1)-
                                    np.linalg.norm(points-center+np.eye(2)[j]*1e-6,axis=1))/2e-6 for j in range(2)])
        np.testing.assert_allclose(cylinder_jacobian(center,points),numerical,atol=1e-8)

if __name__=='__main__':unittest.main()
