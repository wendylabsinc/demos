import unittest
import numpy as np
from camera_input import target_red_mask
class RedLabelTests(unittest.TestCase):
    def test_white_label_stripes_preserve_only_red_measurements(self):
        rgb=np.zeros((100,150,3));rgb[20:80,30:70,0]=200
        rgb[43:49,30:70]=255
        m=target_red_mask(rgb,np.ones((100,150))*.5)
        self.assertGreater(m.sum(),100)
        self.assertFalse(m[43:49].any())
    def test_separate_cans_remain_ambiguous(self):
        rgb=np.zeros((100,150,3));rgb[20:80,20:50,0]=200;rgb[20:80,90:120,0]=200
        with self.assertRaisesRegex(RuntimeError,'ambiguous'):target_red_mask(rgb,np.ones((100,150))*.5)
    def test_partially_offscreen_red_can_is_rejected(self):
        rgb=np.zeros((100,150,3));rgb[65:,55:90,0]=200
        with self.assertRaisesRegex(RuntimeError,'clipped'):
            target_red_mask(rgb,np.ones((100,150))*.5)
if __name__=="__main__":unittest.main()


class TrackedCylinderTests(unittest.TestCase):
    def frame(self):
        import time
        depth=np.ones((120,160),dtype=float)
        rgb=np.zeros((120,160,3),dtype=np.uint8)
        rays=np.zeros((120,160,3));rays[:,:,2]=1
        for row_start in (25,65):
            ys,xs=np.mgrid[row_start:row_start+20,60:100]
            angle=np.linspace(2.,4.2,40)[None,:]
            rgb[ys,xs]=[200,0,0]
            depth[ys,xs]=.95
            rays[ys,xs,0]=(.35+.033*np.cos(angle))/.95
            rays[ys,xs,1]=(.18+.033*np.sin(angle))/.95
        return dict(color_bgr=rgb[:,:,::-1],depth_m=depth,rays=rays,
                    timestamp=time.monotonic(),meta=dict(camera_serial='test',frame_id=1))

    def config(self):
        return dict(camera_serial='test',camera_to_world=np.eye(4).tolist(),
                    can_radius_m=.033,can_height_m=.122,table_height_m=.762)

    def test_finger_split_label_merges_only_with_expected_cylinder(self):
        from camera_input import geometry
        frame=self.frame()
        with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):
            geometry(frame,self.config(),require_table=False)
        observation,vision=geometry(frame,self.config(),require_table=False,expected_center_xy=[.35,.18])
        np.testing.assert_allclose(vision['center_xy_m'],[.35,.18],atol=1e-6)
        self.assertLess(vision['fit_error_m'],.006)

    def test_off_target_red_is_rejected(self):
        from camera_input import geometry
        with self.assertRaisesRegex(RuntimeError,'absent/ambiguous'):
            geometry(self.frame(),self.config(),require_table=False,expected_center_xy=[.7,.18])

    def test_fragment_tracking_cannot_enable_initial_detection(self):
        from camera_input import geometry
        with self.assertRaisesRegex(ValueError,'only allowed after lifting'):
            geometry(self.frame(),self.config(),require_table=True,expected_center_xy=[.35,.18])

    def test_tracking_preserves_clipped_label_guard(self):
        from camera_input import geometry
        frame=self.frame();frame['color_bgr'][0:5,60:100]=[0,0,200]
        with self.assertRaisesRegex(RuntimeError,'clipped'):
            geometry(frame,self.config(),require_table=False,expected_center_xy=[.35,.18])

    def test_candidate_roi_accepts_shifted_cylinder_but_not_wider_grip(self):
        from camera_input import geometry
        frame=self.frame();frame['rays'][:,:,0]-=.035/frame['depth_m']
        _,vision=geometry(frame,self.config(),require_table=False,expected_center_xy=[.35,.18])
        np.testing.assert_allclose(vision['center_xy_m'],[.315,.18],atol=1e-6)
        frame=self.frame();frame['rays'][:,:,0]-=.05/frame['depth_m']
        with self.assertRaisesRegex(RuntimeError,'over 40 mm'):
            geometry(frame,self.config(),require_table=False,expected_center_xy=[.35,.18])
