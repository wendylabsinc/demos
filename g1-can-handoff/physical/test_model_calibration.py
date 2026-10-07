import copy
import time
import unittest

import numpy as np
from physical_core import calibration, calibration_start_guard, calibrated_scene


class ModelCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(commissioned=True, device='unitree-g1-nx-2.local', camera_serial='camera',
            calibration_method='operator_accepted_factory_table', operator_accepted=True,
            factory_mount_position_confirmed=True, stationary_base_confirmed=True,
            independent_validation=False, table_plane_rmse_m=.0012, tilt_correction_deg=5.8,
            accepted_at_unix_s=time.time(), source_frame_id=100, factory_mount_source='official URDF',
            assumptions=['factory position'], reference_base_q=[0]*15, reference_imu_rpy=[0]*3,
            camera_to_world=np.eye(4).tolist(), pelvis_to_world=np.eye(4).tolist(),
            table_height_m=.762, table_center_xy_m=[5.23, 0], table_half_size_xy_m=[5, 5],
            table_near_edge_conservative_x_m=.23, can_radius_m=.033, can_height_m=.122,
            operator_with_stop=True, support='factory_balance')
        self.snapshot = {name:dict(time=time.monotonic(),q=[0]*n,dq=[0]*n,tau_est=[0]*n,
            temperature_max=30,faults=[0]*n,imu_rpy=[0,0,0])
            for name,n in [('body',29),('left',7),('right',7)]}

    def test_explicit_model_acceptance_preserves_estimated_provenance(self):
        self.assertIs(calibration(self.config, 'camera'), self.config)
        self.assertFalse(self.config['independent_validation'])
        calibration_start_guard(self.config, self.snapshot)

    def test_refuses_missing_acceptance_false_precision_and_bad_scene(self):
        for key, value in [('operator_accepted',False),('factory_mount_position_confirmed',False),
                           ('stationary_base_confirmed',False),('independent_validation',True),
                           ('position_validation_rmse_m',.001),('table_plane_rmse_m',.004),
                           ('tilt_correction_deg',13),('accepted_at_unix_s',time.time()-1801),
                           ('accepted_at_unix_s',float('nan')),('reference_base_q',[0]*14),
                           ('table_half_size_xy_m',[.25,.32])]:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(ValueError):
                    calibration(config, 'camera')

    def test_posture_change_requires_new_calibration(self):
        self.snapshot['body']['q'][12] = .05
        with self.assertRaisesRegex(ValueError, 'posture changed'):
            calibration_start_guard(self.config, self.snapshot)
        self.snapshot['body']['q'][12] = 0
        self.snapshot['body']['imu_rpy'][1] = .03
        with self.assertRaisesRegex(ValueError, 'attitude changed'):
            calibration_start_guard(self.config, self.snapshot)

    def test_imu_heading_wrap_is_not_a_pose_change(self):
        self.config['reference_imu_rpy'][2] = np.pi-.001
        self.snapshot['body']['imu_rpy'][2] = -np.pi+.001
        calibration_start_guard(self.config, self.snapshot)

    def test_extended_table_collides_beyond_old_table_bounds(self):
        import mujoco
        import xml.etree.ElementTree as ET
        def probe(root):
            body = ET.SubElement(root.find('worldbody'), 'body', name='probe', pos='2 1 .77')
            ET.SubElement(body, 'freejoint')
            ET.SubElement(body, 'geom', name='probe_geom', type='sphere', size='.03', mass='.1')
        config = copy.deepcopy(self.config)
        config['pelvis_to_world'][2][3] = .793
        plant = calibrated_scene(config, decorate=probe)
        mujoco.mj_forward(plant.model, plant.data)
        ids = {plant.model.geom(name).id for name in ('table','probe_geom')}
        self.assertTrue(any({c.geom1,c.geom2} == ids and c.dist < 0 for c in plant.data.contact))


if __name__ == '__main__':
    unittest.main()
