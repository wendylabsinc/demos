import copy
import time
import unittest
from unittest.mock import patch
import test_model_calibration
from scene_refresh import refresh_scene


class SceneRefreshTests(unittest.TestCase):
    def setUp(self):
        fixture=test_model_calibration.ModelCalibrationTests();fixture.setUp()
        self.config=fixture.config;self.snapshot=fixture.snapshot
        self.frame={'meta':{'camera_serial':'camera'}}
        self.fresh=copy.deepcopy(self.config)
        self.fresh.update(commissioned=False,calibration_method='factory_mount_and_table_plane_candidate',source_frame_id=101)

    def test_expired_scene_replaced_by_new_observation_and_current_pose(self):
        self.config['accepted_at_unix_s']=time.time()-3600
        self.snapshot['body']['q'][12]=.1
        before=copy.deepcopy(self.config)
        with patch('scene_refresh.candidate',return_value=self.fresh) as observe:
            result=refresh_scene(self.config,self.frame,self.snapshot,'camera')
        observe.assert_called_once_with(self.frame,self.snapshot)
        self.assertEqual(result['source_frame_id'],101)
        self.assertEqual(result['reference_base_q'][12],.1)
        self.assertFalse(result['independent_validation'])
        self.assertTrue(result['scene_refresh_automatic'])
        self.assertEqual(self.config,before)

    def test_refresh_does_not_create_operator_authorization(self):
        for field in ('commissioned','operator_accepted','factory_mount_position_confirmed','stationary_base_confirmed'):
            config=dict(self.config);config[field]=False
            with patch('scene_refresh.candidate') as observe:
                with self.assertRaises(ValueError):refresh_scene(config,self.frame,self.snapshot,'camera')
                observe.assert_not_called()

    def test_bad_new_geometry_and_wrong_camera_are_rejected(self):
        self.fresh['table_plane_rmse_m']=.01
        with patch('scene_refresh.candidate',return_value=self.fresh):
            with self.assertRaisesRegex(ValueError,'table_plane'):refresh_scene(self.config,self.frame,self.snapshot,'camera')
        self.frame['meta']['camera_serial']='other'
        with patch('scene_refresh.candidate') as observe:
            with self.assertRaisesRegex(ValueError,'identity'):refresh_scene(self.config,self.frame,self.snapshot,'camera')
            observe.assert_not_called()


if __name__=='__main__':unittest.main()
