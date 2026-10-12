"""Shared scene frame for camera and measured FK; loss blocks arm publication."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from physical_core import PhysicalPlant, calibrated_scene
from pose_alignment import corrected_config


class SceneReferenceFKTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict('os.environ', {'HANDOFF_CAMERA_MAX_AGE_MS':'500'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        pelvis = np.eye(4)
        pelvis[2, 3] = .8
        camera = np.eye(4)
        camera[:3, 3] = [.07, .01, 1.25]
        self.config = dict(
            calibration_method='operator_accepted_factory_table',
            reference_imu_rpy=[0, 0, 0],
            reference_base_q=[0] * 15,
            pelvis_to_world=pelvis.tolist(),
            camera_to_world=camera.tolist(),
            table_height_m=.762,
            table_center_xy_m=[.5, 0],
            table_half_size_xy_m=[.4, 1.],
        )
        self.snapshot = dict(
            body=dict(q=[0] * 29, imu_rpy=[.002, .015, .01], time=99.99),
            left=dict(q=[0] * 7), right=dict(q=[0] * 7),
        )
        self.subject = PhysicalPlant.__new__(PhysicalPlant)
        self.subject.config = copy.deepcopy(self.config)
        self.subject.ik = calibrated_scene(copy.deepcopy(self.config))
        self.subject.clock = SimpleNamespace(monotonic=lambda: 100.)
        self.subject.adapter = Mock()
        self.transform = np.eye(4)
        self.transform[:3, :3] = Rotation.from_euler('xyz', [.001, -.002, .018]).as_matrix()
        self.transform[:3, 3] = [.003, -.002, .001]

    def test_shared_transform_preserves_camera_relative_fk_and_config(self):
        from scene_reference import apply_scene_transform

        self.subject.sync(self.snapshot)
        before = self.subject.ik.data.site_xpos[self.subject.ik.site].copy()
        orientation_before = self.subject.ik.data.site_xmat[self.subject.ik.site].reshape(3, 3).copy()
        table_before = self.subject.ik.data.geom_xpos[self.subject.ik.model.geom('table').id].copy()
        corrected = corrected_config(self.config, dict(
            timestamp=self.snapshot['body']['time'], rpy=self.snapshot['body']['imu_rpy'], waist_q=[0] * 3))
        camera_before = np.asarray(corrected['camera_to_world'])
        self.subject.scene_pose_provider = lambda: (self.transform, 99.9)
        self.subject.sync(self.snapshot)
        after = self.subject.ik.data.site_xpos[self.subject.ik.site].copy()
        np.testing.assert_allclose(after, (self.transform @ np.r_[before, 1])[:3], atol=1e-12)
        np.testing.assert_allclose(self.subject.ik.data.site_xmat[self.subject.ik.site].reshape(3, 3),
                                   self.transform[:3, :3] @ orientation_before, atol=1e-12)
        corrected_scene = apply_scene_transform(corrected, self.transform)
        camera_after = np.asarray(corrected_scene['camera_to_world'])
        np.testing.assert_allclose(np.linalg.inv(camera_before) @ np.r_[before, 1],
                                   np.linalg.inv(camera_after) @ np.r_[after, 1], atol=1e-12)
        np.testing.assert_array_equal(self.subject.ik.data.geom_xpos[self.subject.ik.model.geom('table').id], table_before)
        self.assertEqual(self.subject.config, self.config)
        # Repeated measured updates apply the latest transform once, never accumulate it.
        self.subject.sync(self.snapshot)
        np.testing.assert_array_equal(self.subject.ik.data.site_xpos[self.subject.ik.site], after)

    def test_unconfigured_provider_preserves_existing_imu_fk(self):
        self.subject.sync(self.snapshot)
        before = self.subject.ik.data.site_xpos[self.subject.ik.site].copy()
        self.subject.scene_pose_provider = None
        self.subject.sync(self.snapshot)
        np.testing.assert_array_equal(self.subject.ik.data.site_xpos[self.subject.ik.site], before)

    def test_invalid_stale_future_and_missing_correction_block_arm_publication(self):
        bad_rotation = self.transform.copy()
        bad_rotation[:3, :3] = Rotation.from_euler('z', .0401).as_matrix()
        bad_translation = self.transform.copy()
        bad_translation[:3, 3] = [.0201, 0, 0]
        nonfinite = self.transform.copy()
        nonfinite[0, 3] = float('nan')
        nonrigid = self.transform.copy()
        nonrigid[0, 0] *= 1.5
        self.subject.sync(self.snapshot)
        before_position = self.subject.ik.model.body_pos.copy()
        before_rotation = self.subject.ik.model.body_quat.copy()
        for value in [
            None, (bad_rotation, 99.9), (bad_translation, 99.9),
            (nonfinite, 99.9), (nonrigid, 99.9),
            (self.transform, 99.499), (self.transform, 100.001),
            (self.transform, float('nan')), (self.transform, float('inf')),
            (self.transform, True),
        ]:
            with self.subTest(value=value):
                self.subject.scene_pose_provider = lambda: value
                with self.assertRaises((ValueError, RuntimeError)):
                    self.subject.command_arm(np.zeros(7), self.snapshot)
                self.subject.adapter.ctrl_dual_arm.assert_not_called()
                np.testing.assert_array_equal(self.subject.ik.model.body_pos, before_position)
                np.testing.assert_array_equal(self.subject.ik.model.body_quat, before_rotation)

    def test_provider_exception_blocks_arm_publication(self):
        def lost():
            raise RuntimeError('Registration lost static support')
        self.subject.scene_pose_provider = lost
        with self.assertRaisesRegex(RuntimeError, 'lost static support'):
            self.subject.command_arm(np.zeros(7), self.snapshot)
        self.subject.adapter.ctrl_dual_arm.assert_not_called()

    def test_configured_scene_provider_rejects_unsupported_calibration(self):
        self.subject.config['calibration_method'] = 'test_static'
        self.subject.scene_pose_provider = lambda: (self.transform, 99.9)
        with self.assertRaisesRegex(RuntimeError, 'accepted factory/table'):
            self.subject.command_arm(np.zeros(7), self.snapshot)
        self.subject.adapter.ctrl_dual_arm.assert_not_called()
