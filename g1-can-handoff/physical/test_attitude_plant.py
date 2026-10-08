"""Measured robot attitude changes FK/gravity while the world table stays fixed."""
import copy
import unittest
from unittest.mock import Mock,patch
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from physical_core import PhysicalPlant,calibrated_scene,BODY_NAMES,SIM_TO_BRIDGE


class AttitudePlantTests(unittest.TestCase):
    def test_fk_and_gravity_match_a_scene_built_at_current_attitude(self):
        base=np.eye(4);base[2,3]=.8
        camera=np.eye(4);camera[:3,3]=[.07,0,1.25]
        config={'calibration_method':'operator_accepted_factory_table','reference_imu_rpy':[0,0,0],
                'reference_base_q':[0]*15,'pelvis_to_world':base.tolist(),'camera_to_world':camera.tolist(),
                'table_height_m':.762,'table_center_xy_m':[.5,0],'table_half_size_xy_m':[.4,1.]}
        subject=PhysicalPlant.__new__(PhysicalPlant);subject.config=config;subject.ik=calibrated_scene(config)
        snapshot={'body':{'q':[0]*29,'imu_rpy':[0,0,0],'time':1},'left':{'q':[0]*7},'right':{'q':[0]*7}}
        subject.sync(snapshot);before=subject.ik.data.site_xpos[subject.ik.site].copy()
        table_before=subject.ik.data.geom_xpos[subject.ik.model.geom('table').id].copy()
        snapshot['body']['imu_rpy']=[0,.02,0];subject.sync(snapshot)
        rotation=Rotation.from_euler('y',.02).as_matrix()
        expected=base[:3,3]+rotation@(before-base[:3,3])
        np.testing.assert_allclose(subject.ik.data.site_xpos[subject.ik.site],expected,atol=1e-10)
        np.testing.assert_array_equal(subject.ik.data.geom_xpos[subject.ik.model.geom('table').id],table_before)
        other=PhysicalPlant.__new__(PhysicalPlant);other.config=copy.deepcopy(config)
        other.config['calibration_method']='test_static';base[:3,:3]=rotation
        other.config['pelvis_to_world']=base.tolist();other.ik=calibrated_scene(other.config);other.sync(snapshot)
        np.testing.assert_allclose(subject.ik.data.qfrc_bias,other.ik.data.qfrc_bias,atol=1e-9)
        subject.gravity_data=mujoco.MjData(subject.ik.model)
        subject.left_hand=np.arange(7)*.01;subject.right_hand=np.arange(7)*.02;subject.right_arm=np.arange(7)*.03
        subject.adapter=Mock()
        def reject_exact_target(data):
            for name,q in zip(BODY_NAMES[22:29],subject.right_arm):
                self.assertAlmostEqual(data.qpos[subject.ik.model.jnt_qposadr[subject.ik.model.joint(name).id]],q)
            for side,values in [('left',subject.left_hand),('right',subject.right_hand)]:
                for suffix,q in zip(('thumb_0','thumb_1','thumb_2','index_0','index_1','middle_0','middle_1'),values[SIM_TO_BRIDGE]):
                    self.assertAlmostEqual(data.qpos[subject.ik.model.jnt_qposadr[subject.ik.model.joint(side+'_hand_'+suffix+'_joint').id]],q)
            return {'bodies':['table','hand']}
        with patch.object(subject.ik,'collision_details',side_effect=reject_exact_target):
            with self.assertRaisesRegex(RuntimeError,'Live target intersects'):
                subject.command_arm(np.zeros(7),snapshot)
        subject.adapter.ctrl_dual_arm.assert_not_called()

if __name__=='__main__':unittest.main()
