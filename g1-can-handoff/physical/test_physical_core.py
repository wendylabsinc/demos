import unittest
import numpy as np
from physical_core import rigid,calibration,sim_hand_to_bridge,snapshot_guard

class PhysicalGuards(unittest.TestCase):
    def test_finger_mapping_is_not_hidden_by_symmetric_grip(self):
        np.testing.assert_array_equal(sim_hand_to_bridge([0,1,2,3,4,5,6]),[0,1,2,5,6,3,4])
    def test_nonrigid_and_reflected_camera_frames_rejected(self):
        reflected=np.eye(4);reflected[0,0]=-1
        for bad in (reflected,np.ones((4,4)),np.eye(3)):
            with self.assertRaises(ValueError):rigid(bad)
    def test_missing_or_stale_feedback_fails_before_motion(self):
        snapshot={name:{'time':10,'q':[0]*n,'imu_rpy':[0,0,0],'dq':[0]*n,'tau_est':[0]*n,'faults':[0],'temperature_max':35} for name,n in [('body',29),('left',7),('right',7)]}
        snapshot_guard(snapshot,10.1)
        snapshot['left']['time']=9
        with self.assertRaises(ValueError):snapshot_guard(snapshot,10.1)
    def test_motor_fault_and_heat_fail_before_motion(self):
        snapshot={name:{'time':10,'q':[0]*n,'imu_rpy':[0,0,0],'dq':[0]*n,'tau_est':[0]*n,'faults':[0],'temperature_max':35} for name,n in [('body',29),('left',7),('right',7)]}
        snapshot['body']['faults']=[8]
        with self.assertRaises(ValueError):snapshot_guard(snapshot,10.1)
        snapshot['body']['faults']=[0];snapshot['right']['temperature_max']=75
        with self.assertRaises(ValueError):snapshot_guard(snapshot,10.1)
    def test_calibration_uses_held_out_points_and_rejects_bad_measurement(self):
        from fit_camera_calibration import fit
        points=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]],float)
        world=points+np.array([.2,-.1,.3])
        result=fit({'camera_points_m':points.tolist(),'world_points_m':world.tolist(),'validation_indices':[4,5]})
        np.testing.assert_allclose(np.array(result['camera_to_world'])[:3,3],[.2,-.1,.3],atol=1e-12)
        self.assertFalse(result['commissioned'])
        world[5]+=.04
        with self.assertRaises(ValueError):fit({'camera_points_m':points.tolist(),'world_points_m':world.tolist(),'validation_indices':[4,5]})
    def test_uncommissioned_and_other_camera_calibration_rejected(self):
        with self.assertRaises(ValueError):calibration({'commissioned':False},'338622073335')
        with self.assertRaises(ValueError):calibration({'commissioned':True,'camera_serial':'other'},'338622073335')
    def test_all_motors_accept_below_75_but_reject_at_75(self):
        snapshot={name:{'time':10,'q':[0]*n,'dq':[0]*n,'tau_est':[0]*n,
                        'imu_rpy':[0,0,0],'faults':[0]*n,'temperature_max':74.99}
                  for name,n in [('body',29),('left',7),('right',7)]}
        snapshot_guard(snapshot,10.1)
        for name in snapshot:
            snapshot[name]['temperature_max']=75
            with self.assertRaisesRegex(ValueError,'reaches the 75 C guard'):
                snapshot_guard(snapshot,10.1)
            snapshot[name]['temperature_max']=74.99

    def test_nonfinite_temperature_velocity_and_attitude_rejected(self):
        import copy
        base={name:{'time':10,'q':[0]*n,'dq':[0]*n,'tau_est':[0]*n,
                    'faults':[0]*n,'temperature_max':35,'imu_rpy':[0,0,0]}
              for name,n in [('body',29),('left',7),('right',7)]}
        for field,value in [('temperature_max',float('nan')),('dq',[float('nan')]*29),
                            ('imu_rpy',[0,float('nan'),0])]:
            bad=copy.deepcopy(base);bad['body'][field]=value
            with self.assertRaises(ValueError):snapshot_guard(bad,10.1)

if __name__=='__main__':unittest.main()
