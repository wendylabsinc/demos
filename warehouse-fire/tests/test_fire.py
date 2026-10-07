import json
import sys
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from twin import ThermalTwin,HEIGHT,WIDTH
from controller import FireController
from hardware import HardwareAdapter

class ThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(); cls.twin=ThermalTwin(Path(cls.temp.name)/'warehouse.xml')
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()
    def controller(self):
        events=[]; valves=[]
        return FireController(self.twin.hits,self.twin.cameras,lambda kind,**v:events.append((kind,v)),lambda zone,on:valves.append((zone,on))),events,valves
    def test_ambient_never_actuates(self):
        controller,events,valves=self.controller()
        for tick in range(20): controller.update(self.twin.thermal_frames([]),tick*.1)
        self.assertEqual(valves,[]); self.assertEqual(controller.incidents,[])
    def test_camera_confirmation_cannot_accumulate_across_cameras(self):
        controller,events,valves=self.controller()
        frames=self.twin.thermal_frames([dict(position=[7.5,7.5,0],temperature=200)])
        controller.update(frames,0); controller.update(frames,.1)
        self.assertEqual(valves,[])
        controller.update(frames,.2); self.assertEqual(len(controller.incidents),1);self.assertTrue(valves[0][1])
    def test_disconnected_blobs_cannot_count_as_multiple_scans(self):
        hits=np.zeros((1,1,5,3));hits[0,0,:,0]=np.arange(5)*.1
        valves=[]
        controller=FireController(hits,[dict(id='T01')],lambda *a,**k:None,lambda *v:valves.append(v))
        frame=np.array([[[100,100,22,100,100]]])
        controller.update(frame,0);controller.update(frame,.1)
        self.assertEqual(valves,[])
        controller.update(frame,.2);self.assertEqual(len(valves),1)
    def test_occluded_pixel_maps_to_rack_not_floor(self):
        points=self.twin.hits[:64].reshape(-1,3)
        top=points[(abs(points[:,0]-10)<.8)&(abs(points[:,1]-16)<2)]
        self.assertGreater(len(top),20)
        self.assertTrue(np.all(top[:,2]>=2.49))
    def test_local_sprinkler_and_cool_hold(self):
        controller,events,valves=self.controller()
        frame=self.twin.thermal_frames([dict(position=[37.5,32.5,0],temperature=190)])
        for tick in range(3):controller.update(frame,tick*.1)
        self.assertEqual(valves,[(55,True)])
        cool=self.twin.thermal_frames([])
        controller.update(cool,5); self.assertEqual(len(valves),1)
        controller.update(cool,5.3); self.assertEqual(valves[-1],(55,False));self.assertEqual(controller.incidents[0]['state'],'resolved')
    def test_reset_keeps_surface_raycast_valid(self):
        import mujoco
        mujoco.mj_resetData(self.twin.model,self.twin.data)
        mujoco.mj_forward(self.twin.model,self.twin.data)
        self.assertAlmostEqual(self.twin.surface(7.5,7.5)[2],0)
        self.assertAlmostEqual(self.twin.surface(30.4,23.5)[2],2.5)
    def test_valve_fault_is_visible(self):
        events=[]
        def fail(*args):raise OSError('Valve offline')
        controller=FireController(self.twin.hits,self.twin.cameras,lambda kind,**v:events.append(kind),fail)
        frames=self.twin.thermal_frames([dict(position=[7.5,7.5,0],temperature=200)])
        for tick in range(3):controller.update(frames,tick*.1)
        self.assertIn('actuator_fault',events);self.assertEqual(controller.incidents[0]['state'],'actuator_fault')

class HardwareTests(unittest.TestCase):
    def setUp(self):self.adapter=HardwareAdapter(dict(thermal_url='http://sensor/',valve_url='http://valve/'))
    def response(self,value):
        import io
        return io.BytesIO(json.dumps(value).encode())
    def test_stale_capture_is_rejected(self):
        with patch('urllib.request.urlopen',return_value=self.response(dict(timestamp=time.time()-5,sequence=1,celsius=[[[100,100]]]))):
            with self.assertRaisesRegex(ValueError,'Stale'):self.adapter.frames((1,1,2))
    def test_repeated_sequence_is_rejected(self):
        value=dict(timestamp=time.time(),sequence=1,celsius=[[[100,100]]])
        with patch('urllib.request.urlopen',return_value=self.response(value)):self.adapter.frames((1,1,2))
        with patch('urllib.request.urlopen',return_value=self.response(value)):
            with self.assertRaisesRegex(ValueError,'Repeated'):self.adapter.frames((1,1,2))
    def test_bad_shape_is_rejected(self):
        with patch('urllib.request.urlopen',return_value=self.response(dict(timestamp=time.time(),sequence=1,celsius=[100]))):
            with self.assertRaisesRegex(ValueError,'dimensions'):self.adapter.frames((1,1,2))
    def test_wrong_valve_ack_is_rejected(self):
        with patch('urllib.request.urlopen',return_value=self.response(dict(ok=True,zone=4,on=False))):
            with self.assertRaisesRegex(ValueError,'acknowledge'):self.adapter.actuate(4,True)
    def test_exact_valve_ack_succeeds(self):
        with patch('urllib.request.urlopen',return_value=self.response(dict(ok=True,zone=4,on=True))):self.adapter.actuate(4,True)

if __name__=='__main__':unittest.main()
