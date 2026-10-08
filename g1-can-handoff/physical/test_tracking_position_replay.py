"""Runtime ownership replay and immutable checked-path tracking reference."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from offline.tracking_position_replay import run
from runtime import Runtime

class TrackingPositionReplayTests(unittest.TestCase):
    def test_native_shift_rejects_old_sequence_but_owned_replan_reaches_checked_sink(self):
        old=run(True);new=run(False)
        self.assertEqual(old['error'],'Tracked can moved outside the checked pickup path')
        self.assertFalse(old['recorded_command_sinks'])
        self.assertTrue(new['checked_pickup_sink_reached'],new['error'])
        self.assertEqual(len(new['baselines']),2)
        self.assertGreater(new['native_shift_m'],.01)
        self.assertLess(new['native_shift_m'],.011)
        self.assertTrue(all(c['samples']>0 for c in new['clearance']))
        self.assertFalse(new['actuation']);self.assertFalse(new['physical_lift_verified'])
        self.assertFalse(new['measured_settling_verified'])

    def test_genuine_displacement_after_owned_preflight_blocks_pickup_sink(self):
        result=run(False,.025)
        self.assertEqual(result['error'],'Tracked can moved outside the checked pickup path')
        self.assertFalse(result['recorded_command_sinks'])
        self.assertFalse(result['checked_pickup_sink_reached'])

    def test_new_samples_never_rebase_and_invalid_position_never_publishes(self):
        runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock()
        runtime.tracking_error=None;runtime.tracking_observation=None
        runtime.tracking_baseline=SimpleNamespace(center=np.array([.3,.15,.823]))
        with patch('runtime.time.monotonic',return_value=100.):
            for x,stamp in ((.304,99.8),(.308,99.85)):
                runtime.publish_tracking_observation(SimpleNamespace(center=np.array([x,.15,.823]),timestamp=stamp),{},True)
            accepted=runtime.tracking_observation
            for center in ([.312,.15,.823],[float('nan'),.15,.823],[.3,.15]):
                with self.assertRaises(RuntimeError):
                    runtime.publish_tracking_observation(SimpleNamespace(center=np.array(center),timestamp=99.9),{},True)
                self.assertIs(runtime.tracking_observation,accepted)
            np.testing.assert_array_equal(runtime.tracking_baseline.center,[.3,.15,.823])

    def test_start_tracking_copies_the_path_reference(self):
        runtime=Runtime.__new__(Runtime);runtime.cancel=threading.Event();runtime.tracking_stop=threading.Event()
        runtime.grasp_started=False;runtime.tracking_guard=Mock();runtime.observe_frame=Mock(return_value=(None,{}))
        baseline=SimpleNamespace(center=np.array([.3,.15,.823]))
        with patch('runtime.threading.Thread'):
            runtime.start_tracking({},baseline)
        baseline.center[0]=.4
        np.testing.assert_array_equal(runtime.tracking_baseline.center,[.3,.15,.823])
        with self.assertRaises(ValueError):runtime.tracking_baseline.center[0]=.4

if __name__=='__main__':unittest.main()
