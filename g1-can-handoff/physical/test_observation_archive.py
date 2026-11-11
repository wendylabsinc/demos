"""Evidence persistence and final freshness must precede a usable observation."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import Mock,patch
import tempfile

with tempfile.TemporaryDirectory() as directory,patch.dict('os.environ',{'HANDOFF_DATA':directory}):
    import runtime


class ObservationArchiveTests(unittest.TestCase):
    def setUp(self):
        self.subject=runtime.Runtime.__new__(runtime.Runtime)
        self.subject.archive_executor=ThreadPoolExecutor(max_workers=1)
        self.subject.event=Mock()
        self.frame={'timestamp':time.monotonic(),'meta':{'frame_id':1}}
        class Camera:
            def frame(inner,**kwargs):return self.frame
        self.subject.camera=Camera()
    def tearDown(self):
        self.subject.archive_executor.shutdown(wait=True)
    def test_archive_error_prevents_returning_successful_vision(self):
        def archive(*args):raise OSError('Evidence disk failed')
        self.subject.archive_frame=archive
        with patch.object(runtime,'geometry',return_value=('observation','vision')):
            with self.assertRaisesRegex(OSError,'Evidence disk failed'):
                self.subject.observe_frame({},'planning',True)
    def test_archive_overlaps_geometry_and_is_joined_before_freshness(self):
        started=threading.Event();finish=threading.Event();persisted=[]
        def archive(frame,phase):
            started.set()
            if not finish.wait(2):raise RuntimeError('Geometry did not overlap archive')
            persisted.append(frame)
        def geometry(*args,**kwargs):
            self.assertTrue(started.wait(2));finish.set()
            return 'observation','vision'
        def fresh(frame):self.assertEqual(persisted,[frame])
        self.subject.archive_frame=archive
        with patch.object(runtime,'geometry',side_effect=geometry),patch.object(runtime,'fresh',side_effect=fresh):
            self.assertEqual(self.subject.observe_frame({},'planning',True),('observation','vision'))
    def test_exposure_becoming_stale_during_archive_is_rejected(self):
        def archive(frame,phase):frame['timestamp']=time.monotonic()-.401
        self.subject.archive_frame=archive
        with patch.object(runtime,'geometry',return_value=('observation','vision')):
            with self.assertRaisesRegex(RuntimeError,'stale'):
                self.subject.observe_frame({},'planning',True)

    def test_213_ms_acquisition_is_usable_only_if_final_observation_is_fresh(self):
        self.frame['timestamp']=time.monotonic()-.213
        self.subject.archive_frame=lambda *args:None
        with patch.object(runtime,'geometry',return_value=('observation','vision')):
            self.assertEqual(self.subject.observe_frame({},'planning',True),('observation','vision'))

    def test_old_acquisitions_are_bounded_and_never_sent_to_inference(self):
        self.frame['timestamp']=time.monotonic()-.31
        self.subject.archive_frame=lambda *args:self.fail('Old frame was archived as an observation')
        with patch.object(self.subject.camera,'frame',return_value=self.frame) as camera,patch.object(runtime,'geometry') as geometry:
            with self.assertRaisesRegex(RuntimeError,'300 ms acquisition'):
                self.subject.observe_frame({},'planning',True)
            self.assertEqual(camera.call_count,3)
            geometry.assert_not_called()


if __name__=='__main__':unittest.main()
