"""Replay the parked-prefetch failure through the real one-slot buffer."""
import threading
import time
import unittest
from latest_frame import LatestFrameBuffer


class LatestFrameTests(unittest.TestCase):
    def test_readiness_waits_for_a_fresh_decoded_frame_without_consuming_it(self):
        class Camera:
            def __init__(self):
                self.count=0;self.next_frame=threading.Event();self.finished=threading.Event()
            def frame(self,**kwargs):
                self.count+=1
                if self.count==1:return {'timestamp':time.monotonic()-.5,'meta':{'frame_id':1}}
                if self.count==2:
                    self.next_frame.wait(2)
                    return {'timestamp':time.monotonic()-.05,'meta':{'frame_id':2}}
                self.finished.wait(2);raise RuntimeError('fixture stopped')
        camera=Camera();buffer=LatestFrameBuffer(camera);ready=threading.Event();errors=[]
        def wait():
            try:buffer.wait_ready(max_age_s=.3,timeout=.5);ready.set()
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=wait);thread.start()
        try:
            self.assertFalse(ready.wait(.03));camera.next_frame.set();thread.join(1)
            self.assertEqual(errors,[]);self.assertTrue(ready.is_set())
            self.assertIsNone(buffer.consumed)
            self.assertEqual(buffer.next()['meta']['frame_id'],2)
        finally:
            buffer.stopped.set();camera.next_frame.set();camera.finished.set();thread.join(1);buffer.close()

    def test_readiness_rejects_future_frame_and_stall(self):
        class Camera:
            def __init__(self):self.calls=0;self.stop=threading.Event()
            def frame(self,**kwargs):
                self.calls+=1
                if self.calls==1:return {'timestamp':time.monotonic()+10,'meta':{'frame_id':1}}
                self.stop.wait(2);raise RuntimeError('fixture stopped')
        camera=Camera();buffer=LatestFrameBuffer(camera)
        try:
            with self.assertRaisesRegex(RuntimeError,'readiness timeout'):
                buffer.wait_ready(max_age_s=.3,timeout=.03)
        finally:buffer.stopped.set();camera.stop.set();buffer.close()

    def test_processing_pause_consumes_newest_not_parked_frame(self):
        class Camera:
            def __init__(self):
                self.calls=0; self.gates=[threading.Event() for _ in range(4)]
                self.started=[threading.Event() for _ in range(4)]
            def frame(self, **kwargs):
                index=min(self.calls,3); self.calls+=1; self.started[index].set()
                if not self.gates[index].wait(2): raise RuntimeError('fixture stopped')
                return {'timestamp':100.+index*.067,'meta':{'frame_id':index+1}}
        camera=Camera(); buffer=LatestFrameBuffer(camera)
        try:
            camera.gates[0].set(); self.assertEqual(buffer.next()['meta']['frame_id'],1)
            # While geometry works on frame 1, frames 2 and 3 arrive. A parked
            # future would return 2; the bounded latest slot must return 3.
            camera.gates[1].set(); self.assertTrue(camera.started[2].wait(1))
            camera.gates[2].set(); self.assertTrue(camera.started[3].wait(1))
            self.assertEqual(buffer.next()['meta']['frame_id'],3)
            self.assertEqual(buffer.latest['timestamp'],100.134)
            with self.assertRaisesRegex(RuntimeError,'timeout'):buffer.next(timeout=.01)
        finally:
            buffer.stopped.set(); camera.gates[3].set(); buffer.close()

    def test_reader_failure_is_latched_even_with_a_cached_frame(self):
        class Camera:
            def frame(self,**kwargs): raise RuntimeError('calibration changed')
        buffer=LatestFrameBuffer(Camera())
        try:
            with self.assertRaisesRegex(RuntimeError,'calibration changed'):buffer.next()
        finally:buffer.close()


if __name__=='__main__': unittest.main()
