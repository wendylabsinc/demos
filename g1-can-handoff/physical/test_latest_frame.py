"""Replay the parked-prefetch failure through the real one-slot buffer."""
import threading
import unittest
from latest_frame import LatestFrameBuffer


class LatestFrameTests(unittest.TestCase):
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
