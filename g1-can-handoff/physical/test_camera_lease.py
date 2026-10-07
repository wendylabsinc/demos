import json
import threading
import unittest

from camera_input import Camera


class CameraLeaseTests(unittest.TestCase):
    def test_lease_renews_without_frames_and_stops_before_release(self):
        camera = Camera()
        renewed = threading.Event()
        calls = []
        def request(path, data=None):
            calls.append(path)
            if path == '/lease/acquire':
                return json.dumps({'token':'lease'}).encode()
            if path == '/lease/renew':
                renewed.set()
            return b'{}'
        camera.request = request
        camera.acquire()
        worker = camera.lease_thread
        try:
            self.assertTrue(renewed.wait(2), 'Lease must renew while planning has no camera frames')
        finally:
            camera.close()
        self.assertFalse(worker.is_alive())
        self.assertIsNone(camera.token)
        self.assertEqual(calls[-1], '/lease/release')


if __name__ == '__main__':
    unittest.main()
