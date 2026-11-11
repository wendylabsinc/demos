import io
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from sensorlink_camera import FRAME_MAX_AGE, MAX_JPEG_BYTES, JPEGFrames, SensorLinkCamera


class JPEGFramesTests(unittest.TestCase):
    def test_fragmented_and_coalesced_frames_preserve_exact_bytes(self):
        parser = JPEGFrames()
        first, second = b'\xff\xd8first\xff\xd9', b'\xff\xd8second\xff\xd9'
        self.assertEqual(parser.feed(b'noise\xff'), [])
        self.assertEqual(parser.feed(first[1:5]), [])
        self.assertEqual(parser.feed(first[5:] + second), [first, second])
        self.assertEqual(parser.pending, b'')

    def test_unfinished_frame_cannot_grow_without_bound(self):
        parser = JPEGFrames()
        with self.assertRaisesRegex(ValueError, '2 MiB'):
            parser.feed(b'\xff\xd8' + b'x' * MAX_JPEG_BYTES)
        self.assertEqual(parser.pending, b'')
        self.assertEqual(parser.feed(b'\xff\xd8new\xff\xd9'), [b'\xff\xd8new\xff\xd9'])


class SensorLinkCameraTests(unittest.TestCase):
    def test_listing_uses_manifest_and_excludes_non_mjpeg_without_pairing(self):
        camera = SensorLinkCamera('wendy-lite:esp.local')
        manifest = [{'channel_id': 0, 'name': 'camera', 'video': {'codec': 'MJPEG', 'width': 640, 'height': 480}},
                    {'channel_id': 7, 'name': 'other', 'video': {'codec': 'H264'}}]
        with patch('sensorlink_camera.subprocess.run', return_value=Mock(returncode=0, stdout=json.dumps(manifest).encode())) as run:
            result = camera.sources()
        self.assertEqual([source['id'] for source in result['sources']], ['0'])
        self.assertEqual(result['unsupported_channels'], 1)
        self.assertEqual(run.call_args.args[0], ['wendy', '--json', 'device', 'camera', 'list', '--device', 'wendy-lite:esp.local'])
        with self.assertRaises(ValueError):
            camera.start(7)
        with self.assertRaises(ValueError):
            camera.start(False)

    def test_no_configuration_performs_no_discovery(self):
        with patch('sensorlink_camera.subprocess.run') as run:
            self.assertEqual(SensorLinkCamera().sources(), {'sources': [], 'configured': False})
            run.assert_not_called()

    def test_camera_frame_rejects_stale_and_never_replays_same_sequence(self):
        camera = SensorLinkCamera('esp')
        camera.process = Mock()
        camera.latest = b'actual-frame'
        camera.sequence = 23
        camera.received_at = time.monotonic()
        self.assertEqual(camera.frame(22), (b'actual-frame', 23))
        self.assertEqual(camera.frame(23), (None, 23))
        camera.received_at -= FRAME_MAX_AGE + 1
        with self.assertRaisesRegex(ConnectionError, 'fresh'):
            camera.frame(0)

    def test_busy_error_is_camera_specific_and_preserved(self):
        camera = SensorLinkCamera('esp')
        process = Mock(stderr=io.BytesIO(b'subscribing to SensorLink camera: device returned error: device busy\n'),
                       stdout=io.BytesIO(b''))
        camera.process = process
        camera._errors(process)
        with self.assertRaisesRegex(ConnectionError, 'camera is busy'):
            camera.frame()

    def test_subscription_keeps_latest_frame_and_terminates_when_closed(self):
        camera = SensorLinkCamera('wendy-lite:esp.local')
        camera.channels = {0: {'id': '0', 'name': 'camera'}}
        # A real local subprocess exercises pipe draining and process lifetime.
        # These test bytes never enter the application's hardware input path.
        command = [sys.executable, '-u', '-c',
            "import sys,time;sys.stdout.buffer.write(b'\\xff\\xd8one\\xff\\xd9\\xff\\xd8two\\xff\\xd9');sys.stdout.flush();time.sleep(30)"]
        spawn = subprocess.Popen
        with patch('sensorlink_camera.subprocess.Popen', side_effect=lambda *_args, **kwargs: spawn(command, **kwargs)) as run:
            camera.start(0)
            process = camera.process
            try:
                deadline = time.monotonic() + 3
                while camera.latest is None and time.monotonic() < deadline:
                    time.sleep(.01)
                frame, sequence = camera.frame()
                self.assertEqual(frame, b'\xff\xd8two\xff\xd9')
                self.assertEqual(sequence, 2)
                self.assertEqual(run.call_args.args[0], ['wendy', 'device', 'camera', 'view', '--device', 'wendy-lite:esp.local', '--id', '0', '--stdout', '--non-interactive'])
            finally:
                camera.close()
            self.assertIsNotNone(process.poll())
            self.assertIsNone(camera.latest)
