"""Bounded Wendy Lite MJPEG input for the loopback dashboard.

The Wendy CLI owns discovery, identity checking and SensorLink subscription.
Only explicitly configured devices and their advertised MJPEG channels are used.
"""
from collections import deque
import json
import subprocess
import threading
import time

MAX_JPEG_BYTES = 2 * 1024 * 1024
FRAME_MAX_AGE = 2.0
VIEWER_IDLE_SECONDS = 15.0


class JPEGFrames:
    """Split the CLI's concatenated MJPEG stdout without retaining a backlog."""
    def __init__(self):
        self.pending = bytearray()

    def feed(self, chunk):
        self.pending.extend(chunk)
        frames = []
        while self.pending:
            start = self.pending.find(b'\xff\xd8')
            if start < 0:
                self.pending[:] = self.pending[-1:] if self.pending[-1:] == b'\xff' else b''
                break
            if start:
                del self.pending[:start]
            end = self.pending.find(b'\xff\xd9', 2)
            if end < 0:
                if len(self.pending) > MAX_JPEG_BYTES:
                    self.pending.clear()
                    raise ValueError('SensorLink camera frame exceeds the 2 MiB limit.')
                break
            end += 2
            if end > MAX_JPEG_BYTES:
                self.pending.clear()
                raise ValueError('SensorLink camera frame exceeds the 2 MiB limit.')
            frames.append(bytes(self.pending[:end]))
            del self.pending[:end]
        return frames


class SensorLinkCamera:
    def __init__(self, device=None):
        self.device = device
        self.channels = {}
        self.process = None
        self.control = threading.Lock()
        self.lock = threading.Lock()
        self.latest = None
        self.sequence = 0
        self.received_at = 0
        self.last_poll = 0
        self.channel = None
        self.diagnostic = ''
        self.messages = deque(maxlen=8)

    def sources(self):
        if not self.device:
            return {'sources': [], 'configured': False}
        with self.control:
            result = subprocess.run(
                ['wendy', '--json', 'device', 'camera', 'list', '--device', self.device],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=20, check=False)
            if result.returncode:
                raise ConnectionError('Could not list the configured SensorLink camera. Check the device connection.')
            rows = json.loads(result.stdout)
            if not isinstance(rows, list):
                raise ValueError('The CLI returned an unexpected camera manifest.')
            channels = {}
            unsupported = 0
            for row in rows:
                video = row.get('video') or {}
                if video.get('codec') != 'MJPEG':
                    unsupported += 1
                    continue
                channel = int(row['channel_id'])
                if not 0 <= channel <= 4294967295:
                    continue
                channels[channel] = {'id': str(channel), 'name': row.get('name') or f'Camera {channel}',
                    'device': self.device, 'width': video.get('width'),
                    'height': video.get('height'), 'fps': video.get('fps'), 'kind': 'sensorlink'}
            self.channels = channels
            return {'sources': list(channels.values()), 'configured': True,
                    'unsupported_channels': unsupported}

    def start(self, channel):
        with self.control:
            if type(channel) is not int or channel not in self.channels:
                raise ValueError('Select an advertised MJPEG SensorLink camera first.')
            self._stop()
            process = subprocess.Popen(
                ['wendy', 'device', 'camera', 'view', '--device', self.device,
                 '--id', str(channel), '--stdout', '--non-interactive'],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self.lock:
                self.process = process
                self.channel = channel
                self.latest = None
                self.diagnostic = ''
                self.messages.clear()
                self.last_poll = time.monotonic()
            threading.Thread(target=self._read, args=(process,), daemon=True).start()
            threading.Thread(target=self._errors, args=(process,), daemon=True).start()
            threading.Thread(target=self._watch, args=(process,), daemon=True).start()
            return self.channels[channel]

    def _errors(self, process):
        with process.stderr:
            for line in iter(lambda: process.stderr.readline(2048), b''):
                with self.lock:
                    if self.process is not process:
                        return
                    message = line.decode(errors='replace').strip()
                    self.messages.append(message)
                    if 'device busy' in message.lower():
                        self.diagnostic = 'SensorLink camera is busy. Stop its other viewer or use its existing gateway before reconnecting.'

    def _read(self, process):
        parser = JPEGFrames()
        failure = 'SensorLink camera disconnected. Select the camera again to reconnect.'
        try:
            with process.stdout:
                while chunk := process.stdout.read1(65536):
                    frames = parser.feed(chunk)
                    if frames:
                        with self.lock:
                            if self.process is not process:
                                return
                            # Never queue captured images behind inference or a slow viewer.
                            self.latest = frames[-1]
                            self.sequence += len(frames)
                            self.received_at = time.monotonic()
        except (OSError, ValueError) as error:
            failure = str(error)
        finally:
            with self.lock:
                if self.process is process:
                    self.latest = None
                    self.diagnostic = self.diagnostic or failure
            if process.poll() is None:
                process.terminate()

    def _watch(self, process):
        while process.poll() is None:
            time.sleep(1)
            with self.control:
                with self.lock:
                    idle = self.process is process and time.monotonic() - self.last_poll > VIEWER_IDLE_SECONDS
                if idle:
                    self._stop()
                    return

    def frame(self, after=0):
        with self.lock:
            self.last_poll = time.monotonic()
            if self.process is None:
                raise ConnectionError('SensorLink camera is off.')
            if self.diagnostic:
                raise ConnectionError(self.diagnostic)
            if self.latest is None or time.monotonic() - self.received_at > FRAME_MAX_AGE:
                raise ConnectionError('Waiting for a fresh SensorLink camera frame.')
            if self.sequence <= after:
                return None, self.sequence
            return self.latest, self.sequence

    def _stop(self):
        with self.lock:
            process, self.process = self.process, None
            self.latest = None
            self.channel = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def close(self):
        with self.control:
            self._stop()
