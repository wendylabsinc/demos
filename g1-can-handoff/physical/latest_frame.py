"""One-slot continuously refreshed RGB-D buffer, without actuator ownership."""
import threading
import time


class LatestFrameBuffer:
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.latest = None
        self.error = None
        self.consumed = None
        self.thread = threading.Thread(target=self._capture, daemon=True, name='latest-rgbd')
        self.thread.start()

    def _capture(self):
        try:
            while not self.stopped.is_set():
                frame = self.camera.frame(materialize=False)
                with self.condition:
                    if self.latest is not None:
                        old = self.latest
                        if frame['timestamp'] <= old['timestamp'] or frame['meta']['frame_id'] <= old['meta']['frame_id']:
                            raise RuntimeError('Latest camera frame did not advance')
                    self.latest = frame
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def next(self, timeout=4.):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError('Latest camera reader failed: ' + str(self.error)) from self.error
                if self.stopped.is_set():
                    raise RuntimeError('Latest camera reader stopped')
                if self.latest is not None and self.latest['meta']['frame_id'] != self.consumed:
                    self.consumed = self.latest['meta']['frame_id']
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Latest camera frame timeout')
                self.condition.wait(remaining)

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(6.)
        if self.thread.is_alive():
            raise RuntimeError('Latest camera reader has not stopped')
