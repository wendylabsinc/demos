"""Run through native container_exec after deployment; no control endpoints."""
import json
from pathlib import Path
import time
import urllib.request

from camera_input import Camera, geometry
from target_tracker import TargetTracker
from yolo_client import YoloClient


def status():
    return json.load(urllib.request.urlopen('http://127.0.0.1:8102/status', timeout=3))


before = status()
assert before['phase'] == 'PASSIVE' and not before['adapter_initialized'] and not before['demo_running'], before
assert before['target_identity'] == 'color_depth' and before['yolo_required'] is False, before
assert before['motion_commands_sent'] == 0, before


def forbidden(*args, **kwargs):
    raise AssertionError('YOLO must not be called by color/depth perception')


YoloClient.detect = forbidden
camera = Camera()
tracker = TargetTracker()
config = json.loads(Path('/data/calibration.json').read_text())
rows = []
try:
    camera.acquire()
    camera.frame()  # Warm ray cache before measuring the exact-frame path.
    for _ in range(5):
        start = time.monotonic()
        frame = camera.frame()
        acquired = time.monotonic()
        observation, vision = geometry(frame, config, tracker=tracker)
        tracker.accept(observation, vision, frame)
        done = time.monotonic()
        rows.append(dict(frame_id=frame['meta']['frame_id'], detector=vision['detector'],
                         tracking=vision['tracking']['mode'], camera_ms=(acquired-start)*1000,
                         geometry_ms=(done-acquired)*1000, exposure_age_ms=(done-frame['timestamp'])*1000,
                         center_xy_m=vision['center_xy_m'], cylinder_error_m=vision['fit_error_m']))
finally:
    camera.close()
after = status()
assert after['phase'] == 'PASSIVE' and not after['adapter_initialized'] and not after['demo_running'], after
assert after['motion_commands_sent'] == 0, after
print(json.dumps(dict(passed=True, yolo_calls=0, physical_motion=False,
                      stored_calibration_reference_only=True, movement_authorized=False, frames=rows)))
