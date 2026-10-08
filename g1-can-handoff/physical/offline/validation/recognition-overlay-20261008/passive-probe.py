"""Run through native container_exec after deployment; no control endpoints."""
import json
from pathlib import Path
import time
import threading
import urllib.request

from camera_input import Camera, geometry
from target_tracker import TargetTracker
from yolo_client import YoloClient
from scene_refresh import refresh_scene
from pose_alignment import PoseHistory, corrected_config


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
poses = PoseHistory()
stop = threading.Event()
poll_errors = []


def record_poses():
    while not stop.is_set():
        try:
            body = status()['measured']['body']
            poses.add(body['time'], body['imu_rpy'], body['q'][12:15])
        except ValueError as exc:
            if 'must advance' not in str(exc):
                poll_errors.append(str(exc))
        except Exception as exc:
            poll_errors.append(str(exc))
        stop.wait(.01)


sampler = threading.Thread(target=record_poses, daemon=True)
sampler.start()
try:
    camera.acquire()
    camera.frame()  # Warm ray cache before measuring the exact-frame path.
    time.sleep(.5)  # Collect exposure-aligned poses without opening a motor adapter.
    scene_frame = camera.frame()
    config = refresh_scene(config, scene_frame, status()['measured'],
                           scene_frame['meta']['camera_serial'],
                           exposure_pose=poses.nearest(scene_frame['timestamp']))
    for _ in range(5):
        start = time.monotonic()
        frame = camera.frame()
        acquired = time.monotonic()
        aligned = corrected_config(config, poses.nearest(frame['timestamp']))
        observation, vision = geometry(frame, aligned, tracker=tracker)
        tracker.accept(observation, vision, frame)
        done = time.monotonic()
        rows.append(dict(frame_id=frame['meta']['frame_id'], detector=vision['detector'],
                         tracking=vision['tracking']['mode'], camera_ms=(acquired-start)*1000,
                         geometry_ms=(done-acquired)*1000, exposure_age_ms=(done-frame['timestamp'])*1000,
                         center_xy_m=vision['center_xy_m'], cylinder_error_m=vision['fit_error_m']))
finally:
    camera.close()
    stop.set()
    sampler.join(timeout=4)
after = status()
assert after['phase'] == 'PASSIVE' and not after['adapter_initialized'] and not after['demo_running'], after
assert after['motion_commands_sent'] == 0, after
print(json.dumps(dict(passed=True, yolo_calls=0, physical_motion=False,
                      scene_refreshed_in_memory=True, exposure_pose_synchronized=True,
                      calibration_persisted=False, pose_poll_errors=poll_errors,
                      movement_authorized=False, frames=rows)))
