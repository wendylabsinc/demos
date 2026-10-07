"""YOLO detections bound to the exact calibrated RGB-D frame being fitted."""
import hashlib
import json
import os
import time
import urllib.request
import urllib.error

import numpy as np
from yolo.frame_limits import frame_age_limit_s

MODEL_SHA256 = '74b62f016b7a5aeb50c52422e3ee6dd76472456bf731935d2263fa000b96a862'
SCHEMA = 'wendy.handoff.coke-yolo.v1'


def identity(frame,raw=None):
    image = np.asarray(frame['color_bgr'])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError('YOLO requires uint8 HWC BGR from the exact RGB-D frame')
    return dict(frame_id=frame['meta']['frame_id'], camera_serial=frame['meta']['camera_serial'],
                timestamp=frame['timestamp'], shape=list(image.shape),
                image_sha256=hashlib.sha256(image.tobytes() if raw is None else raw).hexdigest())


def fresh(frame):
    age=time.monotonic()-frame['timestamp']
    if not 0 <= age <= frame_age_limit_s():
        raise RuntimeError(f'YOLO RGB-D exposure timestamp invalid/stale after inference ({age*1000:.1f} ms)')


def validate_result(frame, result):
    fresh(frame)
    if result.get('schema') != SCHEMA or result.get('source') != identity(frame):
        raise RuntimeError('YOLO response does not match exact RGB-D image/frame identity')
    if result.get('model_sha256') != MODEL_SHA256 or result.get('backend') != 'pytorch_cuda':
        raise RuntimeError('YOLO checkpoint/backend is not the pinned Coke CUDA detector')
    if result.get('truncated') is not False:
        raise RuntimeError('YOLO detection limit reached; target ambiguity is unknown')
    boxes = result.get('detections')
    if not isinstance(boxes, list) or len(boxes) > 20:
        raise RuntimeError('Invalid YOLO detections')
    height, width = frame['color_bgr'].shape[:2]
    for box in boxes:
        xyxy = np.asarray(box['xyxy'], dtype=float)
        confidence = box['confidence']
        if (xyxy.shape != (4,) or not np.isfinite(xyxy).all() or
                not np.isfinite(confidence) or not 0 <= confidence <= 1 or
                box.get('class_id') != 0 or box.get('class_name') != 'Coca-Cola' or
                not 0 <= xyxy[0] < xyxy[2] <= width or
                not 0 <= xyxy[1] < xyxy[3] <= height):
            raise RuntimeError('Invalid Coke class, confidence or image box')
    return result


class YoloClient:
    def __init__(self):
        self.url = os.getenv('HANDOFF_YOLO_URL', 'http://127.0.0.1:8112')

    def detect(self, frame):
        fresh(frame)
        raw = frame['color_bgr'].tobytes()
        source = identity(frame,raw)
        request = urllib.request.Request(self.url + '/detect', data=raw,
                    headers={'Content-Type': 'application/octet-stream',
                             'X-Frame-Identity': json.dumps(source)}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=1) as response:
                result = json.loads(response.read(65537))
        except urllib.error.HTTPError as exc:
            detail=exc.read(1024).decode('utf-8',errors='replace')
            raise RuntimeError('Exact-frame Coke YOLO inference rejected: '+detail) from exc
        except Exception as exc:
            raise RuntimeError('Exact-frame Coke YOLO inference unavailable: ' + str(exc)) from exc
        return validate_result(frame, result)
