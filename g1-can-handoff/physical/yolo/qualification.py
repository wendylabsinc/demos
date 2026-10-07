"""Saved-frame recognition gates shared by offline replay and CUDA readiness."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from postprocessing import box_iou


def fixtures(root):
    root = Path(root)
    manifest = json.loads((root/'qualification.json').read_text())
    if not manifest['fixtures']:
        raise RuntimeError('No Coke qualification fixtures')
    for row in manifest['fixtures']:
        path = root/row['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Qualification image checksum mismatch: '+row['image'])
        image = np.asarray(Image.open(path).convert('RGB'))[:, :, ::-1].copy()
        yield row, image


def check(row, boxes, saturated=False):
    good = [b for b in boxes if b['confidence'] > .30]
    expected = row['bbox_xyxy']
    if saturated or len(good) != (0 if expected is None else 1):
        raise RuntimeError('Coke recognition qualification failed: '+row['image'])
    if expected is not None:
        keys = ('x1', 'y1', 'x2', 'y2')
        overlap = box_iou(dict(zip(keys, good[0]['xyxy'])), dict(zip(keys, expected)))
        if overlap < .5:
            raise RuntimeError('Coke box qualification failed: '+row['image'])
    else:
        overlap = None
    return dict(image=row['image'], role=row['role'], detections=good, iou=overlap)
