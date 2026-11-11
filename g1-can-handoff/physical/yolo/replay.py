"""Saved-frame semantic recognition and optimized decoding parity; no robot IO."""
from confidence import MIN_CONFIDENCE
import hashlib
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import torch
from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates
from qualification import fixtures, check


def main():
    root = Path(__file__).parent
    manifest = json.loads((root/'model/SOURCE.json').read_text())
    weights = root/'model/coke-can.pt'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise RuntimeError('Checkpoint checksum mismatch')
    model = YOLO(weights)
    torch.set_num_threads(2)
    if model.names != {0: 'Coca-Cola'}:
        raise RuntimeError('Expected Coca-Cola checkpoint')
    checks = []
    for row, image in fixtures(root/'fixtures'):
        result = model.predict(image, device='cpu', imgsz=640, rect=False,
                               conf=MIN_CONFIDENCE, iou=.7, verbose=False)[0]
        reference = [dict(xyxy=b.xyxy[0].tolist(), confidence=float(b.conf[0])) for b in result.boxes]
        report = check(row, reference)
        chw, scale, pad_x, pad_y = letterbox_bgr(image, 640)
        with torch.inference_mode():
            raw = model.model(torch.from_numpy(chw).float().div_(255).unsqueeze(0))
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if tuple(raw.shape) != (1, 5, 8400):
            raise RuntimeError('Unexpected raw one-class output')
        values, indices = raw[0, 4].topk(129, largest=True, sorted=True)
        decoded = decode_one_class_candidates(raw[0, :, indices[:128]].T.numpy(), confidence=MIN_CONFIDENCE,
            nms_iou=.7, max_detections=21, frame_width=image.shape[1], frame_height=image.shape[0],
            scale=scale, pad_x=pad_x, pad_y=pad_y)
        optimized = [dict(xyxy=[b[k] for k in ('x1', 'y1', 'x2', 'y2')], confidence=b['conf']) for b in decoded]
        check(row, optimized, bool(values[-1] > MIN_CONFIDENCE))
        ref = [b for b in reference if b['confidence'] > MIN_CONFIDENCE]
        opt = [b for b in optimized if b['confidence'] > MIN_CONFIDENCE]
        for a, b in zip(ref, opt):
            if np.max(np.abs(np.asarray(a['xyxy'])-b['xyxy'])) > 1 or abs(a['confidence']-b['confidence']) > .01:
                raise RuntimeError('Optimized decoding parity failed: '+row['image'])
        checks.append(report)
    print(json.dumps(dict(checkpoint_sha256=digest, offline_fixture_passed=True,
                         backend='cpu_offline_replay', fixtures=checks,
                         optimized_decoder_parity_passed=True, physical_pickup_verified=False), indent=2))


if __name__ == '__main__':
    main()
