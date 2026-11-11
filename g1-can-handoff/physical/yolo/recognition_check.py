"""Exact saved-frame recognition regression; never accesses a robot."""
from confidence import MIN_CONFIDENCE
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection
    return float(intersection / union)


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--box', type=float, nargs=4, required=True)
    parser.add_argument('--model', type=Path, default=root/'model/coke-can.pt')
    parser.add_argument('--size', type=int, default=640)
    args = parser.parse_args()
    result = YOLO(args.model).predict(str(args.image), device='cpu', imgsz=args.size,
                                    rect=False, conf=.001, verbose=False)[0]
    boxes = [dict(confidence=float(b.conf[0]), xyxy=b.xyxy[0].tolist(),
                  iou=iou(b.xyxy[0].numpy(), args.box)) for b in result.boxes]
    matches = [b for b in boxes if b['confidence'] > MIN_CONFIDENCE and b['iou'] >= .5]
    passed = len(matches) == 1 and sum(b['confidence'] > MIN_CONFIDENCE for b in boxes) == 1
    print(json.dumps(dict(passed=passed, input_size=args.size, detections=boxes[:5],
                         physical_motion=False), indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
