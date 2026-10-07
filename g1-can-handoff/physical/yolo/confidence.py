"""Operator-required semantic confidence floor shared by both services."""
import math
import os

MIN_CONFIDENCE = .10


def confidence_threshold():
    threshold=float(os.getenv('HANDOFF_YOLO_MIN_CONFIDENCE', str(MIN_CONFIDENCE)))
    if not math.isfinite(threshold) or not MIN_CONFIDENCE<=threshold<=.99:
        raise ValueError('HANDOFF_YOLO_MIN_CONFIDENCE must be in [0.10, 0.99]')
    return threshold
