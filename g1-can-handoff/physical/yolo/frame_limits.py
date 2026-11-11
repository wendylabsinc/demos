"""Shared camera exposure budget; milliseconds, bounded and fail closed."""
import math
import os

def frame_age_limit_s(max_age_ms=None):
    value=float(os.getenv('HANDOFF_CAMERA_MAX_AGE_MS','400') if max_age_ms is None else max_age_ms)
    if not math.isfinite(value) or not 50 <= value <= 500:
        raise ValueError('HANDOFF_CAMERA_MAX_AGE_MS must be finite and within 50..500 ms')
    return value/1000
