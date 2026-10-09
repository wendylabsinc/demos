"""Bounded exposure/IMU alignment for the accepted stationary-base model.

This models attitude about the calibrated pelvis pivot, not base translation
or changed waist joints. It never reads hardware or changes acceptance gates.
"""

from collections import deque
from copy import deepcopy
import threading

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


class PoseHistory:
    def __init__(self, max_samples=512):
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, int)
            or not 1 <= max_samples <= 512
        ):
            raise ValueError("Pose history capacity must be between 1 and 512")
        self._samples = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def add(self, timestamp, rpy, waist_q):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("Pose timestamp must be finite")
        sample = dict(
            timestamp=timestamp,
            rpy=_vector(rpy, "rpy").tolist(),
            waist_q=_vector(waist_q, "waist_q").tolist(),
        )
        with self._lock:
            if self._samples and timestamp <= self._samples[-1]["timestamp"]:
                raise ValueError("Pose timestamps must advance")
            self._samples.append(sample)

    def nearest(self, exposure_timestamp):
        exposure_timestamp = float(exposure_timestamp)
        if not np.isfinite(exposure_timestamp):
            raise ValueError("Exposure timestamp must be finite")
        with self._lock:
            if not self._samples:
                raise RuntimeError("No recorded pose for camera exposure")
            sample = min(
                self._samples,
                key=lambda item: abs(item["timestamp"] - exposure_timestamp),
            )
            # Numerical slack only accommodates binary representation of 20 ms.
            if abs(sample["timestamp"] - exposure_timestamp) > 0.020 + 1e-12:
                raise RuntimeError(
                    "Recorded pose differs from camera exposure by more than 20 ms"
                )
            return deepcopy(sample)


def corrected_config(config, pose):
    if config.get("calibration_method") != "operator_accepted_factory_table":
        raise ValueError("Pose alignment requires accepted factory/table calibration")
    reference = _vector(config["reference_imu_rpy"], "reference_imu_rpy")
    current = _vector(pose["rpy"], "rpy")
    reference_waist = _vector(config["reference_base_q"][12:15], "reference waist_q")
    waist = _vector(pose["waist_q"], "waist_q")
    delta_rpy = (current - reference + np.pi) % (2 * np.pi) - np.pi
    if np.max(np.abs(delta_rpy)) > 0.04 + 1e-12:
        raise RuntimeError("Body attitude exceeds the 0.04 rad alignment bound")
    if np.max(np.abs(waist - reference_waist)) > 0.002 + 1e-12:
        raise RuntimeError("Waist moved beyond the 0.002 rad unmodeled alignment bound")
    timestamp = float(pose["timestamp"])
    if not np.isfinite(timestamp):
        raise ValueError("Pose timestamp must be finite")
    transforms = {}
    for key in ("camera_to_world", "pelvis_to_world"):
        transform = np.asarray(config[key], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{key} must be a finite 4 by 4 transform")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
            raise ValueError(f"{key} must be a homogeneous transform")
        transforms[key] = transform
    # The accepted world's zero yaw follows the reference robot forward axis.
    before = Rotation.from_euler("xyz", [reference[0], reference[1], 0]).as_matrix()
    after = Rotation.from_euler(
        "xyz", [current[0], current[1], delta_rpy[2]]
    ).as_matrix()
    delta = after @ before.T
    pivot = transforms["pelvis_to_world"][:3, 3]
    result = deepcopy(config)
    for key, original in transforms.items():
        transformed = original.copy()
        transformed[:3, :3] = delta @ original[:3, :3]
        transformed[:3, 3] = pivot + delta @ (original[:3, 3] - pivot)
        result[key] = transformed.tolist()
    result["pose_alignment"] = dict(
        timestamp=timestamp,
        rpy=current.tolist(),
        waist_q=waist.tolist(),
        delta_rpy_rad=delta_rpy.tolist(),
        pivot_world_m=pivot.tolist(),
        base_translation_modeled=False,
    )
    return result
