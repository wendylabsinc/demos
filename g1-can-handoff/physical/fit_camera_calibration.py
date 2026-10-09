"""Fit a rigid camera->robot world transform from independently measured points.
Input JSON: camera_points_m, world_points_m, validation_indices (held out).
Use at least four non-collinear training points and two held-out points.
This writes a calibration candidate; it does not commission or move a robot.
"""

import argparse
import json
import numpy as np
from physical_core import rigid


def fit(data):
    camera = np.asarray(data["camera_points_m"], float)
    world = np.asarray(data["world_points_m"], float)
    held = np.asarray(data["validation_indices"], int)
    if (
        camera.shape != world.shape
        or camera.ndim != 2
        or camera.shape[1] != 3
        or not np.isfinite(camera).all()
        or not np.isfinite(world).all()
    ):
        raise ValueError("Matched finite Nx3 measurements required")
    if len(set(held)) < 2 or np.any(held < 0) or np.any(held >= len(camera)):
        raise ValueError("At least two distinct held-out validation points required")
    train = np.ones(len(camera), bool)
    train[held] = False
    if train.sum() < 4:
        raise ValueError("At least four training points required")
    a, b = camera[train], world[train]
    ac, bc = a.mean(0), b.mean(0)
    if np.linalg.matrix_rank(a - ac) < 2:
        raise ValueError("Calibration points are collinear")
    u, s, vt = np.linalg.svd((a - ac).T @ (b - bc))
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1] *= -1
        rot = vt.T @ u.T
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = bc - rot @ ac
    rigid(t)
    errors = np.linalg.norm(camera @ rot.T + t[:3, 3] - world, axis=1)
    validation_rmse = float(np.sqrt(np.mean(errors[held] ** 2)))
    if validation_rmse > 0.01 or float(errors[held].max()) > 0.015:
        raise ValueError("Held-out calibration exceeds 10 mm RMS / 15 mm maximum")
    return {
        "commissioned": False,
        "camera_to_world": t.tolist(),
        "position_validation_rmse_m": validation_rmse,
        "held_out_errors_m": errors[held].tolist(),
        "training_rmse_m": float(np.sqrt(np.mean(errors[train] ** 2))),
        "measurement_evidence": data,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("measurements")
    p.add_argument("output")
    a = p.parse_args()
    with open(a.measurements) as f:
        result = fit(json.load(f))
    with open(a.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
