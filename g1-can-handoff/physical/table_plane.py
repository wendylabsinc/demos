"""Fit a broad measured tabletop; tape establishes association and near edge.

A narrow tape strip alone gives a poorly constrained table normal. Fit and
validate on disjoint surface pixels, retaining the tape as an association
check rather than treating its low residual as calibration quality.
"""

import numpy as np


def _surface(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected table XYZ points")
    return points[
        np.isfinite(points).all(1) & (points[:, 2] > 0.1) & (points[:, 2] < 2)
    ]


def _broad_fit(points, seed_normal, seed_center):
    points = _surface(points)
    selected = np.abs((points - seed_center) @ seed_normal) < 0.025
    for _ in range(6):
        fitted = points[selected]
        if len(fitted) < 500:
            raise ValueError("Too few broad tabletop depth points")
        center = fitted.mean(0)
        _, sv, vt = np.linalg.svd(fitted - center, full_matrices=False)
        if sv[1] / np.sqrt(len(fitted)) < 0.05:
            raise ValueError(
                "Tabletop depth support is too narrow to determine its normal"
            )
        normal = vt[-1]
        if normal[2] > 0:
            normal = -normal
        updated = np.abs((points - center) @ normal) < 0.006
        if np.array_equal(updated, selected):
            break
        selected = updated
    fitted = points[selected]
    if len(fitted) < 500:
        raise ValueError("Too few broad tabletop depth points")
    center = fitted.mean(0)
    _, sv, vt = np.linalg.svd(fitted - center, full_matrices=False)
    if sv[1] / np.sqrt(len(fitted)) < 0.05:
        raise ValueError("Tabletop depth support is too narrow to determine its normal")
    normal = vt[-1]
    if normal[2] > 0:
        normal = -normal
    return normal, center, fitted, sv / np.sqrt(len(fitted))


def fit_table_plane(cloud, surface, holdout):
    cloud = _surface(cloud)
    if len(cloud) < 100:
        raise ValueError("Too few yellow tape depth points")
    rng = np.random.default_rng(17)
    best = np.zeros(len(cloud), dtype=bool)
    for _ in range(200):
        a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        inside = np.abs((cloud - a) @ normal) < 0.006
        if inside.sum() > best.sum():
            best = inside
    fitted = cloud[best]
    if len(fitted) < 100 or best.mean() < 0.7:
        raise ValueError("Yellow tape does not establish one dominant plane")
    center = fitted.mean(axis=0)
    _, _, vt = np.linalg.svd(fitted - center, full_matrices=False)
    up = vt[-1]
    if up[2] > 0:
        up = -up
    tape = fitted
    normal, center, broad, spread = _broad_fit(surface, up, center)
    check_normal, check_center, check_points, _ = _broad_fit(holdout, up, tape.mean(0))
    angle = float(np.arccos(np.clip(normal @ check_normal, -1, 1)))
    check_error = (check_points - center) @ normal
    if angle > np.deg2rad(0.5) or abs((check_center - center) @ normal) > 0.003:
        raise ValueError("Independent tabletop pixel groups disagree on the plane")
    check_rmse = float(np.sqrt(np.mean(check_error**2)))
    if check_rmse > 0.004 or np.quantile(np.abs(check_error), 0.95) > 0.006:
        raise ValueError("Table plane fails held-out tabletop depth validation")
    if np.quantile(np.abs((tape - center) @ normal), 0.9) > 0.008:
        raise ValueError("Broad tabletop plane does not agree with the tape")
    return (
        normal,
        center,
        tape,
        dict(
            rmse_m=float(np.sqrt(np.mean(((broad - center) @ normal) ** 2))),
            heldout_rmse_m=check_rmse,
            heldout_normal_angle_deg=float(np.degrees(angle)),
            fitted_points=len(broad),
            heldout_points=len(check_points),
            axis_spread_m=spread.tolist(),
            method="tape_associated_broad_tabletop_with_disjoint_pixel_validation",
        ),
    )
