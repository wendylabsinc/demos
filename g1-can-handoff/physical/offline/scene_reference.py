"""Offline candidate for correcting a common RGB-D scene reference drift.

No camera endpoints, SDK, publishers or runtime admission changes. The caller
must exclude the target and robot from independent stationary landmarks.
Applying this to hardware additionally requires consistent camera AND pelvis
transforms and revalidation of the path; correcting the target alone is unsafe.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _rigid(source, reference):
    a, b = source.mean(0), reference.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (reference - b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, b - rotation @ a


def estimate_scene_reference(source, reference, *, seed=17):
    """Fit current-to-reference rigid coordinates from independent landmarks."""
    source, reference = np.asarray(source, float), np.asarray(reference, float)
    if (source.ndim != 2 or source.shape[1:] != (3,) or source.shape != reference.shape
            or len(source) < 30 or len(source) > 2000
            or not np.isfinite(source).all() or not np.isfinite(reference).all()):
        raise ValueError('Require 30..2000 paired finite stationary XYZ landmarks')
    best = np.zeros(len(source), dtype=bool)
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        ids = rng.choice(len(source), 3, replace=False)
        if np.linalg.norm(np.cross(source[ids[1]]-source[ids[0]],
                                   source[ids[2]]-source[ids[0]])) < 1e-6:
            continue
        r, t = _rigid(source[ids], reference[ids])
        inside = np.linalg.norm(source @ r.T + t - reference, axis=1) < .006
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 30 or best.mean() < .6:
        raise ValueError('Stationary landmark correspondence is insufficient/ambiguous')
    for points in (source[best], reference[best]):
        spread = np.linalg.svd(points-points.mean(0), compute_uv=False)/np.sqrt(best.sum())
        if spread[1] < .05:
            raise ValueError('Stationary landmarks must span a broad surface')
    r, t = _rigid(source[best], reference[best])
    residual = np.linalg.norm(source[best] @ r.T + t - reference[best], axis=1)
    angle = float(np.linalg.norm(Rotation.from_matrix(r).as_rotvec()))
    if np.quantile(residual, .95) > .006 or np.median(residual) > .003:
        raise ValueError('Stationary landmark rigid fit exceeds depth uncertainty bounds')
    if angle > .04 or np.linalg.norm(t) > .02:
        raise ValueError('Scene correction exceeds bounded offline candidate range')
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    return transform, dict(matches=len(source), inliers=int(best.sum()),
                          median_residual_mm=float(np.median(residual)*1000),
                          p95_residual_mm=float(np.quantile(residual,.95)*1000),
                          rotation_rad=angle, translation_m=float(np.linalg.norm(t)))


def transform_point(point, transform):
    point, transform = np.asarray(point, float), np.asarray(transform, float)
    if (point.shape != (3,) or transform.shape != (4,4)
            or not np.isfinite(point).all() or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0,0,0,1], atol=1e-12, rtol=0)
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-10,rtol=0)
            or abs(np.linalg.det(transform[:3,:3])-1)>1e-10):
        raise ValueError('Scene transform requires a finite proper rigid transform')
    return transform[:3,:3] @ point + transform[:3,3]
