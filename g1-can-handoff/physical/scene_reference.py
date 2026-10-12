"""Independent measured tabletop reference shared by RGB-D and robot FK.

This module opens no endpoints or publishers. Static table landmarks exclude
the target; one bounded correction transforms camera and pelvis together.
Failure never falls back to an uncorrected or relabeled stale observation.
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
    iterations = 1000
    for iteration in range(1000):
        ids = rng.choice(len(source), 3, replace=False)
        if np.linalg.norm(np.cross(source[ids[1]]-source[ids[0]],
                                   source[ids[2]]-source[ids[0]])) < 1e-6:
            continue
        r, t = _rigid(source[ids], reference[ids])
        inside = np.linalg.norm(source @ r.T + t - reference, axis=1) < .006
        if inside.sum() > best.sum():
            best = inside
            ratio = float(best.mean())
            if ratio >= 1:
                iterations = 32
            elif ratio > 0:
                iterations = max(32, int(np.ceil(np.log(.001)/np.log(1-ratio**3))))
        if iteration+1 >= iterations:
            break
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


def validate_scene_transform(transform):
    from scipy.spatial.transform import Rotation
    transform = np.asarray(transform, dtype=float)
    transform_point(np.zeros(3), transform)
    if (np.linalg.norm(Rotation.from_matrix(transform[:3,:3]).as_rotvec()) > .04
            or np.linalg.norm(transform[:3,3]) > .02):
        raise ValueError('Scene correction exceeds bounded pose range')
    return transform.copy()


def apply_scene_transform(config, transform):
    from copy import deepcopy
    transform = validate_scene_transform(transform)
    result = deepcopy(config)
    for name in ('camera_to_world', 'pelvis_to_world'):
        original = np.asarray(config[name], dtype=float)
        transform_point(np.zeros(3), original)
        result[name] = (transform @ original).tolist()
    return result


class TableSceneReference:
    """One immutable exact-frame tabletop anchor; never track the can as a landmark."""
    def __init__(self):
        import cv2
        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.detector = cv2.SIFT_create(nfeatures=1800)
        self.anchor = None
        self.center = None

    def _features(self, frame, config):
        from camera_input import world_points
        from yolo_client import fresh
        fresh(frame)
        world = world_points(frame, config)
        mask = np.isfinite(world).all(axis=2) & (np.abs(world[:,:,2]-config['table_height_m']) < .01)
        mask &= np.sum((world[:,:,:2]-self.center)**2,axis=2) > .09**2
        keypoints, descriptors = self.detector.detectAndCompute(frame['color_bgr'], (mask*255).astype(np.uint8))
        if descriptors is None or len(keypoints)<30:
            raise RuntimeError('Static tabletop landmarks absent/insufficient')
        return world,keypoints,descriptors

    def capture(self, frame, config, center):
        from yolo_client import fresh
        if self.anchor is not None:
            raise RuntimeError('Static table reference cannot silently rebase')
        center = np.asarray(center, dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise ValueError('Scene anchor requires finite observed target position')
        self.center = center[:2].copy()
        world, keypoints, descriptors = self._features(frame, config)
        fresh(frame)
        self.anchor = dict(world=world,keypoints=keypoints,descriptors=descriptors,
                           metadata=dict(frame['meta']),timestamp=frame['timestamp'])
        return self._diagnostic(frame, np.eye(4), dict(anchor=True,landmarks=len(keypoints)))

    def _diagnostic(self, frame, transform, metrics):
        return dict(transform=validate_scene_transform(transform).tolist(),
                    timestamp=float(frame['timestamp']),frame_id=frame['meta']['frame_id'],
                    camera_serial=frame['meta']['camera_serial'],
                    stream_id=frame['meta'].get('stream_id'),
                    reference_frame_id=self.anchor['metadata']['frame_id'],
                    reference_timestamp=self.anchor['timestamp'],metrics=metrics,
                    source='independent_static_table_landmarks')

    def correct(self, frame, config):
        from yolo_client import fresh
        if self.anchor is None:
            raise RuntimeError('Static table reference has not been captured')
        a=self.anchor
        for key in ('camera_serial','stream_id','intrinsics'):
            if frame['meta'].get(key)!=a['metadata'].get(key):
                raise RuntimeError('Static scene reference camera/stream/calibration changed')
        if frame['timestamp']<=a['timestamp'] or frame['meta']['frame_id']<=a['metadata']['frame_id']:
            raise RuntimeError('Static scene reference requires advancing exact frames')
        world,keypoints,descriptors=self._features(frame,config)
        pairs=self.cv2.BFMatcher().knnMatch(a['descriptors'],descriptors,k=2)
        source, reference=[],[]
        for pair in pairs:
            if len(pair)!=2:continue
            match,runner=pair
            if match.distance>=.65*runner.distance:continue
            u0,v0=np.rint(a['keypoints'][match.queryIdx].pt).astype(int)
            u,v=np.rint(keypoints[match.trainIdx].pt).astype(int)
            p,q=world[v,u],a['world'][v0,u0]
            if np.isfinite(p).all() and np.isfinite(q).all():
                source.append(p);reference.append(q)
        transform,metrics=estimate_scene_reference(source,reference)
        fresh(frame)
        return apply_scene_transform(config,transform),self._diagnostic(frame,transform,metrics)
