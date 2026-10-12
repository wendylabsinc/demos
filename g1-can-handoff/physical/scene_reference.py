"""Independent measured static-scene reference shared by RGB-D and robot FK.

This module opens no endpoints or publishers. Static table landmarks exclude
the target; one bounded correction transforms camera and pelvis together.
Failure never falls back to an uncorrected or relabeled stale observation.
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation


def scene_fit_residual_m():
    """Existing rigid-fit residual allowance, configurable up to 12 mm."""
    try:
        value = float(os.getenv('HANDOFF_SCENE_FIT_RESIDUAL_MM', '6'))
        if not np.isfinite(value) or not 0 < value <= 12:
            raise ValueError()
    except (TypeError, ValueError) as exc:
        raise ValueError('Scene fit residual must be finite and within (0, 12] mm') from exc
    return value / 1000


def reference_region():
    region = os.getenv('HANDOFF_SCENE_REFERENCE_REGION', 'table')
    if region not in ('table', 'background'):
        raise ValueError('Scene reference region must be table or background')
    return region


def reference_source():
    return ('independent_static_background_landmarks' if reference_region() == 'background'
            else 'independent_static_table_landmarks')


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
    residual_limit = scene_fit_residual_m()
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
        inside = np.linalg.norm(source @ r.T + t - reference, axis=1) < residual_limit
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
    if np.quantile(residual, .95) > residual_limit or np.median(residual) > residual_limit / 2:
        raise ValueError('Stationary landmark rigid fit exceeds depth uncertainty bounds')
    if angle > .04 or np.linalg.norm(t) > .02:
        raise ValueError('Scene correction exceeds bounded offline candidate range')
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r, t
    return transform, dict(matches=len(source), inliers=int(best.sum()),
                          median_residual_mm=float(np.median(residual)*1000),
                          p95_residual_mm=float(np.quantile(residual,.95)*1000),
                          rotation_rad=angle, translation_m=float(np.linalg.norm(t)),
                          residual_limit_mm=residual_limit * 1000)


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
        self.region = reference_region()
        self.detector = (cv2.SIFT_create(nfeatures=1800) if self.region == 'background'
                         else cv2.ORB_create(nfeatures=2000, fastThreshold=7))
        self.norm = cv2.NORM_L2 if self.region == 'background' else cv2.NORM_HAMMING
        self.anchor = None
        self.center = None

    def _features(self, frame, config):
        from yolo_client import fresh
        fresh(frame)
        # Descriptor extraction proposes image features. Project only their
        # measured depth pixels, then retain independent tabletop support;
        # can geometry still receives its separate full exact-frame fit.
        region = getattr(self, 'region', 'table')
        feature_mask = None
        if region == 'background':
            # The fixed NX2 camera view's lower-left area contains the pickup
            # target and approaching hand. Reuse the recorded diagnostic's
            # exclusion, scaled with image dimensions; the world-space can
            # exclusion below independently covers target placement changes.
            height, width = frame['depth_m'].shape
            feature_mask = np.full((height, width), 255, np.uint8)
            feature_mask[int(height * 255 / 480):, :int(width * 260 / 640)] = 0
        keypoints, descriptors = self.detector.detectAndCompute(frame['color_bgr'], feature_mask)
        if descriptors is None:
            raise RuntimeError('Static tabletop landmarks absent/insufficient')
        xy = np.rint([keypoint.pt for keypoint in keypoints]).astype(int)
        u, v = xy[:, 0], xy[:, 1]
        depth = frame['depth_m'][v, u]
        transform = np.asarray(config['camera_to_world'])
        points = (frame['rays'][v, u] * depth[:, None]) @ transform[:3, :3].T + transform[:3, 3]
        valid = np.isfinite(points).all(axis=1) & (depth > .1) & (depth < 2.5)
        if region == 'table':
            valid &= np.abs(points[:, 2]-config['table_height_m']) < .01
        valid &= np.sum((points[:, :2]-self.center)**2, axis=1) > .09**2
        indices = np.flatnonzero(valid)
        if len(indices) < 30:
            raise RuntimeError('Static tabletop landmarks absent/insufficient')
        world = {(int(v[i]), int(u[i])): points[i] for i in indices}
        return world, [keypoints[i] for i in indices], descriptors[indices]

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
                    source=reference_source(), region=getattr(self, 'region', 'table'))

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
        pairs=self.cv2.BFMatcher(self.norm).knnMatch(a['descriptors'],descriptors,k=2)
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
