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


def visual_threads():
    """Use one or two OpenCV threads; robot control has separate CPU affinity."""
    raw = os.getenv('HANDOFF_VISUAL_THREADS', '1')
    if raw not in ('1', '2'):
        raise ValueError('Visual threads must be 1 or 2')
    return int(raw)


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
    # Evaluate the same seeded, ordered hypotheses in small matrix batches.
    # Adaptive termination and first-best tie behavior match the scalar path;
    # this removes Python/SVD overhead without thinning measured landmarks.
    done = False
    for offset in range(0, 1000, 32):
        ids = np.asarray([rng.choice(len(source), 3, replace=False)
                          for _ in range(min(32, 1000 - offset))])
        samples, targets = source[ids], reference[ids]
        usable = (np.linalg.norm(np.cross(samples[:, 1] - samples[:, 0],
                                          samples[:, 2] - samples[:, 0]), axis=1) >= 1e-6)
        centers, target_centers = samples.mean(1), targets.mean(1)
        covariance = ((samples - centers[:, None]).transpose(0, 2, 1)
                      @ (targets - target_centers[:, None]))
        u, _, vt = np.linalg.svd(covariance)
        rotations = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
        reflected = np.linalg.det(rotations) < 0
        vt[reflected, -1] *= -1
        rotations = vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1)
        translations = target_centers - (rotations @ centers[..., None])[..., 0]
        residuals = np.linalg.norm(source[None] @ rotations.transpose(0, 2, 1)
                                   + translations[:, None] - reference[None], axis=2)
        candidates = residuals < residual_limit
        for index in range(len(ids)):
            iteration = offset + index
            if not usable[index]:
                continue
            inside = candidates[index]
            if inside.sum() > best.sum():
                best = inside
                ratio = float(best.mean())
                if ratio >= 1:
                    iterations = 32
                elif ratio > 0:
                    iterations = max(32, int(np.ceil(np.log(.001)/np.log(1-ratio**3))))
            if iteration + 1 >= iterations:
                done = True
                break
        if done:
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
        cv2.setNumThreads(visual_threads())
        self.cv2 = cv2
        self.region = reference_region()
        self.detector = (cv2.SIFT_create(nfeatures=1800) if self.region == 'background'
                         else cv2.ORB_create(nfeatures=2000, fastThreshold=7))
        self.norm = cv2.NORM_L2 if self.region == 'background' else cv2.NORM_HAMMING
        self.anchor = None
        self.center = None
        self.tracking = os.getenv('HANDOFF_SCENE_TRACKING', 'sift')
        if self.tracking not in ('sift', 'flow'):
            raise ValueError('Scene tracking must be sift or flow')

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
        if self.tracking == 'flow' and self.region == 'background':
            # The reference image and its measured XYZ remain immutable. Every
            # exposure is tracked directly against this image, never a chain
            # of estimated motions that can accumulate drift.
            self.anchor['gray'] = self.cv2.cvtColor(frame['color_bgr'], self.cv2.COLOR_BGR2GRAY)
            self.anchor['pixels'] = np.asarray([k.pt for k in keypoints], np.float32).reshape(-1, 1, 2)
            self.anchor['gray'].setflags(write=False)
            self.anchor['pixels'].setflags(write=False)
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
        fresh(frame)
        recovery = None
        method = 'sift' if self.region == 'background' else 'orb'
        if self.tracking == 'flow' and self.region == 'background':
            try:
                source, reference = self._flow_correspondences(frame, config)
                transform, metrics = estimate_scene_reference(source, reference)
                method = 'immutable_anchor_flow'
            except (ValueError, RuntimeError) as exc:
                # Loss of optical correspondence never reuses a transform.
                # Recover with the original descriptors on this same exposure;
                # the existing rigid-fit and freshness bounds still apply.
                recovery = str(exc)
        if method != 'immutable_anchor_flow':
            source, reference = self._descriptor_correspondences(frame, config)
            transform, metrics = estimate_scene_reference(source, reference)
        metrics['tracking_method'] = method
        if recovery is not None:
            metrics['flow_recovery_reason'] = recovery
        fresh(frame)
        return apply_scene_transform(config,transform),self._diagnostic(frame,transform,metrics)

    def _descriptor_correspondences(self, frame, config):
        a = self.anchor
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
        return source, reference

    def _flow_correspondences(self, frame, config):
        a = self.anchor
        gray = self.cv2.cvtColor(frame['color_bgr'], self.cv2.COLOR_BGR2GRAY)
        parameters = dict(winSize=(21, 21), maxLevel=3,
                          criteria=(self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT, 30, .01))
        pixels, forward, error = self.cv2.calcOpticalFlowPyrLK(a['gray'], gray, a['pixels'], None, **parameters)
        if pixels is None or forward is None or error is None or not np.isfinite(pixels).all():
            raise RuntimeError('Static scene optical correspondence unavailable')
        back, backward, _ = self.cv2.calcOpticalFlowPyrLK(gray, a['gray'], pixels, None, **parameters)
        if back is None or backward is None or not np.isfinite(back).all():
            raise RuntimeError('Static scene reverse correspondence unavailable')
        valid = ((forward[:, 0] > 0) & (backward[:, 0] > 0)
                 & (np.linalg.norm(back - a['pixels'], axis=2)[:, 0] < .75)
                 & (error[:, 0] < 15))
        height, width = frame['depth_m'].shape
        u, v = np.rint(pixels[:, 0]).astype(int).T
        valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid &= ~((v >= int(height * 255 / 480)) & (u < int(width * 260 / 640)))
        indices = np.flatnonzero(valid)
        u, v = u[indices], v[indices]
        depth = frame['depth_m'][v, u]
        transform = np.asarray(config['camera_to_world'])
        points = ((frame['rays'][v, u] * depth[:, None]) @ transform[:3, :3].T
                  + transform[:3, 3])
        valid = (np.isfinite(points).all(axis=1) & (depth > .1) & (depth < 2.5)
                 & (np.sum((points[:, :2] - self.center)**2, axis=1) > .09**2))
        source = points[valid]
        reference = []
        for index in indices[valid]:
            u0, v0 = np.rint(a['keypoints'][index].pt).astype(int)
            reference.append(a['world'][v0, u0])
        return source, reference
