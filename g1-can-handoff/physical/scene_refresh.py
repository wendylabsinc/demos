"""Refresh an already authorized model estimate from a new camera frame.

This retains estimated provenance and never commissions an unapproved setup.
Runtime persists the result only after the complete passive plan passes.
"""
import time
from copy import deepcopy
import numpy as np
from factory_table_candidate import candidate
from physical_core import calibration, snapshot_guard


def refresh_scene(previous, frame, snapshot, serial, exposure_pose=None):
    if previous.get('calibration_method') != 'operator_accepted_factory_table':
        raise ValueError('Automatic refresh requires approved factory/table mode')
    # Existing authorization is reusable; its old scene age is not. Validate
    # all other commissioning fields before processing a fresh observation.
    authorized = dict(previous, accepted_at_unix_s=time.time())
    calibration(authorized, serial)
    snapshot_guard(snapshot)
    if frame['meta']['camera_serial'] != serial:
        raise ValueError('Refresh frame camera identity mismatch')
    if abs(previous['table_height_m']-.762) > 1e-6:
        raise ValueError('Automatic refresh requires the approved 0.762 m table')
    # Health belongs to the current live snapshot checked above. Geometry and
    # its reference attitude belong to the camera's immutable exposure time.
    # Other joints remain the fresh measured values; waist is the only joint
    # group in the camera's torso-to-pelvis transform.
    measured = deepcopy(snapshot)
    if exposure_pose is not None:
        timestamp = float(exposure_pose['timestamp'])
        exposure = float(frame['timestamp'])
        if not np.isfinite([timestamp, exposure]).all() or abs(timestamp-exposure) > .020 + 1e-12:
            raise ValueError('Calibration exposure pose must be within 20 ms of frame')
        for key in ('rpy', 'waist_q'):
            values = np.asarray(exposure_pose[key], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f'Calibration exposure {key} must contain three finite values')
        measured['body']['q'] = list(measured['body']['q'])
        measured['body']['q'][12:15] = list(exposure_pose['waist_q'])
        measured['body']['imu_rpy'] = list(exposure_pose['rpy'])
    fresh = candidate(frame, measured)
    result = dict(previous)
    result.update(fresh)
    result.update(commissioned=True, calibration_method='operator_accepted_factory_table',
                  accepted_at_unix_s=time.time(), scene_refresh_automatic=True,
                  reference_base_q=list(measured['body']['q'][:15]),
                  reference_imu_rpy=list(measured['body']['imu_rpy']),
                  independent_validation=False, position_validation_rmse_m=None)
    if exposure_pose is not None:
        result['calibration_exposure_pose'] = dict(timestamp=timestamp,
            frame_timestamp=exposure, skew_ms=(timestamp-exposure)*1000,
            rpy=list(measured['body']['imu_rpy']), waist_q=list(measured['body']['q'][12:15]))
    return calibration(result, serial)
