"""Refresh an already authorized model estimate from a new camera frame.

This retains estimated provenance and never commissions an unapproved setup.
Runtime persists the result only after the complete passive plan passes.
"""
import time
from factory_table_candidate import candidate
from physical_core import calibration, snapshot_guard


def refresh_scene(previous, frame, snapshot, serial):
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
    fresh = candidate(frame, snapshot)
    result = dict(previous)
    result.update(fresh)
    result.update(commissioned=True, calibration_method='operator_accepted_factory_table',
                  accepted_at_unix_s=time.time(), scene_refresh_automatic=True,
                  reference_base_q=list(snapshot['body']['q'][:15]),
                  reference_imu_rpy=list(snapshot['body']['imu_rpy']),
                  independent_validation=False, position_validation_rmse_m=None)
    return calibration(result, serial)
