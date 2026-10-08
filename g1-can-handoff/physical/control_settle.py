"""Bounded, measured ownership settling before camera calibration and planning."""
import time
import os
import numpy as np
from physical_core import snapshot_guard


def wait_for_stable_control(read,cancel,event,archive_guard,clock=time):
    try:
        max_speed=float(os.environ.get('HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S','1'))
        if not np.isfinite(max_speed) or not 0<max_speed<=1:raise ValueError()
    except ValueError as exc:
        raise ValueError('Invalid settling speed limit: require finite 0 < rad/s <= 1') from exc
    started=clock.monotonic();samples=[];last_archive=started;previous_stamp=None
    while clock.monotonic()-started<5.:
        if cancel.is_set():raise RuntimeError('Operator cancelled control settling')
        now=clock.monotonic();snapshot=snapshot_guard(read(),now=now);body=snapshot['body']
        stamp=body['time']
        if previous_stamp is not None and stamp<previous_stamp:
            raise RuntimeError('Body feedback timestamp moved backward during control settling')
        if stamp!=previous_stamp:
            pose=np.r_[body['imu_rpy'],body['q']]
            samples.append((stamp,pose,np.asarray(body['dq'])))
            samples=[row for row in samples if stamp-row[0]<=.55]
            previous_stamp=stamp
        if now-last_archive>=.25:archive_guard();last_archive=now
        if samples and samples[-1][0]-samples[0][0]>=.5:
            poses=np.asarray([row[1] for row in samples])
            poses[:,:3]=np.unwrap(poses[:,:3],axis=0)
            span=np.ptp(poses,axis=0)
            speed=float(np.max(np.abs([row[2] for row in samples])))
            if np.all(span[:3]<=.002) and np.all(span[3:18]<=.005) and np.all(span[18:]<=.015) and speed<max_speed:
                event('control_settled',duration_s=now-started,pose_span_rad=span.tolist(),max_speed_rad_s=speed,
                      max_speed_limit_rad_s=max_speed)
                return snapshot
        clock.sleep(.02)
    raise RuntimeError('Control handover did not settle within 5 seconds')
