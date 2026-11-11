"""Shared physical/simulation arm gains; no changes to non-arm motor gains.

HANDOFF_ARM_KP: shoulder/elbow Nm/rad, default 150, range (0, 150].
Wrist gains remain 40/1.5; shoulder/elbow damping remains 3 Nm/(rad/s).
Changing this variable requires restarting the controller while released.
"""
import os
import numpy as np


def arm_gains(kp=None):
    kp=float(os.environ.get('HANDOFF_ARM_KP','150') if kp is None else kp)
    if not np.isfinite(kp) or not 0<kp<=150:
        raise ValueError('HANDOFF_ARM_KP must be finite, greater than zero and at most 150 Nm/rad')
    return np.asarray([kp]*4+[40.]*3),np.asarray([3.]*4+[1.5]*3)
