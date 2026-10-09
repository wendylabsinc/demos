"""Measured finger-stop detection and a bounded proportional holding target.

Positions are radians; effort is a nominal P-term in Nm, not calibrated torque.
No DDS, hardware output, simulation truth or motor-fault bypass lives here.
"""
from collections import deque
import os
import numpy as np

GROUPS = {'thumb': (1, 2), 'middle': (3, 4), 'index': (5, 6)}


def setting(name, default, low, high):
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and in [{low}, {high}]')
    return value


class GentleGrasp:
    def __init__(self, start, goal, kp=2, stop_s=None, effort_nm=None):
        self.start = np.asarray(start, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.kp = float(kp)
        if (self.start.shape != (7,) or self.goal.shape != (7,)
                or not np.isfinite(self.start).all() or not np.isfinite(self.goal).all()
                or not np.isfinite(self.kp) or self.kp <= 0):
            raise ValueError('Grasp needs seven finite positions and positive finger P')
        self.stop_s = (setting('HANDOFF_GRIP_STOP_S', .2, .05, 1)
                       if stop_s is None else float(stop_s))
        self.effort_nm = (setting('HANDOFF_GRIP_HOLD_EFFORT_NM', .5, .05, 1)
                          if effort_nm is None else float(effort_nm))
        if not .05 <= self.stop_s <= 1 or not .05 <= self.effort_nm <= 1:
            raise ValueError('Invalid grasp stop duration or proportional effort')
        self.margin = self.effort_nm / self.kp
        self.stop_span_rad = setting('HANDOFF_GRIP_STOP_SPAN_RAD', .005, .001, .02)
        self.direction = np.sign(self.goal - self.start)
        self.history = deque()
        self.holding_joints = set()
        self.targets = self.goal.copy()
        self.last_time = None
        self.trigger_time = None
        self.all_stopped = False

    @property
    def groups(self):
        return [name for name, joints in GROUPS.items()
                if self.holding_joints.intersection(joints)]

    @property
    def grasped(self):
        # Empty endpoints/one blocked finger do not establish an object grip.
        return len(self.groups) >= 2 and self.all_stopped

    def update(self, sample_time, measured, command):
        q = np.asarray(measured, dtype=float)
        target = np.asarray(command, dtype=float).copy()
        if q.shape != (7,) or target.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(target).all():
            raise ValueError('Invalid grasp feedback or finger command')
        if self.grasped:
            return self.hold(q)
        if self.last_time is None or sample_time > self.last_time:
            self.last_time = sample_time
            self.history.append((sample_time, q.copy()))
            while len(self.history) > 2 and self.history[1][0] <= sample_time - self.stop_s:
                self.history.popleft()
            if sample_time - self.history[0][0] >= self.stop_s - 1e-9:
                span = np.ptp(np.array([r[1] for r in self.history]), axis=0)
                self.all_stopped = bool(np.all(span[self.direction != 0] <= self.stop_span_rad))
                gap = (target - q) * self.direction
                moved = (q - self.start) * self.direction
                stopped = np.flatnonzero((span <= self.stop_span_rad) & (gap >= .12) & (moved >= .02))
                for joint in stopped:
                    if int(joint) not in self.holding_joints:
                        self.holding_joints.add(int(joint))
                        self.targets[joint] = q[joint] + self.direction[joint] * min(
                            self.margin, abs(self.goal[joint] - q[joint]))
                if self.grasped:
                    self.trigger_time = sample_time
                    # Stop the closure ramp for the other joints too; never
                    # continue squeezing them toward the full closed endpoint.
                    for joint in range(7):
                        if joint not in self.holding_joints:
                            self.targets[joint] = target[joint]
                    return self.hold(q)
        for joint in self.holding_joints:
            target[joint] = np.clip(self.targets[joint], q[joint] - self.margin, q[joint] + self.margin)
        return target

    def hold(self, measured):
        q = np.asarray(measured, dtype=float)
        # Anchor the target at contact, rather than repeatedly adding a margin
        # to a moving finger. Clamp effort without advancing that anchor.
        return np.clip(self.targets, q - self.margin, q + self.margin)

    def status(self):
        return {'grasped': self.grasped, 'groups': self.groups,
                'stopped_for_s': self.stop_s, 'nominal_hold_effort_nm': self.effort_nm,
                'stop_span_rad': self.stop_span_rad,
                'holding_joint_slots': sorted(self.holding_joints),
                'all_closing_fingers_stopped': self.all_stopped,
                'trigger_time': self.trigger_time}
