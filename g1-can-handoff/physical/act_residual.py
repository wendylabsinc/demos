"""Map fresh robot feedback into ACT's 94-value observation and bounded IK residual.

The policy controls the left arm only. The commissioned hand/contact controller,
right-arm hold, collision checks, telemetry guards and stop/release stay outside
the learned policy. This module never opens a DDS publisher.
"""
import os
import numpy as np
from physical_core import SIM_TO_BRIDGE, BODY_NAMES


def inference_budget_ms():
    """Per-call deadline in milliseconds, bounded by ACT's 100 ms period."""
    try:
        value = float(os.getenv('HANDOFF_ACT_INFERENCE_BUDGET_MS', '40'))
        if not np.isfinite(value) or not 0 < value <= 100:
            raise ValueError()
    except (TypeError, ValueError) as exc:
        raise ValueError('ACT inference budget must be finite and within (0, 100] ms') from exc
    return value


def motor_reference_max_age_ms():
    """Bounded SDK-reference age, in milliseconds; at most one ACT period."""
    try:
        value = float(os.getenv('HANDOFF_ACT_MOTOR_REFERENCE_MAX_AGE_MS', '40'))
        if not np.isfinite(value) or not 0 < value <= 100:
            raise ValueError()
    except (TypeError, ValueError) as exc:
        raise ValueError('ACT motor reference age must be finite and within (0, 100] ms') from exc
    return value


def actor_observation(q, velocity, arm_target, hand_position, nominal_hand_position,
                      estimated_can, held_age_s, phase_index, previous_action,
                      grasped, finger_target):
    q, velocity = np.asarray(q), np.asarray(velocity)
    if q.shape != (28,) or velocity.shape != (28,) or not 0 <= phase_index < 8:
        raise ValueError('ACT robot-state or phase shape mismatch')
    phase = np.eye(8, dtype=np.float32)[phase_index]
    result = np.concatenate((q, velocity, arm_target, hand_position,
                             nominal_hand_position, estimated_can, [held_age_s],
                             phase, previous_action, [float(grasped)],
                             np.asarray(finger_target) - q[14:21])).astype(np.float32)
    if result.shape != (94,) or not np.isfinite(result).all():
        raise ValueError('ACT observation must contain 94 finite values')
    return result


class ACTResidual:
    def __init__(self, model, plant, pose_supplier):
        import mujoco
        self.model, self.plant, self.pose_supplier = model, plant, pose_supplier
        self.inference_budget_ms = inference_budget_ms()
        self.motor_reference_max_age_ms = motor_reference_max_age_ms()
        if model.metadata['active_names'][:14] != BODY_NAMES[15:29]:
            raise ValueError('ACT body joint order differs from the robot adapter')
        if not np.allclose(plant.ik.model.site_pos[plant.ik.site], model.metadata['grasp_local'], atol=1e-8, rtol=0):
            raise ValueError('ACT grasp frame differs from the commissioned robot model')
        self.active = True
        self.action = np.zeros(5, dtype=np.float32)
        self.last_inference = None
        self.nominal_target = plant.left_arm.copy()
        self.wire = plant.left_arm.copy()
        self.held_at = None
        self.last_can = None
        self.last_observation = None
        self.observation_count = 0
        self.qadr = np.array([plant.ik.model.jnt_qposadr[plant.ik.model.joint(n).id]
                              for n in model.metadata['active_names']])
        self.dofs = np.array([plant.ik.model.jnt_dofadr[plant.ik.model.joint(n).id]
                              for n in model.metadata['active_names'][:7]])
        self.scratch = mujoco.MjData(plant.ik.model)
        self.jac_position = np.zeros((3, plant.ik.model.nv))
        self.jac_rotation = self.jac_position.copy()
        self.bounds = np.asarray(plant.config['arm_limits_rad'])[:7]
        self.residual_scale = np.asarray(model.metadata['residual_scale'])
        self.model.reset()

    def motor_reference(self, snapshot):
        """Read the SDK's filtered reference; never add a second servo filter."""
        adapter = self.plant.adapter
        if hasattr(adapter, 'arm_reference_report'):
            report = adapter.arm_reference_report()
            age = report.get('age_s', float('inf')) if report else float('inf')
            if not report or not 0 <= age <= self.motor_reference_max_age_ms / 1000:
                measured = f'{age * 1000:.1f} ms' if np.isfinite(age) else 'missing/nonfinite'
                raise RuntimeError('ACT motor reference is missing or stale '
                                   f'(age {measured}; limit {self.motor_reference_max_age_ms:g} ms)')
            if (report.get('max_velocity_rad_s') != .25
                    or report.get('max_acceleration_rad_s2') != 1):
                raise RuntimeError('ACT motor reference limits differ from its training servo')
            q = np.asarray(report['q'])
        elif hasattr(adapter, 'wire_arm_q'):
            q = np.asarray(adapter.wire_arm_q)  # SimulationIO; no hardware publisher.
        elif adapter is None:
            q = np.asarray(snapshot['body']['q'][15:29])  # Read-only analysis.
        else:
            raise RuntimeError('ACT adapter cannot expose its motor-clock reference')
        if q.shape != (14,) or not np.isfinite(q).all():
            raise RuntimeError('ACT motor reference must contain fourteen finite angles')
        return q[:7].copy()

    def validate_can(self, center):
        """Validate the input; physical placement admission belongs to IK."""
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('ACT can position is invalid')

    def update(self, nominal_target, snapshot):
        import mujoco
        from yolo_client import frame_age_limit_s
        plant = self.plant
        now = plant.clock.monotonic()
        self.nominal_target = np.asarray(nominal_target, dtype=float).copy()
        if self.nominal_target.shape != (7,) or not np.isfinite(self.nominal_target).all():
            raise ValueError('ACT nominal arm target must be seven finite values')
        self.wire = self.motor_reference(snapshot)
        phase_name = plant.ik.contact_phase
        if phase_name == 'pregrasp':
            phase_name = 'settling'
        if phase_name not in self.model.metadata['phase_names']:
            raise RuntimeError('ACT received an unknown motion phase: ' + phase_name)
        grasped = bool(getattr(getattr(plant, 'grip', None), 'grasped', False))
        if grasped and self.held_at is None:
            self.held_at = now
        # Compute the nominal target/FK/Jacobian in scratch data. Live measured
        # geometry remains unchanged and is checked on the returned command.
        self.scratch.qpos[:] = plant.ik.data.qpos
        self.scratch.qpos[self.qadr[:7]] = self.nominal_target
        # Site pose needs kinematics; its Jacobian also needs the current
        # center-of-mass/dof axes. Scratch dynamics and contacts are unused.
        mujoco.mj_kinematics(plant.ik.model, self.scratch)
        mujoco.mj_comPos(plant.ik.model, self.scratch)
        nominal_position = self.scratch.site_xpos[plant.ik.site].copy()
        if self.last_inference is None or now - self.last_inference >= .1 - 1e-7:
            if not grasped:
                center, timestamp = self.pose_supplier()
                age = now - timestamp
                if not 0 <= age <= frame_age_limit_s():
                    raise RuntimeError(f'ACT input exposure stale ({age*1000:.1f} ms)')
                self.last_can = np.asarray(center, dtype=float).copy()
            elif self.last_can is None:
                raise RuntimeError('ACT grasp has no retained pregrasp observation')
            q = plant.ik.data.qpos[self.qadr].copy()
            # Reuse PhysicalPlant.sync's canonical left/right joint ordering.
            velocity = np.r_[snapshot['body']['dq'][15:29],
                             np.asarray(snapshot['left']['dq'])[SIM_TO_BRIDGE],
                             snapshot['right']['dq']]
            observation = actor_observation(q, velocity, self.wire,
                plant.ik.data.site_xpos[plant.ik.site], nominal_position,
                self.last_can, 0 if self.held_at is None else now - self.held_at,
                self.model.metadata['phase_names'].index(phase_name), self.action,
                grasped, np.asarray(plant.left_hand)[SIM_TO_BRIDGE])
            started = plant.clock.monotonic()
            self.action = self.model.predict(observation)
            elapsed = plant.clock.monotonic() - started
            if elapsed > self.inference_budget_ms / 1000:
                raise RuntimeError(f'ACT inference exceeds the {self.inference_budget_ms:g} ms control budget ({elapsed*1000:.1f} ms)')
            self.last_inference = now
            self.last_observation = observation
            self.observation_count += 1
            plant.event('act_inference', action=self.action.tolist(),
                        observation=observation.tolist(), phase_name=phase_name,
                        elapsed_ms=elapsed*1000, budget_ms=self.inference_budget_ms,
                        checkpoint_sha256=self.model.metadata['checkpoint_sha256'])
        mujoco.mj_jacSite(plant.ik.model, self.scratch, self.jac_position,
                         self.jac_rotation, plant.ik.site)
        jac = np.vstack((self.jac_position[:, self.dofs], self.jac_rotation[1:3, self.dofs]))
        correction = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(5)*.0025, np.eye(5))
        delta = np.clip(correction @ (self.action * self.residual_scale), -.1, .1)
        target = self.nominal_target + delta
        if not np.isfinite(target).all() or np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
            raise RuntimeError('ACT corrected target exceeds commissioned joint limits')
        plant.event('act_reference', nominal_q=self.nominal_target.tolist(),
                    correction_rad=delta.tolist(), requested_q=target.tolist(),
                    previous_motor_reference_q=self.wire.tolist(),
                    reference_filter='existing SDK 100 Hz; max speed .25 rad/s, acceleration 1 rad/s2')
        return target

    def stop(self):
        self.active = False
        self.plant.event('act_policy_stopped', inference_count=self.observation_count,
                         retain_last_command=True)
