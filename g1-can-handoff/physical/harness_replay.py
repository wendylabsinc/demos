"""Bounded, operator-requested arm tracking replay in a clear harness workspace.

No pickup/perception policy is run. No finger commands are sent. Self collision,
joint limits, feedback health, tracking, torque slew, cancellation and deadline
checks remain active. Environment absence is an explicit operator attestation.
"""

import hashlib
import json
from pathlib import Path
import time
import urllib.request
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from adapter.arm_gains import arm_gains
from physical_core import (
    PhysicalPlant,
    calibrated_scene,
    snapshot_guard,
    ARM_GRAVITY_LIMITS_NM,
    ARM_GRAVITY_SLEW_NM_S,
)
from staging import joint_path


def validate_fixture(value, limits):
    q = np.asarray(value["left_q"], dtype=float)
    tau = np.asarray(value["left_tau"], dtype=float)
    limits = np.asarray(limits, dtype=float)[:7]
    if (
        limits.shape != (7, 2)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Invalid commissioned replay joint limits")
    if (
        q.ndim != 2
        or q.shape[1] != 7
        or not 2 <= len(q) <= 4000
        or tau.shape != q.shape
    ):
        raise ValueError(
            "Replay requires 2–4000 paired seven-joint targets and torques"
        )
    if not np.isfinite(q).all() or not np.isfinite(tau).all():
        raise ValueError("Replay contains invalid values")
    if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
        raise ValueError("Replay exceeds commissioned joint limits")
    if np.any(np.abs(tau) > ARM_GRAVITY_LIMITS_NM[:7]):
        raise ValueError("Replay exceeds arm feedforward limits")
    if np.max(np.abs(np.diff(q, axis=0))) > 0.02:
        raise ValueError("Replay has discontinuous joint targets")
    if value.get("period_s") != 0.02:
        raise ValueError("Replay period must be 20 ms")
    return q, tau


def harness_config(previous, snapshot):
    # Only robot geometry and commissioned joint limits carry over. The table
    # and can are absent by operator confirmation, not by camera inference.
    config = dict(
        previous,
        calibration_method="harness_clear_workspace_joint_replay",
        support="harness",
        workspace_clear_confirmed=True,
        table_center_xy_m=[10.0, 10.0],
        table_half_size_xy_m=[2.0, 2.0],
    )
    root = np.eye(4)
    root[:3, 3] = [0.0, 0.0, 0.85]
    root[:3, :3] = Rotation.from_euler(
        "xyz", [*snapshot["body"]["imu_rpy"][:2], 0.0]
    ).as_matrix()
    config["pelvis_to_world"] = root.tolist()
    return config


def preflight(plant, targets):
    original = plant.ik.data.qpos.copy()
    try:
        plant.set_contact_phase("pregrasp")
        path, _ = joint_path(
            plant.ik,
            targets[0],
            np.asarray(plant.config["arm_limits_rad"])[:7],
            check=plant.check,
        )
        for goal in path:
            plant._joint_poses(goal)
            plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]] = goal
            mujoco.mj_forward(plant.ik.model, plant.ik.data)
        prior = targets[0]
        for target in targets:
            # Include each intervening edge, not just the recorded endpoints.
            for alpha in np.linspace(
                0.0,
                1.0,
                max(2, int(np.ceil(np.max(np.abs(target - prior)) / 0.004)) + 1),
            ):
                plant.check()
                plant.check_live_geometry(left_q=prior + (target - prior) * alpha)
            prior = target
    finally:
        plant.ik.data.qpos[:] = original
        mujoco.mj_forward(plant.ik.model, plant.ik.data)


def move_joint_path(runtime, plant, goal, label):
    plant.sync(plant.check())
    path, metadata = joint_path(
        plant.ik,
        goal,
        np.asarray(plant.config["arm_limits_rad"])[:7],
        check=plant.check,
    )
    runtime.event(
        "replay_transition", label=label, goal_q=np.asarray(goal).tolist(), **metadata
    )
    for waypoint in path:
        # Fingers are never commanded in this mode; use their fresh measured
        # geometry for both planning and the target collision checks.
        plant.left_hand = np.asarray(plant.check()["left"]["q"]).copy()
        plant.sync(plant.check())
        target, poses = plant._joint_poses(waypoint)
        plant._execute_poses(target, poses)


def command(runtime, plant, q, left_tau):
    started = time.monotonic()
    with plant.command_lock:
        snapshot = plant.check()
        plant.sync(snapshot)
        plant.check_live_geometry(left_q=q)
        error = np.asarray(snapshot["body"]["q"][15:22]) - q
        if np.max(np.abs(error)) > 0.15:
            raise RuntimeError("Measured arm tracking error exceeds 0.15 rad")
        data = plant.gravity_data
        data.qpos[:] = plant.ik.data.qpos
        data.qvel[:] = 0
        mujoco.mj_forward(plant.ik.model, data)
        requested = np.r_[left_tau, data.qfrc_bias[plant.arm_dofs][7:]]
        if not np.isfinite(requested).all():
            raise RuntimeError("Invalid replay feedforward torque")
        requested = np.clip(requested, -ARM_GRAVITY_LIMITS_NM, ARM_GRAVITY_LIMITS_NM)
        dt = min(0.02, max(0.0, started - plant.arm_tau_time))
        plant.arm_tau_command += np.clip(
            requested - plant.arm_tau_command,
            -ARM_GRAVITY_SLEW_NM_S * dt,
            ARM_GRAVITY_SLEW_NM_S * dt,
        )
        plant.arm_tau_time = started
        runtime.arm.ctrl_dual_arm(
            np.r_[q, plant.right_arm], plant.arm_tau_command.copy()
        )
        plant.left_arm = np.asarray(q).copy()
        row = dict(
            sample_monotonic_s=started,
            feedback_age_s=started - snapshot["body"]["time"],
            commanded_left_q=np.asarray(q).tolist(),
            measured_left_q=snapshot["body"]["q"][15:22],
            measured_left_dq=snapshot["body"]["dq"][15:22],
            temperature_max_c=snapshot["body"]["temperature_max"],
            error_rad=error.tolist(),
            feedforward_nm=plant.arm_tau_command.tolist(),
            sdk_reference=runtime.arm.arm_reference_report(),
        )
        runtime.event("joint_replay_sample", **row)
    time.sleep(max(0.0, 0.02 - (time.monotonic() - started)))
    return row


def run(runtime, data):
    if (
        data.get("harness_confirmed") is not True
        or data.get("workspace_clear_confirmed") is not True
    ):
        raise ValueError(
            "Harness replay requires explicit supported robot and clear arm-sweep workspace"
        )
    kp = float(data["kp"])
    arm_gains(kp)
    path = Path(data["fixture"]).resolve()
    if path.parent != Path("/data/joint-replays") or path.suffix != ".json":
        raise ValueError("Invalid replay fixture path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["fixture_sha256"]:
        raise ValueError("Replay fixture checksum mismatch")
    previous = json.loads(Path("/data/calibration.json").read_text())
    if previous.get("device") != "unitree-g1-nx-2.local" or not previous.get(
        "commissioned"
    ):
        raise ValueError("Commissioned robot joint-limit identity is unavailable")
    targets, torques = validate_fixture(json.loads(raw), previous["arm_limits_rad"])
    snapshot = snapshot_guard(runtime.read())
    runtime.archive_guard()
    config = harness_config(previous, snapshot)
    model = calibrated_scene(config, x=10.0, y=-10.0)
    plant = PhysicalPlant(
        model,
        runtime.arm,
        None,
        runtime.read,
        config,
        runtime.event,
        cancel=runtime.cancel,
    )
    # All hand geometry follows live feedback; this sender writes nothing.
    plant.sender = SimpleNamespace(ctrl_dual_hand=lambda *a: None)
    origin = np.asarray(snapshot["body"]["q"][15:22]).copy()
    preflight(plant, targets)
    if runtime.cancel.is_set():
        raise RuntimeError("Operator cancelled before replay ownership")
    if runtime.arm is None:
        try:
            urllib.request.urlopen("http://127.0.0.1:8090/status", timeout=2).read()
        except Exception:
            pass
        else:
            raise RuntimeError("Existing onboard controller is still running")
        archive = runtime.archive_guard()
        sdk_age = archive["age_s"].get("rt/arm_sdk")
        if sdk_age is not None and sdk_age < 2:
            raise RuntimeError("Another arm SDK publisher is active")
        from g1_arm import G1_29_ArmController

        runtime.arm = G1_29_ArmController(
            motion_mode=True,
            simulation_mode=False,
            lowstate_hub=runtime.hub,
            record_lowcmd=runtime.command_written,
        )
    runtime.arm.set_arm_motion_limits(
        {"max_velocity_rad_s": 0.25, "max_acceleration_rad_s2": 1.0}
    )
    runtime.arm.set_arm_position_limits(config["arm_limits_rad"])
    runtime.arm.hold_arm_measured()
    gains, damping = arm_gains(kp)
    for index, (p, d) in enumerate(zip(np.tile(gains, 2), np.tile(damping, 2))):
        runtime.arm.set_joint_gains(index, float(p), float(d))
    plant.adapter = runtime.arm
    runtime.plant = plant
    runtime.active = True
    runtime.trial_deadline = time.monotonic() + 110
    runtime.phase = "HARNESS_CONTROL_SETTLING"
    runtime.event(
        "joint_replay_start",
        fixture_sha256=data["fixture_sha256"],
        kp=kp,
        actual_gains=runtime.arm.get_gains(),
        harness_confirmed=True,
        workspace_clear_confirmed=True,
        finger_commands_enabled=False,
        calibration=config,
    )
    plant.settle_control(archive_guard=runtime.archive_guard)
    plant.sync(plant.check())
    preflight(plant, targets)
    runtime.phase = "HARNESS_REPLAY_POSITIONING"
    move_joint_path(runtime, plant, targets[0], "to-recorded-start")
    runtime.phase = "HARNESS_REPLAY_PRELOAD"
    for _ in range(50):
        command(runtime, plant, targets[0], torques[0])
    runtime.phase = "HARNESS_JOINT_REPLAY"
    samples = []
    for q, tau in zip(targets, torques):
        samples.append(command(runtime, plant, q, tau))
    runtime.phase = "HARNESS_REPLAY_SETTLING"
    settle = []
    for _ in range(100):
        settle.append(command(runtime, plant, targets[-1], torques[-1]))
    errors = np.asarray([r["error_rad"] for r in samples])
    final = np.asarray(settle[-1]["error_rad"])
    final_window = np.asarray([r["error_rad"] for r in settle[-10:]])
    metrics = dict(
        kp=kp,
        commands_replayed=len(samples),
        rms_error_rad=float(np.sqrt(np.mean(errors**2))),
        max_tracking_error_rad=float(np.max(np.abs(errors))),
        shoulder_roll_rms_error_rad=float(np.sqrt(np.mean(errors[:, 1] ** 2))),
        final_error_rad=final.tolist(),
        final_max_error_rad=float(np.max(np.abs(final))),
        final_shoulder_roll_error_rad=float(abs(final[1])),
        final_window_max_error_rad=float(np.max(np.abs(final_window))),
        max_measured_speed_rad_s=float(
            np.max(np.abs([r["measured_left_dq"] for r in samples + settle]))
        ),
        max_temperature_c=max(r["temperature_max_c"] for r in samples + settle),
        settle_pass=bool(np.max(np.abs(final_window)) < 0.04),
    )
    runtime.event("joint_replay_metrics", **metrics)
    runtime.demo_result = {"joint_replay": metrics, "physical_lift_verified": False}
    if not metrics["settle_pass"]:
        raise RuntimeError("Harness replay arm did not settle within 0.04 rad")
    runtime.phase = "HARNESS_REPLAY_RETURNING"
    move_joint_path(runtime, plant, origin, "to-measured-origin")
    runtime.active = False
    runtime.relax_owned_arms("harness_replay_verified_return")
    return {
        "joint_replay": metrics,
        "physical_lift_verified": False,
        "returned_to_measured_origin": True,
        "control_released": True,
    }
