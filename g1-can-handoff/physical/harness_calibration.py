"""Short arm-only PD and IK experiments, returning/releasing between trials.

No fingers, can detector or table-contact policy. The harness and empty sweep
are operator attestations. Encoder FK is model-derived, not optical ground truth.
"""

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from harness_replay import harness_config, preflight
from physical_core import PhysicalPlant, calibrated_scene, snapshot_guard

PROFILES = {
    "baseline": (0.0, "unit"),
    "jac_scaled": (0.0, "jac"),
    "nearby_regularized": (0.003, "jac"),
    "damped_jacobian": (0.0, "analytic_dls"),
}
PERIOD_S = 0.02


def validate_request(data):
    if any(data.get(k) is not True for k in (
        "harness_confirmed", "workspace_clear_confirmed", "hands_empty_confirmed"
    )):
        raise ValueError("Confirm harness, clear workspace and empty hands")
    kp, kd = float(data.get("kp", 50)), float(data.get("kd", 3))
    if not np.isfinite([kp, kd]).all() or not (20 <= kp <= 100 and 1 <= kd <= 6):
        raise ValueError("Calibration gains require 20<=P<=100 and 1<=D<=6")
    kind = data.get("kind", "joint")
    rise=float(data.get('max_temperature_rise_c',8))
    if not np.isfinite(rise) or not 8<=rise<=(12 if kind=='demo_replay' else 8):
        raise ValueError('Temperature rise limit is 8 C for small tests, 8..12 C for demo replay')
    if kind == "joint":
        joint = data.get("joint", 1)
        amplitude = float(data.get("amplitude_rad", 0.06))
        if type(joint) is not int or not 0 <= joint <= 6:
            raise ValueError("Joint index must be 0..6 in the left arm")
        if not np.isfinite(amplitude) or not 0.02 <= amplitude <= 0.12:
            raise ValueError("Joint amplitude must be 0.02..0.12 rad")
    elif kind == "demo_replay":
        path=Path(data.get("fixture", ""))
        digest=data.get("fixture_sha256", "")
        if path.parent != Path('/data/joint-replays') or path.suffix != '.json' or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Replay requires a content-bound fixture in /data/joint-replays')
    elif kind == "ik":
        if data.get("solver", "baseline") not in PROFILES:
            raise ValueError("Unknown IK profile")
        offset = np.asarray(data.get("offset_m", [0.015, 0, 0.015]), dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or np.linalg.norm(offset) > 0.04:
            raise ValueError("IK offset must be finite and within 40 mm")
    else:
        raise ValueError("Calibration kind must be joint or ik")
    return kp, kd


def load_replay(data):
    path=Path(data["fixture"]).resolve()
    if path.parent != Path('/data/joint-replays') or path.suffix != '.json':
        raise ValueError('Replay must be in the commissioned replay directory')
    raw=path.read_bytes()
    if len(raw)>1000000 or hashlib.sha256(raw).hexdigest()!=data['fixture_sha256']:
        raise ValueError('Replay fixture checksum or size invalid')
    value=json.loads(raw)
    q=np.asarray(value['left_q'],dtype=float)
    t=np.asarray(value['time_s'],dtype=float)
    if (q.ndim!=2 or q.shape[1]!=7 or not 2<=len(q)<=4000 or t.shape!=(len(q),)
            or not np.isfinite(q).all() or not np.isfinite(t).all() or t[0]!=0
            or not 0<t[-1]<=45 or np.any(np.diff(t)<=0)):
        raise ValueError('Replay requires ordered finite seven-joint samples within 45 seconds')
    if np.max(np.abs(np.diff(q,axis=0))/np.diff(t)[:,None])>.5:
        raise ValueError('Recorded trajectory exceeds 0.5 rad/s')
    return value, q, t


def segments(origin, goals, data):
    """Original timestamps drive replay; return follows the same checked path."""
    if data.get('kind')=='demo_replay':
        _,q,t=load_replay(data)
        interpolation=lambda elapsed: np.array([np.interp(elapsed,t,q[:,j]) for j in range(7)])
        yield 'entry',max(1.5,float(np.max(abs(q[0]-origin)))/.05),origin,q[0],None,.5
        yield 'recorded-forward',float(t[-1]),q[0],q[-1],interpolation,5.0
        yield 'recorded-return',float(t[-1]),q[-1],q[0],lambda elapsed:interpolation(t[-1]-elapsed),.5
        yield 'return-origin',max(1.5,float(np.max(abs(q[0]-origin)))/.05),q[0],origin,None,.5
    else:
        prior=origin
        for index,goal in enumerate(goals):
            yield str(index),max(1.5,float(np.max(abs(goal-prior)))/.05),prior,goal,None,.5
            prior=goal


def model_pose(plant, q):
    d = mujoco.MjData(plant.ik.model)
    d.qpos[:] = plant.ik.data.qpos
    d.qpos[plant.ik.qadr[plant.ik.arm]] = q
    mujoco.mj_forward(plant.ik.model, d)
    return (d.site_xpos[plant.ik.site].copy(),
            d.site_xmat[plant.ik.site].reshape(3, 3).copy(),
            d.qfrc_bias[plant.arm_dofs][:7].copy())


def damped_solve(plant, position, rotation):
    """Independent local analytic-Jacobian damped least-squares comparison."""
    from scipy.spatial.transform import Rotation
    model = plant.ik.model
    d = mujoco.MjData(model)
    d.qpos[:] = plant.ik.data.qpos
    adr = plant.ik.qadr[plant.ik.arm]
    dofs = model.jnt_dofadr[plant.ik.joints[plant.ik.arm]]
    q = d.qpos[adr].copy()
    low, high = model.jnt_range[plant.ik.joints[plant.ik.arm]].T
    for iteration in range(180):
        plant.check()
        d.qpos[adr] = q
        mujoco.mj_forward(model, d)
        orientation = Rotation.from_matrix(rotation @ d.site_xmat[plant.ik.site].reshape(3,3).T).as_rotvec()
        error = np.r_[position-d.site_xpos[plant.ik.site], .15*orientation]
        if np.linalg.norm(error[:3]) < .002 and np.linalg.norm(orientation) < .02:
            if plant.ik.collision_details(d):
                raise ValueError("Damped IK target collides")
            return q, {"method":"analytic_damped_least_squares", "iterations":iteration+1,"damping":.015}
        linear, angular = np.zeros((3,model.nv)), np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,d,linear,angular,plant.ik.site)
        jacobian = np.vstack([linear[:,dofs],.15*angular[:,dofs]])
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T+.015**2*np.eye(6),error)
        q = np.clip(q+np.clip(step,-.025,.025),low+.025,high-.025)
    raise ValueError("Damped IK did not converge within 180 iterations")


def plan(plant, data):
    """Plan from the exact supplied encoder state without owning hardware."""
    origin = plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]].copy()
    xyz, rotation, _ = model_pose(plant, origin)
    began = time.monotonic()
    metadata = {"origin_q": origin.tolist(), "origin_fk_m": xyz.tolist()}
    if data.get('kind')=='demo_replay':
        value, q, t=load_replay(data)
        goals=list(q)+list(q[::-1])
        metadata.update(fixture_sha256=data['fixture_sha256'],source=value['source'],
                        recorded_duration_s=float(t[-1]),recorded_samples=len(q),
                        timing_mode='original timestamps with 20 ms interpolation',
                        terminal_hold_s=5,return_mode='reverse recorded path')
        if data.get('model_output'):
            Path(data['model_output']).with_name('replay.json').write_text(json.dumps(value))
    elif data.get("kind", "joint") == "joint":
        goals = []
        for sign in (1, -1):
            q = origin.copy()
            q[data.get("joint", 1)] += sign * float(data.get("amplitude_rad", 0.06))
            goals.append(q)
    else:
        profile = data.get("solver", "baseline")
        weight, scaling = PROFILES[profile]
        plant.ik.ik_reference_weight = weight
        plant.ik.ik_position_tolerance = 0.005
        target = xyz + np.asarray(data.get("offset_m", [0.015, 0, 0.015]))
        if profile == "damped_jacobian":
            goal, settings = damped_solve(plant,target,rotation)
        else:
            plant.ik.IK_SETTINGS = dict(plant.ik.IK_SETTINGS, x_scale=1.0 if scaling == "unit" else "jac")
            goal, settings = plant.ik.solve(target,rotation), plant.ik.IK_SETTINGS
        goals = [goal]
        metadata.update(solver=profile, target_xyz_m=target.tolist(), target_rotation=rotation.tolist(),
                        reference_weight=weight, solver_settings=settings)
    goals.append(origin)
    limits = np.asarray(plant.config["arm_limits_rad"])[:7]
    excursion=1.5 if data.get("kind")=="demo_replay" else .3
    if any(np.max(np.abs(q - origin)) > excursion or np.any(q < limits[:, 0] + 0.025)
           or np.any(q > limits[:, 1] - 0.025) for q in goals):
        raise ValueError("Calibration targets exceed the experiment excursion or joint margin")
    # Check every interpolated edge at 4 mrad spacing, including return.
    dense = [origin]
    prior = origin
    for q in goals:
        dense.extend(prior + (q - prior) * t for t in np.linspace(0, 1, max(2, int(np.ceil(np.max(np.abs(q-prior))/.004))+1)))
        prior = q
    preflight(plant, dense)
    model_bytes = b""
    # Save the complete compiled model so later FK uses identical geometry.
    model_path = data.get("model_output")
    if model_path:
        mujoco.mj_saveModel(plant.ik.model, str(model_path), None)
        model_bytes = Path(model_path).read_bytes()
    metadata.update(planning_seconds=time.monotonic()-began, goals_q=[q.tolist() for q in (goals if data.get("kind")!="demo_replay" else [goals[0],goals[len(goals)//2-1],origin])],
                    estimated_gravity_nm=[model_pose(plant, q)[2].tolist() for q in (goals if data.get("kind")!="demo_replay" else [goals[0],goals[len(goals)//2-1],origin])],
                    model_sha256=hashlib.sha256(model_bytes).hexdigest() if model_bytes else None,
                    collision_preflight_passed=True, independent_cartesian_ground_truth=False)
    return origin, goals, metadata


def experiment_guard(snapshot, initial_temperatures, torque_limit=12.0, max_temperature_rise_c=8):
    """Lower limits for short characterization, independent of pickup overrides."""
    body = snapshot["body"]
    temperatures = np.asarray(body["temperatures"])[15:29]
    torque = np.asarray(body["tau_est"])[15:29]
    speed = np.asarray(body["dq"])[15:29]
    if not np.isfinite(np.r_[temperatures.ravel(), torque, speed]).all():
        raise RuntimeError("Invalid calibration feedback")
    if np.max(temperatures[:, 0]) >= 60 or np.max(temperatures[:, 1]) >= 65:
        raise RuntimeError("Calibration temperature ceiling reached (60 casing / 65 winding C)")
    if np.max(temperatures - initial_temperatures) >= max_temperature_rise_c:
        raise RuntimeError(f"Calibration temperature rose by {max_temperature_rise_c:g} C")
    limits = np.tile([torque_limit]*4 + [4.0]*3, 2)
    if np.any(np.abs(torque) > limits):
        raise RuntimeError("Calibration measured torque ceiling reached")
    if np.max(np.abs(speed)) > 0.7:
        raise RuntimeError("Calibration measured speed exceeds 0.7 rad/s")


def metrics(rows):
    if not rows:
        return {"samples": 0}
    err = np.asarray([r["error_rad"] for r in rows])
    torque = np.asarray([r["measured_tau_est_nm"] for r in rows])
    temp = np.asarray([r["temperatures_c"] for r in rows])
    settled = [r for r in rows if r["segment"].endswith("hold")]
    return dict(samples=len(rows), rms_error_rad=np.sqrt(np.mean(err**2, axis=0)).tolist(),
                peak_error_rad=np.max(np.abs(err), axis=0).tolist(),
                peak_measured_torque_nm=np.max(np.abs(torque), axis=0).tolist(),
                temperature_rise_c=(np.max(temp, axis=0)-temp[0]).tolist(),
                peak_temperatures_c=np.max(temp, axis=0).tolist(),
                hold_mean_error_rad=np.mean([r["error_rad"] for r in settled], axis=0).tolist() if settled else None,
                max_feedback_age_ms=max(r["feedback_age_ms"] for r in rows),
                model_fk_error_max_m=max(r["model_fk_error_m"] for r in rows),
                independent_cartesian_ground_truth=False)


def run(runtime, data):
    kp, kd = validate_request(data)
    if runtime.arm is not None or runtime.hand is not None:
        raise RuntimeError("Calibration starts only from released control")
    snapshot = snapshot_guard(runtime.read())
    initial_temp = np.asarray(snapshot["body"]["temperatures"])[15:29]
    if np.max(initial_temp[:, 0]) > 55 or np.max(initial_temp[:, 1]) > 55:
        raise RuntimeError("Cool both arms to at most 55 C before calibration")
    rise_limit=float(data.get('max_temperature_rise_c',8))
    experiment_guard(snapshot, initial_temp,max_temperature_rise_c=rise_limit)
    archive = runtime.archive_guard()
    sdk_age = archive["age_s"].get("rt/arm_sdk")
    if sdk_age is not None and sdk_age < 2:
        raise RuntimeError("Another arm SDK publisher is active")
    previous = json.loads(Path("/data/calibration.json").read_text())
    if previous.get("device") != "unitree-g1-nx-2.local" or not previous.get("commissioned"):
        raise ValueError("Robot identity and commissioned joint limits are required")
    config = harness_config(previous, snapshot)
    plant = PhysicalPlant(calibrated_scene(config, x=10, y=-10), None, None,
                          runtime.read, config, runtime.event, cancel=runtime.cancel)
    plant.sender = SimpleNamespace(ctrl_dual_hand=lambda *args: None)
    plant.sync(snapshot)
    directory = Path("/data/calibration-trials") / str(time.time_ns())
    directory.mkdir(parents=True)
    request = dict(data, model_output=str(directory / "model.mjb"))
    origin, goals, planned = plan(plant, request)
    runtime.event("calibration_plan", request=data, planning=planned, artifact=str(directory))
    if np.max(np.abs(np.asarray(runtime.read()["body"]["q"])[15:22] - origin)) > .02:
        raise RuntimeError("Arm changed during calibration planning; replan")
    rows = []
    result = dict(request=data, planning=planned, artifact=str(directory), completed=False,
                  physical_lift_verified=False, hand_commands_sent=0)
    (directory / "request.json").write_text(json.dumps(result, indent=2))
    began = time.monotonic()
    try:
        from g1_arm import G1_29_ArmController
        runtime.arm = G1_29_ArmController(motion_mode=True, simulation_mode=False,
            lowstate_hub=runtime.hub, record_lowcmd=runtime.command_written)
        runtime.arm.hold_arm_measured()
        runtime.arm.set_arm_motion_limits({"max_velocity_rad_s": .5 if data.get("kind")=="demo_replay" else .12, "max_acceleration_rad_s2": 2.0 if data.get("kind")=="demo_replay" else .5})
        runtime.arm.set_arm_position_limits(config["arm_limits_rad"])
        for i in range(14):
            runtime.arm.set_joint_gains(i, kp if i%7<4 else min(kp,40), kd if i%7<4 else min(kd,3))
        plant.adapter = runtime.arm
        runtime.plant = plant
        runtime.active = True
        runtime.trial_deadline = began + (110 if data.get("kind")=="demo_replay" else 35)
        runtime.phase = "HARNESS_CALIBRATION"
        runtime.event("calibration_start", **result, gains=runtime.arm.get_gains())
        prior = origin.copy()
        with (directory / "samples.jsonl").open("x", buffering=1) as output:
            for segment,duration,prior,goal,interpolate,hold_s in segments(origin,goals,data):
                segment_start = time.monotonic()
                while True:
                    tick = time.monotonic()
                    elapsed = tick-segment_start
                    alpha = min(1.0, elapsed/duration)
                    q = interpolate(min(elapsed,duration)) if interpolate else prior+(goal-prior)*(alpha*alpha*(3-2*alpha))
                    with plant.command_lock:
                        snap = plant.check()
                        measured = np.asarray(snap["body"]["q"])[15:22]
                        error = q-measured
                        fk, _, _ = model_pose(plant, measured)
                        target_fk, _, _ = model_pose(plant, q)
                        row = dict(unix_ns=time.time_ns(), monotonic_s=tick,
                            elapsed_s=tick-began, segment=f"{segment}-"+("hold" if alpha==1 else "move"),
                            commanded_q=q.tolist(), measured_q=measured.tolist(), error_rad=error.tolist(),
                            measured_dq=np.asarray(snap["body"]["dq"])[15:22].tolist(),
                            measured_tau_est_nm=np.asarray(snap["body"]["tau_est"])[15:22].tolist(),
                            temperatures_c=np.asarray(snap["body"]["temperatures"])[15:22].tolist(),
                            all_arm_temperatures_c=np.asarray(snap['body']['temperatures'])[15:29].tolist(),
                            all_arm_tau_est_nm=np.asarray(snap['body']['tau_est'])[15:29].tolist(),
                            all_arm_q=np.asarray(snap['body']['q'])[15:29].tolist(),
                            all_arm_dq=np.asarray(snap['body']['dq'])[15:29].tolist(),
                            hand_feedback_age_ms={side:(tick-snap[side]['time'])*1000 for side in ('left','right')},
                            feedback_age_ms=(tick-snap["body"]["time"])*1000,
                            encoder_fk_m=fk.tolist(), commanded_fk_m=target_fk.tolist(),
                            model_fk_error_m=float(np.linalg.norm(fk-target_fk)),
                            estimated_gravity_nm=model_pose(plant,q)[2].tolist(),
                            previous_feedforward_nm=plant.arm_tau_command.tolist(),
                            sdk_reference=runtime.arm.arm_reference_report(), gains=runtime.arm.get_gains())
                        # Persist the failing sample before checking additional experiment limits.
                        rows.append(row)
                        output.write(json.dumps(row, allow_nan=False)+"\n")
                        experiment_guard(snap, initial_temp,max_temperature_rise_c=rise_limit)
                        if np.max(np.abs(error)) > .15:
                            raise RuntimeError("Calibration tracking error exceeds 0.15 rad")
                        if tick > runtime.trial_deadline:
                            raise RuntimeError("Calibration deadline exceeded")
                        plant.sync(snap)
                        plant.check_live_geometry(left_q=q)
                        plant.left_arm = q.copy()
                        plant.hold_arm(snap)
                        runtime.event("calibration_command", sample_unix_ns=row["unix_ns"],
                                      requested_q=q.tolist(), feedforward_nm=plant.arm_tau_command.tolist())
                    if elapsed >= duration+hold_s:
                        break
                    time.sleep(max(0, PERIOD_S-(time.monotonic()-tick)))
                prior = goal
        result["completed"] = True
        result["returned_to_origin"] = bool(np.max(np.abs(np.asarray(runtime.read()["body"]["q"])[15:22]-origin)) < .04)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        runtime.event("calibration_error", error=result["error"])
        raise
    finally:
        result["metrics"] = metrics(rows)
        runtime.active = False
        # Empty hands + supported robot were attested. Do not leave a high-P
        # target latched after a fault; hold measured, then return to Unitree.
        if runtime.arm is not None:
            runtime.arm.hold_arm_measured()
            for i in range(14):
                runtime.arm.set_joint_gains(i, 20, 2)
            runtime.relax_owned_arms("empty_hand_harness_calibration_finished")
        result["control_released"] = runtime.arm is None
        (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        runtime.event("calibration_result", **result)
        runtime.demo_result = {"calibration": result, "physical_lift_verified": False}
    return runtime.demo_result
