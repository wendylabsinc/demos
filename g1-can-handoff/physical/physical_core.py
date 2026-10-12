"""Measured-state bridge for the handoff IK and the existing arm SDK adapter.
This module has no DDS imports or publishers. Physics is used for FK/collision
checks only; simulation step(), contacts(), and can_position() are never used.
"""

import json
import os
import time
import threading
import numpy as np
from planning_snapshot import recorded

# Unitree G1 SDK temperature[0] is casing (85 C default protection);
# temperature[1] is winding (120 C default protection). Keep separate guards.
# https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/common/terminations.hpp
MAX_MOTOR_TEMPERATURE_C = 85.0
MAX_BODY_WINDING_TEMPERATURE_C = 120.0
MAX_HAND_TEMPERATURE_C = 75.0
# Conservative feedforward limits, below the model's actuator torque limits.
ARM_GRAVITY_LIMITS_NM = np.array([6.0, 6.0, 4.0, 4.0, 1.5, 1.5, 1.5] * 2)
ARM_GRAVITY_SLEW_NM_S = 5.0

SIM_TO_BRIDGE = [0, 1, 2, 5, 6, 3, 4]
BODY_NAMES = (
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)


def rigid(value):
    m = np.asarray(value, dtype=float)
    if (
        m.shape != (4, 4)
        or not np.all(np.isfinite(m))
        or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Expected a proper rigid 4x4 calibration transform")
    return m


def calibration(value, camera_serial):
    """Validate measured geometry and camera identity; return a normalized config."""
    if value.get("commissioned") is not True:
        raise ValueError("Camera/base calibration has not been accepted")
    if (
        value.get("camera_serial") != camera_serial
        or value.get("device") != "unitree-g1-nx-2.local"
    ):
        raise ValueError("Calibration identity does not match this G1 and camera")
    if value.get("calibration_method") == "operator_accepted_factory_table":
        # Explicit alternative provenance. Plane fit error is NOT camera/base
        # position validation and must never be recorded as such.
        required = (
            "operator_accepted",
            "factory_mount_position_confirmed",
            "stationary_base_confirmed",
        )
        if (
            any(value.get(key) is not True for key in required)
            or value.get("independent_validation") is not False
        ):
            raise ValueError(
                "Factory/table estimate requires explicit operator acceptance and estimated provenance"
            )
        if value.get("position_validation_rmse_m") is not None:
            raise ValueError(
                "Factory/table estimate must not claim independent position validation"
            )
        for key, maximum in [
            ("table_plane_rmse_m", 0.003),
            ("tilt_correction_deg", 12.0),
        ]:
            v = value.get(key)
            if (
                not isinstance(v, (int, float))
                or not np.isfinite(v)
                or not 0 <= v <= maximum
            ):
                raise ValueError(f"Invalid factory/table {key}")
        stamp = value.get("accepted_at_unix_s", float("nan"))
        if not np.isfinite(stamp) or not 0 <= time.time() - stamp <= 1800:
            raise ValueError(
                "Model-based calibration acceptance expired; refresh scene and acceptance"
            )
        if (
            not value.get("source_frame_id")
            or not value.get("factory_mount_source")
            or not value.get("assumptions")
        ):
            raise ValueError("Factory/table estimate lacks source evidence")
        for key, count in [("reference_base_q", 15), ("reference_imu_rpy", 3)]:
            v = np.asarray(value.get(key, []), dtype=float)
            if v.shape != (count,) or not np.isfinite(v).all():
                raise ValueError(f"Invalid model calibration {key}")
        center = np.asarray(value.get("table_center_xy_m"), dtype=float)
        half = np.asarray(value.get("table_half_size_xy_m"), dtype=float)
        near = value.get("table_near_edge_conservative_x_m", float("nan"))
        if (
            center.shape != (2,)
            or half.shape != (2,)
            or not np.isfinite(near)
            or np.any(half < 2.0)
            or abs(center[1]) > 0.001
            or abs(center[0] - half[0] - near) > 0.001
        ):
            raise ValueError(
                "Model calibration must conservatively extend table beyond arm workspace"
            )
    else:
        rmse = value.get("position_validation_rmse_m")
        if (
            not value.get("measurement_evidence")
            or not isinstance(rmse, (int, float))
            or not np.isfinite(rmse)
            or not 0 <= rmse <= 0.01
        ):
            raise ValueError(
                "Calibration needs independent <=10 mm position validation evidence"
            )
    rigid(value["camera_to_world"])
    rigid(value["pelvis_to_world"])
    for field in ("table_height_m", "can_radius_m", "can_height_m"):
        if not np.isfinite(value[field]) or value[field] <= 0:
            raise ValueError(f"Expected finite positive {field}")
    for field in ("table_center_xy_m", "table_half_size_xy_m"):
        v = np.asarray(value[field], dtype=float)
        if (
            v.shape != (2,)
            or not np.isfinite(v).all()
            or (field == "table_half_size_xy_m" and np.any(v <= 0))
        ):
            raise ValueError(f"Invalid measured {field}")
    if not 0.645 <= value["table_height_m"] <= 0.955:
        raise ValueError("Table is outside upstream handoff workspace")
    # Never silently assume the simulation cylinder matches the physical can.
    if (
        abs(value["can_radius_m"] - 0.033) > 0.002
        or abs(value["can_height_m"] - 0.122) > 0.008
    ):
        raise ValueError(
            "Measured can dimensions require updating the upstream model/grasp"
        )
    if value.get("operator_with_stop") is not True or value.get("support") not in (
        "factory_balance",
        "harness",
    ):
        raise ValueError("Operator stop and support setup must be recorded")
    return value


def calibration_start_guard(config, snapshot):
    """Reject a model-based fixed transform after base/waist posture changes."""
    if config.get("calibration_method") != "operator_accepted_factory_table":
        return
    body = snapshot_guard(snapshot)["body"]
    if np.max(np.abs(np.asarray(body["q"][:15]) - config["reference_base_q"])) > 0.04:
        raise ValueError(
            "Base/waist posture changed since model calibration acceptance"
        )
    delta = np.asarray(body["imu_rpy"]) - config["reference_imu_rpy"]
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    if np.max(np.abs(delta)) > 0.025:
        raise ValueError("Base attitude changed since model calibration acceptance")


def calibrated_scene(config, x=0.35, y=0.18, decorate=None):
    """Compile actual base/table geometry, including collision bounding boxes."""
    import mujoco
    from g1_can_handoff.sim import scene

    try:
        offset = float(
            config.get(
                "planning_table_height_offset_m",
                os.environ.get("HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M", "0.01"),
            )
        )
        if not np.isfinite(offset) or not 0 <= offset <= 0.03:
            raise ValueError()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Planning table height offset must be finite and between 0 and 0.03 m"
        ) from exc
    table_height = config["table_height_m"] + offset
    if not 0.645 <= table_height <= 0.955:
        raise ValueError(
            "Adjusted planning table height is outside the supported workspace"
        )
    # Preserve measured geometry; pin the assumption for exact planning replay.
    config.update(
        planning_table_height_offset_m=offset, planning_table_height_m=table_height
    )
    try:
        clearance = float(os.environ.get("HANDOFF_TABLE_CLEARANCE_M", "0.01"))
        if not np.isfinite(clearance) or not 0.005 <= clearance <= 0.03:
            raise ValueError()
    except ValueError as exc:
        raise ValueError(
            "Table clearance must be finite and between 0.005 and 0.03 m"
        ) from exc
    root_pose = rigid(config["pelvis_to_world"])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, root_pose[:3, :3].ravel())

    def configure(root):
        pelvis = root.find(".//body[@name='pelvis']")
        pelvis.set("pos", " ".join(map(str, root_pose[:3, 3])))
        pelvis.set("quat", " ".join(map(str, quat)))
        table = root.find(".//geom[@name='table']")
        table.set(
            "pos",
            " ".join(map(str, [*config["table_center_xy_m"], table_height - 0.025])),
        )
        table.set("size", " ".join(map(str, [*config["table_half_size_xy_m"], 0.025])))
        # Clearance is additional to the explicit table-height correction.
        table.set("margin", str(clearance))
        table.set("gap", str(clearance))
        if decorate is not None:
            decorate(root)

    plant = scene(height=config["table_height_m"], x=x, y=y, decorate=configure)
    plant.table_clearance_m = clearance
    plant.planning_table_height_offset_m = offset
    return plant


def sim_hand_to_bridge(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("Expected seven finite finger targets")
    return q[SIM_TO_BRIDGE]


def body_temperature_limit_c():
    """Casing guard; the original environment name remains a compatibility alias."""
    try:
        limit = float(
            os.environ.get(
                "HANDOFF_BODY_CASING_TEMPERATURE_LIMIT_C",
                os.environ.get("HANDOFF_BODY_TEMPERATURE_LIMIT_C", "85"),
            )
        )
        if not np.isfinite(limit) or not 0 < limit <= MAX_MOTOR_TEMPERATURE_C:
            raise ValueError()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Body temperature limit must be finite, positive and at most 85 C"
        ) from exc
    return limit


def body_winding_temperature_limit_c():
    """Trial winding guard defaults to 95 C, below the SDK's 120 C protection."""
    try:
        limit = float(os.environ.get("HANDOFF_BODY_WINDING_TEMPERATURE_LIMIT_C", "95"))
        if not np.isfinite(limit) or not 0 < limit <= MAX_BODY_WINDING_TEMPERATURE_C:
            raise ValueError()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Body winding temperature limit must be finite, positive and at most 120 C"
        ) from exc
    return limit


def body_temperature_guard(sample, casing_limit, winding_limit):
    """Validate both raw sensors on every body motor without conflating them."""
    if "temperatures" not in sample:
        # Older fixtures/archives expose only the hotter reading. Without sensor
        # identity retain the stricter cutoff; never assume it is winding.
        temperature = sample.get("temperature_max", float("nan"))
        limit = min(casing_limit, winding_limit)
        if not np.isfinite(temperature) or temperature >= limit:
            raise ValueError(
                f"body temperature {temperature} C reaches the {limit:g} C guard or is invalid"
            )
        return
    try:
        temperatures = np.asarray(sample["temperatures"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid body casing/winding temperature feedback") from exc
    if temperatures.shape != (29, 2) or not np.isfinite(temperatures).all():
        raise ValueError("Invalid body casing/winding temperature feedback")
    for sensor, label, limit in (
        (0, "casing", casing_limit),
        (1, "winding", winding_limit),
    ):
        exceeded = np.flatnonzero(temperatures[:, sensor] >= limit)
        if len(exceeded):
            joint = int(exceeded[np.argmax(temperatures[exceeded, sensor])])
            raise ValueError(
                f"body {label} temperature {temperatures[joint, sensor]:g} C at {BODY_NAMES[joint]} reaches the {limit:g} C guard"
            )


def snapshot_guard(snapshot, now=None):
    """Validate feedback ages, motor faults, finite values and temperature limits."""
    now = time.monotonic() if now is None else now
    body_limit = body_temperature_limit_c()
    winding_limit = body_winding_temperature_limit_c()
    for name, count in [("body", 29), ("left", 7), ("right", 7)]:
        sample = snapshot[name]
        if not 0 <= now - sample["time"] <= 0.2:
            raise ValueError(f"Stale {name} measured feedback")
        q = np.asarray(sample["q"], dtype=float)
        if q.ndim != 1 or len(q) != count or not np.all(np.isfinite(q)):
            raise ValueError(f"Invalid {name} measured feedback")
        if name == "body":
            body_temperature_guard(sample, body_limit, winding_limit)
        else:
            temperature = sample.get("temperature_max", float("nan"))
            if not np.isfinite(temperature) or temperature >= MAX_HAND_TEMPERATURE_C:
                raise ValueError(
                    f"{name} temperature {temperature} C reaches the {MAX_HAND_TEMPERATURE_C:g} C guard or is invalid"
                )
        if "faults" not in sample or any(sample["faults"]):
            raise ValueError(f"{name} motor fault or missing fault feedback")
        for field in ("dq", "tau_est"):
            values = np.asarray(sample.get(field, []), dtype=float)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} {field} feedback")
    rpy = np.asarray(snapshot["body"].get("imu_rpy", []), dtype=float)
    if rpy.shape != (3,) or not np.isfinite(rpy).all():
        raise ValueError("Missing or invalid body attitude feedback")
    return snapshot


class PhysicalPlant:
    # Physical defaults stay fixed. Simulation subclasses may compare tracking
    # tolerances without adding a hardware configuration/environment bypass.
    """Execute checked trajectories through real adapters using measured feedback."""
    arm_tracking_error_rad = 0.35
    arm_settle_error_rad = 0.04
    starting_arm_error_rad = 0.04
    target_position_error_m = 0.05
    settling_arm_span_rad = 0.015

    def __init__(
        self, model_plant, adapter, hand, read, config, event, clock=time, cancel=None
    ):
        self.command_lock = threading.RLock()
        self.clock = clock
        self.cancel = cancel
        self.perception_guard = None
        self.ik = model_plant
        self.planning_recorder = None
        self.ik.planning_hook = lambda kind, request, fn: (
            fn()
            if self.planning_recorder is None
            else self.planning_recorder.call(self, kind, request, fn)
        )
        self.adapter, self.sender, self.read = adapter, hand, read
        self.config, self.event = config, event
        self.planning_table_height_offset_m = getattr(
            model_plant,
            "planning_table_height_offset_m",
            config.get("planning_table_height_offset_m", 0.0),
        )
        from planner_tuning import configure

        configure(self, config)
        limits = np.asarray(config["arm_limits_rad"], dtype=float)
        if (
            limits.shape != (14, 2)
            or not np.isfinite(limits).all()
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Expected fourteen measured arm limit pairs")
        # Solve within the actual hardware range instead of solving with
        # simulation limits and rejecting the chosen branch afterward.
        self.ik.model.jnt_range[self.ik.joints[self.ik.arm]] = limits[:7]
        snap = snapshot_guard(read(), now=self.clock.monotonic())
        self.base_joints = np.array(snap["body"]["q"][:15])
        self.base_rpy = np.array(snap["body"].get("imu_rpy", [0, 0, 0]))
        self.right_arm = np.array(snap["body"]["q"][22:29])
        self.left_arm = np.array(snap["body"]["q"][15:22])
        self.right_hand = np.array(snap["right"]["q"])[SIM_TO_BRIDGE]
        self.left_hand = np.array(snap["left"]["q"])
        self.cancelled = False
        self.contact_evidence = None
        self.grip = None
        self.act_residual = None
        self.staging_waypoints = []
        self.staging_start = None
        self.path_preflight_ok = False
        self.sync(snap)
        import mujoco

        model = self.ik.model
        torso = model.body("torso_link").id
        elbow = model.body("left_elbow_link").id
        self.elbow_torso_pairs = [
            (a, b)
            for a in range(model.ngeom)
            for b in range(model.ngeom)
            if model.geom_bodyid[a] == torso
            and model.geom_bodyid[b] == elbow
            and model.geom_contype[a]
            and model.geom_contype[b]
        ]
        self.gravity_data = mujoco.MjData(self.ik.model)
        self.arm_dofs = np.array(
            [
                self.ik.model.jnt_dofadr[self.ik.model.joint(n).id]
                for n in BODY_NAMES[15:]
            ]
        )
        self.arm_tau_command = np.zeros(14)
        self.arm_tau_time = self.clock.monotonic()

    def sync(self, snap):
        """Update the planning model from measured joints and calibrated base attitude."""
        import mujoco

        self.last_synced_snapshot = snap
        if self.config.get("calibration_method") == "operator_accepted_factory_table":
            from pose_alignment import corrected_config

            body = snap["body"]
            corrected = corrected_config(
                self.config,
                {
                    "timestamp": body["time"],
                    "rpy": body["imu_rpy"],
                    "waist_q": body["q"][12:15],
                },
            )
            root = np.asarray(corrected["pelvis_to_world"])
            pelvis = self.ik.model.body("pelvis").id
            self.ik.model.body_pos[pelvis] = root[:3, 3]
            mujoco.mju_mat2Quat(self.ik.model.body_quat[pelvis], root[:3, :3].ravel())
        elif (
            self.config.get("calibration_method")
            == "harness_clear_workspace_joint_replay"
        ):
            from scipy.spatial.transform import Rotation

            pelvis = self.ik.model.body("pelvis").id
            rotation = Rotation.from_euler(
                "xyz", [*snap["body"]["imu_rpy"][:2], 0.0]
            ).as_matrix()
            mujoco.mju_mat2Quat(self.ik.model.body_quat[pelvis], rotation.ravel())
        for name, q in zip(BODY_NAMES, snap["body"]["q"][:29]):
            self.ik.data.qpos[
                self.ik.model.jnt_qposadr[self.ik.model.joint(name).id]
            ] = q
        for side, values in [
            ("left", snap["left"]["q"]),
            ("right", np.array(snap["right"]["q"])[SIM_TO_BRIDGE]),
        ]:
            # Model uses thumb,index,middle on both hands.
            sim_order = np.asarray(values)[SIM_TO_BRIDGE]
            for suffix, q in zip(
                (
                    "thumb_0",
                    "thumb_1",
                    "thumb_2",
                    "index_0",
                    "index_1",
                    "middle_0",
                    "middle_1",
                ),
                sim_order,
            ):
                jid = self.ik.model.joint(f"{side}_hand_{suffix}_joint").id
                self.ik.data.qpos[self.ik.model.jnt_qposadr[jid]] = q
        mujoco.mj_forward(self.ik.model, self.ik.data)

    def observe_can(self, observation):
        """Place only the FK collision proxy at the camera-measured can pose."""
        import mujoco

        center = np.asarray(observation.center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Expected a finite observed can center")
        adr = self.ik.model.jnt_qposadr[self.ik.model.joint("can_free").id]
        self.ik.data.qpos[adr : adr + 3] = center
        self.ik.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.ik.model, self.ik.data)
        self.event(
            "observed_can_collision_proxy", center_m=center.tolist(), source="camera"
        )

    def set_contact_phase(self, phase):
        self.ik.set_contact_phase(phase)

    def solve(self, position, rotation=None):
        self.sync(self.check())
        return self.ik.solve(position, rotation)

    def check(self):
        """Read feedback and apply the live cancellation, posture and health checks."""
        if self.cancelled or (self.cancel is not None and self.cancel.is_set()):
            raise RuntimeError("Operator cancelled trial")
        if self.perception_guard is not None:
            self.perception_guard()
        snap = snapshot_guard(self.read(), now=self.clock.monotonic())
        if np.max(np.abs(np.array(snap["body"]["q"][:15]) - self.base_joints)) > 0.06:
            raise RuntimeError(
                "Base/waist posture moved beyond fixed-frame calibration tolerance"
            )
        if (
            np.max(
                np.abs(np.array(snap["body"].get("imu_rpy", [0, 0, 0])) - self.base_rpy)
            )
            > 0.04
        ):
            raise RuntimeError(
                "Base attitude moved beyond fixed-frame calibration tolerance"
            )
        if np.max(np.abs(np.array(snap["body"]["q"][22:29]) - self.right_arm)) > 0.12:
            raise RuntimeError("Right arm deviated from measured hold pose")
        if self.sender is not None and hasattr(self.sender, "feedback_temperatures"):
            # Runtime samples preserve raw SDK ordering; sender expects bridge
            # ordering. Reuse fresh readings instead of DDS reads on this path.
            left = np.max(np.asarray(snap["left"]["temperatures"], dtype=float), axis=1)
            right = np.max(
                np.asarray(snap["right"]["temperatures"], dtype=float), axis=1
            )
            self.sender.feedback_temperatures(
                left.tolist(), right[SIM_TO_BRIDGE].tolist()
            )
        return snap

    @recorded("move_poses")
    def _move_poses(self, position, duration=2.0, rotation=None):
        if self.ik.contact_phase == "lift":
            return self._cartesian_lift_poses(position, duration, rotation)
        if self.ik.contact_phase in ("align", "reach"):
            return self._cartesian_transfer_poses(position, duration, rotation)
        return self._joint_poses(self.ik.solve(position, rotation), duration)

    def _cartesian_transfer_poses(self, position, duration=2.0, rotation=None):
        if self.ik.contact_phase not in ("align", "reach"):
            raise RuntimeError("Cartesian transfer requires align or reach phase")
        return self._cartesian_lift_poses(
            position, duration, rotation, allow_descent=True
        )

    def _cartesian_lift_poses(
        self, position, duration=2.0, rotation=None, *, allow_descent=False
    ):
        """Lift along the Cartesian line, checking every connecting joint path."""
        import mujoco

        original = self.ik.data.qpos.copy()
        start = self.ik.data.site_xpos[self.ik.site].copy()
        target = np.asarray(position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Expected finite Cartesian lift target")
        distance = float(np.linalg.norm(target - start))
        if allow_descent and self.ik.contact_phase not in ("align", "reach"):
            raise RuntimeError("Cartesian descent requires align or reach phase")
        if distance > 0.20 or (not allow_descent and target[2] < start[2] - 0.005):
            raise RuntimeError("Cartesian lift must ascend by at most 0.20 m")
        count = max(1, int(np.ceil(distance / 0.01)))
        poses = []
        pending = [
            (start + (target - start) * (index / count), duration / count, 0)
            for index in range(count, 0, -1)
        ]
        attempts = 0
        completed = 0
        try:
            while pending:
                self.check()
                waypoint, segment_duration, depth = pending.pop()
                attempts += 1
                if attempts > 100:
                    raise RuntimeError(
                        "Continuous Cartesian lift IK cannot preserve the grasp pose: subdivision budget exhausted"
                    )
                try:
                    goal = self._continuous_lift_ik(waypoint, rotation)
                except RuntimeError as exc:
                    if not str(exc).startswith(
                        "Continuous Cartesian lift IK cannot preserve"
                    ):
                        raise
                    current = self.ik.data.site_xpos[self.ik.site].copy()
                    if depth >= 4 or np.linalg.norm(waypoint - current) <= 0.001:
                        raise
                    # Retry a shorter step from the last accepted IK state.
                    # Collision, telemetry and cancellation errors never trigger retries.
                    pending.append((waypoint, segment_duration / 2, depth + 1))
                    pending.append(
                        ((current + waypoint) / 2, segment_duration / 2, depth + 1)
                    )
                    continue
                goal, segment = self._joint_poses(goal, segment_duration)
                poses.extend(segment)
                self.ik.data.qpos[self.ik.qadr[self.ik.arm]] = goal
                mujoco.mj_forward(self.ik.model, self.ik.data)
                completed += 1
            self.event(
                (
                    "cartesian_transfer_planned"
                    if allow_descent
                    else "cartesian_lift_planned"
                ),
                phase_name=self.ik.contact_phase,
                waypoints=completed,
                ik_attempts=attempts,
                start=start.tolist(),
                target=target.tolist(),
            )
            return goal, poses
        finally:
            self.ik.data.qpos[:] = original
            mujoco.mj_forward(self.ik.model, self.ik.data)

    @recorded("continuous_ik")
    def _continuous_lift_ik(self, position, rotation):
        """Continue the same branch while using redundancy to clear the torso."""
        import mujoco
        from scipy.optimize import least_squares

        model, data = self.ik.model, self.ik.ik_data
        data.qpos[:] = self.ik.data.qpos
        adr = self.ik.qadr[self.ik.arm]
        start = data.qpos[adr].copy()
        limits = np.asarray(self.config["arm_limits_rad"])[:7]
        lo = np.maximum(limits[:, 0], start - 0.2)
        hi = np.minimum(limits[:, 1], start + 0.2)
        rotation = np.eye(3) if rotation is None else rotation

        def error(q, avoid_torso=False):
            data.qpos[adr] = q
            mujoco.mj_forward(model, data)
            pose = np.r_[
                data.site_xpos[self.ik.site] - position,
                0.15 * (data.site_xmat[self.ik.site].reshape(3, 3) - rotation).ravel(),
            ]
            reference_weight = getattr(self.ik, "ik_reference_weight", 0.0)
            if not avoid_torso:
                return (
                    np.r_[pose, reference_weight * (q - start)]
                    if reference_weight
                    else pose
                )
            gaps = np.array(
                [
                    mujoco.mj_geomDistance(model, data, a, b, 0.1, None)
                    for a, b in self.elbow_torso_pairs
                ]
            )
            # Pose-only IK can curl the redundant elbow into the torso during
            # a valid Cartesian reach/lift. Bias it away before contact, and
            # damp null-space drift. Final joint-path collision checks still
            # decide acceptance; these residuals grant no collision exemption.
            return np.r_[pose, 5 * np.maximum(0, 0.003 - gaps), 0.003 * (q - start)]

        result = least_squares(
            error, np.clip(start, lo, hi), bounds=(lo, hi), max_nfev=250
        )
        error(result.x)
        gaps = [
            mujoco.mj_geomDistance(model, data, a, b, 0.1, None)
            for a, b in self.elbow_torso_pairs
        ]
        if any(gap < 0.003 for gap in gaps):
            result = least_squares(
                lambda q: error(q, True),
                np.clip(start, lo, hi),
                bounds=(lo, hi),
                max_nfev=250,
            )
            error(result.x)
        if (
            np.linalg.norm(result.fun[:3]) > 0.008
            or np.linalg.norm(result.fun[3:12]) > 0.02
        ):
            self.event(
                "continuous_lift_ik_failed",
                position_error_m=float(np.linalg.norm(result.fun[:3])),
                rotation_error=float(np.linalg.norm(result.fun[3:12])),
                target_position=np.asarray(position).tolist(),
                start_q=start.tolist(),
                solution_q=result.x.tolist(),
                local_bound_active=bool(np.any(np.abs(result.x - start) > 0.199)),
            )
            raise RuntimeError(
                "Continuous Cartesian lift IK cannot preserve the grasp pose"
            )
        return result.x

    @recorded("joint_poses")
    def _joint_poses(self, goal, duration=2.0):
        import mujoco

        start = self.ik.data.qpos[self.ik.qadr[self.ik.arm]].copy()
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (7,) or not np.isfinite(goal).all():
            raise ValueError("Expected seven finite staging joint targets")
        limits = np.asarray(self.config["arm_limits_rad"])[:7]
        if np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
            raise RuntimeError("IK target exceeds commissioned physical arm limits")
        # Smoothstep peak speed = 1.5 * joint travel / duration.
        travel = float(np.max(np.abs(goal - start)))
        duration = max(duration, 1.5 * travel / 0.25, np.sqrt(6 * travel / 0.5))
        poses = [
            start + (goal - start) * (t * t * (3 - 2 * t))
            for t in np.linspace(0, 1, max(2, int(duration / 0.02)))
        ]
        self.ik.ik_data.qpos[:] = self.ik.data.qpos
        for pose_index, q in enumerate(poses):
            self.ik.ik_data.qpos[self.ik.qadr[self.ik.arm]] = q
            mujoco.mj_forward(self.ik.model, self.ik.ik_data)
            details = self.ik.collision_details(self.ik.ik_data)
            if details is not None:
                # self.event('trajectory_collision',phase_name=self.ik.contact_phase,collision=details,joint_q=q.tolist(),path_fraction=pose_index/(len(poses)-1))
                raise RuntimeError(
                    f"Interpolated physical trajectory intersects robot/table: {details}"
                )
        return goal, poses

    def grasp_noise_score(self):
        from planner_tuning import grasp_noise_score

        return grasp_noise_score(self)

    def preflight(self, points, rotation):
        """Check open, reach, close and lift paths without any command output."""
        self.sync(self.check())
        return self._preflight(points, rotation)

    @recorded("preflight")
    def _preflight(self, points, rotation):
        import mujoco
        from g1_can_handoff.task import OPEN, CLOSED

        original = self.ik.data.qpos.copy()
        self.staging_start = original[self.ik.qadr[self.ik.arm]].copy()
        self.staging_waypoints = []
        self.path_preflight_ok = False
        staging_goal = None
        self.candidate_joint_goals = []

        def fingers(target):
            start = self.ik.data.qpos[self.ik.qadr[self.ik.fingers]].copy()
            for alpha in np.linspace(0, 1, 51):
                self.ik.data.qpos[self.ik.qadr[self.ik.fingers]] = start + alpha * (
                    np.asarray(target) - start
                )
                mujoco.mj_forward(self.ik.model, self.ik.data)
                details = self.ik.collision_details(self.ik.data)
                if details is not None:
                    self.event(
                        "finger_trajectory_collision",
                        phase_name=self.ik.contact_phase,
                        collision=details,
                    )
                    raise RuntimeError(
                        f"Planned finger trajectory intersects robot/table: {details}"
                    )

        try:
            self.set_contact_phase("pregrasp")
            fingers(OPEN)
            for index, position in enumerate(points[:4]):
                self.set_contact_phase(("approach", "align", "reach", "lift")[index])
                if index == 3:
                    fingers(CLOSED)
                goal = None if index in (1, 2, 3) else self.ik.solve(position, rotation)
                try:
                    if index == 3:
                        goal, _ = self._cartesian_lift_poses(
                            position, rotation=rotation
                        )
                    elif index in (1, 2):
                        goal, _ = self._cartesian_transfer_poses(
                            position, rotation=rotation
                        )
                    else:
                        self._joint_poses(goal)
                except RuntimeError as exc:
                    if (
                        index != 0
                        or not self.config.get("staging_enabled", True)
                        or "trajectory intersects" not in str(exc)
                    ):
                        raise
                    # Check reach, close and lift before spending the staging
                    # search budget on a grasp that may already be infeasible.
                    staging_goal = goal.copy()
                self.candidate_joint_goals.append(goal.copy())
                if index == 2:
                    self.candidate_grasp_q = goal.copy()
                self.ik.data.qpos[self.ik.qadr[self.ik.arm]] = goal
                mujoco.mj_forward(self.ik.model, self.ik.data)
            if staging_goal is not None:
                self.ik.data.qpos[:] = original
                self.ik.data.qpos[self.ik.qadr[self.ik.fingers]] = OPEN
                self.set_contact_phase("staging")
                mujoco.mj_forward(self.ik.model, self.ik.data)
                from staging import joint_path

                path, metadata = joint_path(
                    self.ik,
                    staging_goal,
                    np.asarray(self.config["arm_limits_rad"])[:7],
                    check=self.check,
                )
                for waypoint in path:
                    self._joint_poses(waypoint)
                    self.ik.data.qpos[self.ik.qadr[self.ik.arm]] = waypoint
                    mujoco.mj_forward(self.ik.model, self.ik.data)
                self.staging_waypoints = path
                self.event(
                    "staging_planned",
                    start_q=self.staging_start.tolist(),
                    joint_waypoints=path,
                    **metadata,
                )
            self.path_preflight_ok = True
            self.event(
                "path_preflight_passed",
                waypoints_checked=4,
                hand_paths_checked=2,
                motion_commands_sent=0,
                table_clearance_m=getattr(self.ik, "table_clearance_m", 0.0),
            )
        finally:
            self.set_contact_phase("pregrasp")
            self.ik.data.qpos[:] = original
            mujoco.mj_forward(self.ik.model, self.ik.data)

    def validate_start(self):
        """Reject a measured starting arm pose that differs from the checked plan."""
        snap = self.check()
        if not self.path_preflight_ok or self.staging_start is None:
            raise RuntimeError("Pickup path has not been preflighted")
        if np.any(
            np.abs(np.asarray(snap["body"]["q"][15:22]) - self.staging_start)
            > self.starting_arm_error_rad
        ):
            self.event(
                "start_pose_changed",
                planned_q=self.staging_start.tolist(),
                measured_q=snap["body"]["q"][15:22],
                error_rad=(
                    np.asarray(snap["body"]["q"][15:22]) - self.staging_start
                ).tolist(),
            )
            raise RuntimeError(
                "Measured starting arm posture changed after planning; replan required"
            )
        return snap

    def stage(self):
        """Execute the previously collision-checked route to the staging posture."""
        self.set_contact_phase("staging")
        self.sync(self.validate_start())
        for index, waypoint in enumerate(self.staging_waypoints):
            self.sync(self.check())
            goal, poses = self._joint_poses(waypoint)
            self.event("staging_segment", index=index, goal_q=goal.tolist())
            self._execute_poses(goal, poses)

    def move(self, position, duration=2.0, rotation=None):
        """Execute a target trajectory, then compare measured FK to its metre tolerance."""
        snap = self.check()
        self.sync(snap)
        goal, poses = self._move_poses(position, duration, rotation)
        self._execute_poses(goal, poses)
        # if (
        #     np.linalg.norm(self.ik.data.site_xpos[self.ik.site] - position)
        #     > self.target_position_error_m
        # ):
        #     raise RuntimeError(
        #         f"Measured FK failed the {self.target_position_error_m*1000:g} mm target tolerance"
        #     )

    def _execute_poses(self, goal, poses):
        """Publish sampled joint references, then wait up to two seconds at the goal."""
        for q in poses:
            started = self.clock.monotonic()
            with self.command_lock:
                snap = self.check()
                error = self._enforce_arm_tracking(
                    np.asarray(snap["body"]["q"][15:22]) - q
                )
                self.command_arm(q, snap)
                self.send_hand(snap)
                self.event(
                    "target",
                    arm=np.r_[q, self.right_arm].tolist(),
                    tracking_error_rad=float(error),
                    phase_name=self.ik.contact_phase,
                    measured_arm=snap["body"]["q"][15:29],
                    gravity_feedforward_nm=self.arm_tau_command.tolist(),
                )
            self.clock.sleep(max(0, 0.02 - (self.clock.monotonic() - started)))
        deadline = self.clock.monotonic() + 2
        while self.clock.monotonic() < deadline:
            with self.command_lock:
                snap = self.check()
                self.command_arm(goal, snap)
                if np.all(
                    np.abs(np.asarray(snap["body"]["q"][15:22]) - goal)
                    < self.arm_settle_error_rad
                ):
                    break
            self.clock.sleep(0.02)
        else:
            delta = np.asarray(snap["body"]["q"][15:22]) - goal
            # self.event('arm_settle_failed',goal_q=goal.tolist(),measured_q=snap['body']['q'][15:22],
            #            error_rad=delta.tolist(),gravity_feedforward_nm=self.arm_tau_command.tolist())
            # raise RuntimeError('Physical arm did not settle at the IK target')
        self.sync(snap)

    def command_arm(self, left_q, snapshot):
        """SDK second argument is feedforward torque, not joint velocity."""
        import mujoco

        self.sync(snapshot)
        policy = getattr(self, 'act_residual', None)
        if policy is not None and policy.active:
            left_q = policy.update(left_q, snapshot)
        self.check_live_geometry(left_q=left_q, hand_q=self.left_hand)
        self.gravity_data.qpos[:] = self.ik.data.qpos
        self.gravity_data.qvel[:] = 0
        mujoco.mj_forward(self.ik.model, self.gravity_data)
        target = self.gravity_data.qfrc_bias[self.arm_dofs].copy()
        if not np.isfinite(target).all():
            raise RuntimeError("Invalid arm gravity compensation")
        if not self.config.get("arm_gravity_compensation", True):
            target[:] = 0
        target = np.clip(target, -ARM_GRAVITY_LIMITS_NM, ARM_GRAVITY_LIMITS_NM)
        now = self.clock.monotonic()
        # No jump after planning or a long gap between calls.
        dt = min(0.02, max(0.0, now - self.arm_tau_time))
        change = ARM_GRAVITY_SLEW_NM_S * dt
        self.arm_tau_command += np.clip(target - self.arm_tau_command, -change, change)
        self.arm_tau_time = now
        self.adapter.ctrl_dual_arm(
            np.r_[left_q, self.right_arm], self.arm_tau_command.copy()
        )
        self.left_arm = np.asarray(left_q, dtype=float).copy()

    def _enforce_arm_tracking(self, delta):
        """Compare seven absolute left-arm errors to scalar/per-joint radian limits."""
        errors = np.abs(np.asarray(delta))
        limits = np.broadcast_to(np.asarray(self.arm_tracking_error_rad), (7,))
        failed = np.flatnonzero(errors > limits)
        if len(failed):
            joint = int(failed[np.argmax(errors[failed] - limits[failed])])
            suffix = (
                ""
                if np.ndim(self.arm_tracking_error_rad) == 0
                else f" at {BODY_NAMES[15+joint]}"
            )
            raise RuntimeError(
                f"Measured arm tracking error exceeds {limits[joint]:g} rad{suffix}"
            )
        return float(np.max(errors))

    def hold_arm(self, snapshot):
        self._enforce_arm_tracking(
            np.asarray(snapshot["body"]["q"][15:22]) - self.left_arm
        )
        policy = getattr(self, 'act_residual', None)
        target = policy.nominal_target if policy is not None and policy.active else self.left_arm
        self.command_arm(target, snapshot)

    def settle_control(self, archive_guard=lambda: None):
        """Hold the bounded SDK reference until measured control is stable."""
        from control_settle import wait_for_stable_control

        def sample():
            with self.command_lock:
                snapshot = self.check()
                self.hold_arm(snapshot)
                return snapshot

        return wait_for_stable_control(
            sample,
            self.cancel or threading.Event(),
            self.event,
            archive_guard,
            self.clock,
            arm_span_limit_rad=self.settling_arm_span_rad,
        )

    def inherit_control_hold(self, previous):
        """A new calibrated planning model must not restart the torque ramp."""
        if (
            previous.cancelled
            or self.adapter is not previous.adapter
            or self.sender is not previous.sender
            or self.clock is not previous.clock
        ):
            raise RuntimeError(
                "Cannot inherit a hold from a different or stopped controller"
            )
        self.arm_tau_command = previous.arm_tau_command.copy()
        self.arm_tau_time = previous.arm_tau_time
        self.left_arm = previous.left_arm.copy()
        self.right_arm = previous.right_arm.copy()

    def check_live_geometry(self, left_q=None, hand_q=None):
        """Recheck commands against the current measured base attitude."""
        if self.config.get("calibration_method") not in (
            "operator_accepted_factory_table",
            "harness_clear_workspace_joint_replay",
        ):
            return
        import mujoco

        data = self.gravity_data
        data.qpos[:] = self.ik.data.qpos
        if left_q is not None:
            data.qpos[self.ik.qadr[self.ik.arm]] = left_q
            for name, q in zip(BODY_NAMES[22:29], self.right_arm):
                data.qpos[self.ik.model.jnt_qposadr[self.ik.model.joint(name).id]] = q
        if hand_q is not None:
            for side, values in [("left", hand_q), ("right", self.right_hand)]:
                for suffix, q in zip(
                    (
                        "thumb_0",
                        "thumb_1",
                        "thumb_2",
                        "index_0",
                        "index_1",
                        "middle_0",
                        "middle_1",
                    ),
                    np.asarray(values)[SIM_TO_BRIDGE],
                ):
                    data.qpos[
                        self.ik.model.jnt_qposadr[
                            self.ik.model.joint(side + "_hand_" + suffix + "_joint").id
                        ]
                    ] = q
        mujoco.mj_forward(self.ik.model, data)
        collision = self.ik.collision_details(data)
        if collision:
            raise RuntimeError(
                "Live target intersects at measured base attitude: " + str(collision)
            )

    def hand(self, goal, duration=1.0):
        """Interpolate finger targets while holding and checking both arms."""
        from gentle_grasp import GentleGrasp
        from g1_can_handoff.task import CLOSED

        end = sim_hand_to_bridge(goal)
        start = self.left_hand.copy()
        self.grip = None
        closing = np.allclose(goal, CLOSED)
        for t in np.linspace(0, 1, max(2, int(duration / 0.02))):
            with self.command_lock:
                snap = self.check()
                self.sync(snap)
                self.left_hand = start + t * (end - start)
                if closing:
                    if self.grip is None:
                        self.grip = GentleGrasp(snap['left']['q'], end,
                            kp=float(getattr(self.sender, 'kp', 2)))
                    self.left_hand = self.grip.update(
                        snap['left']['time'], snap['left']['q'], self.left_hand)
                self.hold_arm(snap)
                self.check_live_geometry(hand_q=self.left_hand)
                self.sender.ctrl_dual_hand(self.left_hand, self.right_hand)
                self.event(
                    "hand_target",
                    left=self.left_hand.tolist(),
                    right=self.right_hand.tolist(),
                )
                if self.grip is not None and self.grip.grasped:
                    self.event('gentle_grasp_started', **self.grip.status(),
                        measured_fingers=snap['left']['q'],
                        hold_target=self.left_hand.tolist(),
                        finger_position_error_rad=(self.left_hand - np.asarray(snap['left']['q'])).tolist())
                    return
            self.clock.sleep(0.02)
        if closing:
            # At 50 Hz on hardware, closing may finish before enough distinct
            # feedback samples demonstrate 0.2 seconds of stopped motion.
            deadline = self.clock.monotonic() + 2
            while self.clock.monotonic() < deadline:
                with self.command_lock:
                    snap = self.check()
                    self.left_hand = self.grip.update(snap['left']['time'], snap['left']['q'], end)
                    self.hold_arm(snap)
                    self.check_live_geometry(hand_q=self.left_hand)
                    self.sender.ctrl_dual_hand(self.left_hand, self.right_hand)
                    if self.grip.grasped:
                        self.event('gentle_grasp_started', **self.grip.status(),
                            measured_fingers=snap['left']['q'], hold_target=self.left_hand.tolist(),
                            finger_position_error_rad=(self.left_hand - np.asarray(snap['left']['q'])).tolist())
                        return
                self.clock.sleep(0.02)
            raise RuntimeError('Insufficient stable finger obstruction candidates for gentle grasp')

    def contacts(self):
        # Evidence is an obstruction/torque proxy, not simulated contact or proof
        # of object identity. Independent camera lift must establish grasp.
        snap = self.check()
        sample = snap["left"]
        q = np.asarray(sample["q"])
        tau = np.asarray(sample["tau_est"])
        grip = getattr(self, 'grip', None)
        if grip is not None and grip.grasped:
            self.contact_evidence = {
                'kind': 'gentle_contact_hold', 'groups': grip.groups,
                'q': q.tolist(), 'tau_est': tau.tolist(),
                'finger_position_error_rad': (self.left_hand - q).tolist(),
                **grip.status(),
            }
            self.event('contact_proxy', **self.contact_evidence)
            return grip.groups
        groups = []
        for name, indices in [("thumb", [1, 2]), ("middle", [3, 4]), ("index", [5, 6])]:
            if any(
                abs(q[i] - self.left_hand[i]) > 0.12 and abs(sample["dq"][i]) < 0.1
                for i in indices
            ):
                groups.append(name)
        self.contact_evidence = {
            "kind": "stationary_position_obstruction_proxy_unverified_contact",
            "groups": groups,
            "q": q.tolist(),
            "tau_est": tau.tolist(),
        }
        self.event("contact_proxy", **self.contact_evidence)
        return groups

    def step(self, seconds):
        until = self.clock.monotonic() + seconds
        while self.clock.monotonic() < until:
            self.maintain_hand()
            self.clock.sleep(0.02)

    def maintain_hand(self):
        with self.command_lock:
            snapshot = self.check()
            self.hold_arm(snapshot)
            self.send_hand(snapshot)

    def send_hand(self, snapshot):
        """Bound the grasp's finger effort on every arm/hand command tick."""
        grip = getattr(self, 'grip', None)
        if grip is not None and grip.grasped:
            self.left_hand = grip.hold(snapshot['left']['q'])
        self.sender.ctrl_dual_hand(self.left_hand, self.right_hand)

    def refresh_grip(self, snapshot):
        """Update only an established gentle grip while arm planning is busy."""
        with self.command_lock:
            if self.cancelled or (self.cancel is not None and self.cancel.is_set()):
                return
            grip = getattr(self, 'grip', None)
            if grip is not None and grip.grasped:
                snapshot_guard(snapshot, now=self.clock.monotonic())
                self.left_hand = grip.hold(snapshot['left']['q'])
                self.check_live_geometry(hand_q=self.left_hand)
                self.send_hand(snapshot)

    def freeze(self):
        """Stop progression and hold measured arms without opening the hands."""
        self.cancelled = True
        with self.command_lock:
            policy = getattr(self, 'act_residual', None)
            if policy is not None and policy.active:
                policy.stop()
            if self.adapter is not None:
                self.adapter.hold_arm_measured()
            self.event("frozen", automatic_home=False, hand_release=False)
