"""CUDA-only path planning: one batch checks every sample of one prepared path.

No command publishers, DDS calls, CPU IK or automatic CPU fallback live here.
Initialization/JIT must finish before timing a ready planner. CPU device access
is exposed only by the explicitly named test factory.
"""

import math
import time
import copy
import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjw
from gpu_kinematics import ArmKinematics, Vec7

PHASES = (
    "pregrasp",
    "staging",
    "approach",
    "align",
    "reach",
    "close",
    "lift",
    "hold",
    "offer",
)


def warmup_cuda(ik=None):
    """Compile/load the G1 CUDA kernels while passive; never acquire ownership."""
    from g1_can_handoff.sim import scene, Plant

    if ik is None:
        ik = scene(x=-10, y=-10)
    else:
        source = ik
        ik = Plant(copy.copy(source.model))
        ik.data.qpos[:] = source.data.qpos
        can = ik.model.jnt_qposadr[ik.model.joint("can_free").id]
        ik.data.qpos[can : can + 2] = [-10, -10]
        mujoco.mj_kinematics(ik.model, ik.data)
    ik.table_clearance_m = 0.01
    ik.model.geom("table").margin = 0.01
    ik.model.geom("table").gap = 0.01
    started = time.monotonic()
    engine = CudaPlanner(ik)
    q = ik.data.qpos[ik.qadr[ik.arm]].copy()
    fingers = ik.data.qpos[ik.qadr[ik.fingers]].copy()
    p, r = engine.arm.forward(q)
    limits = ik.model.jnt_range[ik.joints[ik.arm]]
    engine.arm.solve(p[0], r[0], q, limits, local=True)
    segment = dict(
        phase="pregrasp",
        qpos=ik.data.qpos.copy(),
        start=q,
        goal=q,
        finger_start=fingers,
        finger_goal=fingers,
        count=4096,
    )
    engine.validate([segment], limits)
    return dict(
        device=str(engine.device),
        warmup_s=time.monotonic() - started,
        motion_commands_sent=0,
    )


def contact_thresholds(ik):
    """Same phase exemptions and strict inequalities as Plant.collision_details."""
    model = ik.model
    bodies = [model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)]
    geoms = [model.geom(i).name for i in range(model.ngeom)]
    thresholds = np.full((len(PHASES), model.ngeom, model.ngeom), -np.inf, np.float32)
    for phase, name in enumerate(PHASES):
        for a in range(model.ngeom):
            for b in range(model.ngeom):
                names = (bodies[a], bodies[b])
                if "can" in names:
                    other = names[1] if names[0] == "can" else names[0]
                    if other == "world" or (
                        name in ("reach", "close", "lift", "hold", "offer")
                        and any(
                            other.startswith("left_hand_" + f)
                            for f in ("thumb", "index", "middle")
                        )
                    ):
                        continue
                    thresholds[phase, a, b] = (
                        0.01
                        if name in ("pregrasp", "staging", "approach", "align")
                        else 0.0
                    )
                    continue
                if "table" in (geoms[a], geoms[b]) and any(
                    n.startswith("left_") for n in names
                ):
                    thresholds[phase, a, b] = max(
                        -0.001, getattr(ik, "table_clearance_m", 0.0)
                    )
                    continue
                if all("left_hand" in n or n == "left_wrist_yaw_link" for n in names):
                    continue
                if any(n.startswith("left_") for n in names) and any(
                    n in ("world", "torso_link", "pelvis") or n.startswith("right_")
                    for n in names
                ):
                    thresholds[phase, a, b] = -0.001
    return thresholds


@wp.kernel
def sample_kernel(
    backdrop: wp.array2d(dtype=float),
    starts: wp.array(dtype=Vec7),
    goals: wp.array(dtype=Vec7),
    arm_adr: wp.array(dtype=int),
    finger_adr: wp.array(dtype=int),
    finger_starts: wp.array(dtype=Vec7),
    finger_goals: wp.array(dtype=Vec7),
    segment: wp.array(dtype=int),
    fractions: wp.array(dtype=float),
    qpos: wp.array2d(dtype=float),
    arm_commands: wp.array(dtype=Vec7),
):
    i = wp.tid()
    s = segment[i]
    t = fractions[i]
    alpha = t * t * (3.0 - 2.0 * t)
    for j in range(qpos.shape[1]):
        qpos[i, j] = backdrop[s, j]
    command = Vec7()
    for j in range(7):
        command[j] = starts[s][j] + alpha * (goals[s][j] - starts[s][j])
        qpos[i, arm_adr[j]] = command[j]
        qpos[i, finger_adr[j]] = finger_starts[s][j] + t * (
            finger_goals[s][j] - finger_starts[s][j]
        )
    arm_commands[i] = command


@wp.kernel
def joints_kernel(
    q: wp.array(dtype=Vec7),
    count: int,
    lo: Vec7,
    hi: Vec7,
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    if i < count:
        for j in range(7):
            value = q[i][j]
            if (
                not wp.isfinite(value)
                or value < lo[j] - 1.0e-6
                or value > hi[j] + 1.0e-6
            ):
                wp.atomic_or(invalid, i, 1)
            if i > 0 and wp.abs(value - q[i - 1][j]) > 0.006:
                # 50Hz commands, existing 0.25rad/s smoothstep speed limit.
                wp.atomic_or(invalid, i, 2)


@wp.kernel
def contacts_kernel(
    nacon: wp.array(dtype=int),
    worldid: wp.array(dtype=int),
    geoms: wp.array(dtype=wp.vec2i),
    distances: wp.array(dtype=float),
    phases: wp.array(dtype=int),
    thresholds: wp.array3d(dtype=float),
    active: int,
    invalid: wp.array(dtype=int),
):
    i = wp.tid()
    if i < nacon[0]:
        world = worldid[i]
        if world < active:
            a, b = geoms[i][0], geoms[i][1]
            distance = distances[i]
            if not wp.isfinite(distance) or distance < thresholds[phases[world], a, b]:
                wp.atomic_or(invalid, world, 4)


class CudaPlanner:
    def __init__(self, ik):
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError(
                "CUDA planning requested but no CUDA GPU is available; CPU fallback is disabled"
            )
        self._initialize(ik, wp.get_device("cuda:0"))

    @classmethod
    def for_cpu_tests(cls, ik):
        instance = cls.__new__(cls)
        instance._initialize(ik, wp.get_device("cpu"))
        return instance

    def _initialize(self, ik, device):
        self.ik, self.device = ik, device
        self.arm = ArmKinematics(ik, device)
        # Warp's multi-contact manifold rejects the nonzero clearance margins
        # in this model. Planning needs the nearest separating distance, not
        # a dynamics contact manifold. Preserve margins and use single CCD.
        self.collision_model = copy.copy(ik.model)
        if self.collision_model.opt.disableflags & (
            int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        ):
            raise RuntimeError("CUDA planning requires collision detection enabled")
        self.collision_model.opt.disableflags |= int(
            mujoco.mjtDisableBit.mjDSBL_MULTICCD
        ) | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(self.collision_model)
        self.model.opt.graph_conditional = False
        self.thresholds = wp.array(contact_thresholds(ik), dtype=float, device=device)
        self.buffers = {}
        self.validation_passes = 0

    def sync_model(self):
        # MuJoCo Warp stores MuJoCo-order quaternions; do not reorder xyzw.
        model = self.ik.model
        self.model.body_pos.assign(model.body_pos[None].astype(np.float32))
        self.model.body_quat.assign(model.body_quat[None].astype(np.float32))

    def _buffer(self, size):
        capacity = 1 << (size - 1).bit_length()
        if capacity > 8192:
            raise RuntimeError(
                "CUDA path sample capacity exceeded; no partial path is accepted"
            )
        if capacity not in self.buffers:
            with wp.ScopedDevice(self.device):
                data = mjw.make_data(
                    self.collision_model,
                    nworld=capacity,
                    nconmax=64,
                    nccdmax=64,
                    njmax=1,
                )
            invalid = wp.zeros(capacity, dtype=int, device=self.device)
            phases = wp.zeros(capacity, dtype=int, device=self.device)
            commands = wp.zeros(capacity, dtype=Vec7, device=self.device)
            self.buffers[capacity] = data, invalid, phases, commands
        return self.buffers[capacity]

    def validate(self, segments, limits):
        """Generate joint commands on GPU and check all their geometry in one batch."""
        count = sum(s["count"] for s in segments)
        if not count:
            raise RuntimeError("Cannot validate an empty CUDA path")
        data, invalid, phases, commands = self._buffer(count)
        capacity = data.nworld
        self.last_count, self.last_capacity = count, capacity
        self.sync_model()

        def array(values, dtype):
            return wp.array(values, dtype=dtype, device=self.device)

        segment_ids, fractions, phase_ids, ranges = [], [], [], []
        for index, segment in enumerate(segments):
            start = len(segment_ids)
            segment_ids.extend([index] * segment["count"])
            fractions.extend(np.linspace(0, 1, segment["count"]).tolist())
            phase_ids.extend([PHASES.index(segment["phase"])] * segment["count"])
            ranges.append(slice(start, len(segment_ids)))
        padding = capacity - count
        segment_ids.extend([len(segments) - 1] * padding)
        fractions.extend([1.0] * padding)
        phase_ids.extend([phase_ids[-1]] * padding)
        phases.assign(np.asarray(phase_ids, dtype=np.int32))
        inputs = [
            array([s["qpos"] for s in segments], float),
            array([s["start"] for s in segments], Vec7),
            array([s["goal"] for s in segments], Vec7),
            array(self.ik.qadr[self.ik.arm], int),
            array(self.ik.qadr[self.ik.fingers], int),
            array([s["finger_start"] for s in segments], Vec7),
            array([s["finger_goal"] for s in segments], Vec7),
            array(segment_ids, int),
            array(fractions, float),
            data.qpos,
            commands,
        ]
        invalid.zero_()
        data.overflow.zero_()
        wp.launch(sample_kernel, capacity, inputs, device=self.device)
        wp.launch(
            joints_kernel,
            capacity,
            [commands, count, Vec7(*limits[:, 0]), Vec7(*limits[:, 1]), invalid],
            device=self.device,
        )
        with wp.ScopedDevice(self.device):
            mjw.kinematics(self.model, data)
            mjw.collision(self.model, data)
        wp.launch(
            contacts_kernel,
            data.naconmax,
            [
                data.nacon,
                data.contact.worldid,
                data.contact.geom,
                data.contact.dist,
                phases,
                self.thresholds,
                count,
                invalid,
            ],
            device=self.device,
        )
        self.validation_passes += 1
        if data.nacon.numpy()[0] > data.naconmax or np.any(data.overflow.numpy()):
            raise RuntimeError(
                "CUDA collision capacity overflow; the path was not completely checked"
            )
        errors = invalid.numpy()[:count]
        if np.any(errors):
            index = int(np.flatnonzero(errors)[0])
            reason = (
                "joint limit/continuity"
                if errors[index] & 3
                else "robot/table collision"
            )
            raise RuntimeError(
                f"Interpolated physical trajectory intersects robot/table: CUDA {reason} at sample {index}"
            )
        result = commands.numpy()[:count].astype(float)
        return [result[r].copy() for r in ranges]

    def prepare(self, plant, points, rotation):
        """Build one direct approach + Cartesian align/reach/lift; reject unsafe paths."""
        from g1_can_handoff.task import OPEN, CLOSED

        ik = self.ik
        qpos = ik.data.qpos.copy()
        adr, fingers = ik.qadr[ik.arm], ik.qadr[ik.fingers]
        limits = np.asarray(plant.config["arm_limits_rad"])[:7]
        arm = qpos[adr].copy()
        segments, phase_ranges, steps = [], {}, {}
        started = time.monotonic()

        def append(phase, goal, *, finger_goal=None, duration=2.0):
            nonlocal arm, qpos
            travel = float(np.max(np.abs(goal - arm)))
            seconds = max(duration, 1.5 * travel / 0.25, math.sqrt(6 * travel / 0.5))
            count = max(2, int(math.ceil(seconds / 0.02)) + 1)
            if finger_goal is not None:
                count = 51
            segments.append(
                dict(
                    phase=phase,
                    qpos=qpos.copy(),
                    start=arm.copy(),
                    goal=goal.copy(),
                    count=count,
                    finger_start=qpos[fingers].copy(),
                    finger_goal=qpos[fingers].copy()
                    if finger_goal is None
                    else np.asarray(finger_goal),
                )
            )
            arm = goal.copy()
            qpos[adr] = arm
            if finger_goal is not None:
                qpos[fingers] = finger_goal

        append("pregrasp", arm, finger_goal=OPEN)
        staging_begin = len(segments)
        # A measured arm below the table needs to swing behind the table edge
        # before rising. Use a fixed commissioned G1 corridor, checked in the
        # same batch as the remaining path, rather than a ten-second RRT.
        # This is a proposal, never an exemption from collision checking.
        if (
            plant.config.get("staging_enabled", True)
            and self.arm.forward(arm)[0][0, 2]
            < plant.config.get("table_height_m", -np.inf) + 0.04
        ):
            posture = np.asarray(
                plant.config.get(
                    "cuda_staging_posture_rad",
                    [0.30, 0.52, 1.14, 0.22, -0.53, 0.65, -0.18],
                ),
                dtype=float,
            )
            if posture.shape != (7,) or not np.isfinite(posture).all():
                raise ValueError("Expected seven finite CUDA staging joint targets")
            append("staging", posture)
        staging_end = len(segments)
        for index, phase in enumerate(("approach", "align", "reach", "lift")):
            plant.check()
            if index == 3:
                append("close", arm, finger_goal=CLOSED)
            start = arm.copy()
            begin = len(segments)
            target = np.asarray(points[index], dtype=float)
            if index == 0:
                goal = self.arm.solve(target, rotation, arm, limits)
                append(phase, goal)
            else:
                position = self.arm.forward(arm)[0][0].astype(float)
                distance = float(np.linalg.norm(target - position))
                if distance > 0.20 or (
                    phase == "lift" and target[2] < position[2] - 0.005
                ):
                    raise RuntimeError("Cartesian lift must ascend by at most 0.20 m")
                waypoints = max(1, int(math.ceil(distance / 0.01)))
                for j in range(1, waypoints + 1):
                    plant.check()
                    waypoint = position + (target - position) * (j / waypoints)
                    goal = self.arm.solve(waypoint, rotation, arm, limits, local=True)
                    append(phase, goal, duration=2.0 / waypoints)
            phase_ranges[phase] = (begin, len(segments))
            steps[phase] = dict(
                start=start,
                goal=arm.copy(),
                position=target.copy(),
                rotation=np.asarray(rotation).copy(),
            )
        samples = self.validate(segments, limits)
        plant.staging_waypoints = [
            s["goal"].tolist() for s in segments[staging_begin:staging_end]
        ]
        plant.prepared_staging = [
            (s["start"].copy(), s["goal"].copy(), samples[i].copy())
            for i, s in enumerate(segments)
            if staging_begin <= i < staging_end
        ]
        for phase, (begin, end) in phase_ranges.items():
            steps[phase]["poses"] = np.concatenate(samples[begin:end])
        plant.event(
            "cuda_path_prepared",
            backend="cuda" if self.device.is_cuda else "cpu_test",
            device=str(self.device),
            planning_s=time.monotonic() - started,
            validation_passes=1,
            command_samples=sum(len(s) for s in samples),
            motion_commands_sent=0,
        )
        return steps
