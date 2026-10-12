"""Seven-joint FK and damped least-squares IK, executed entirely in Warp kernels.

Model constants come from the commissioned MuJoCo model; there is no URDF or
alternate geometry model. Angles are never wrapped modulo 2 pi.
"""

import numpy as np
import mujoco
import warp as wp

wp.set_module_options({"enable_backward": False})

Vec7 = wp.types.vector(length=7, dtype=wp.float32)
Mat77 = wp.types.matrix(shape=(7, 7), dtype=wp.float32)
Mat73 = wp.types.matrix(shape=(7, 3), dtype=wp.float32)


@wp.func
def fk(
    q: Vec7,
    base_p: wp.vec3,
    base_r: wp.mat33,
    offsets: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    anchors: wp.array(dtype=wp.vec3),
    axes: wp.array(dtype=wp.vec3),
    zeros: Vec7,
    site_p: wp.vec3,
    site_r: wp.mat33,
):
    p, r = base_p, base_r
    world_axes, world_anchors = Mat73(), Mat73()
    for j in range(7):
        p = p + r * offsets[j]
        r = r * rotations[j]
        a = r * axes[j]
        origin = p + r * anchors[j]
        for k in range(3):
            world_axes[j, k] = a[k]
            world_anchors[j, k] = origin[k]
        rotation = wp.quat_to_matrix(wp.quat_from_axis_angle(axes[j], q[j] - zeros[j]))
        r = r * rotation
        p = origin - r * anchors[j]
    return p + r * site_p, r * site_r, world_axes, world_anchors


@wp.func
def pose_cost(p: wp.vec3, r: wp.mat33, target: wp.vec3, orientation: wp.mat33):
    cost = wp.length_sq(p - target)
    for i in range(3):
        for j in range(3):
            cost += 0.0225 * (r[i, j] - orientation[i, j]) ** 2.0
    return cost


@wp.kernel
def solve_kernel(
    seeds: wp.array(dtype=Vec7),
    lo: Vec7,
    hi: Vec7,
    reference: Vec7,
    base_p: wp.vec3,
    base_r: wp.mat33,
    offsets: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    anchors: wp.array(dtype=wp.vec3),
    axes: wp.array(dtype=wp.vec3),
    zeros: Vec7,
    site_p: wp.vec3,
    site_r: wp.mat33,
    target: wp.vec3,
    orientation: wp.mat33,
    tolerance: float,
    weight: float,
    solutions: wp.array(dtype=Vec7),
    scores: wp.array(dtype=float),
):
    index = wp.tid()
    q = seeds[index]
    for j in range(7):
        q[j] = wp.clamp(q[j], lo[j], hi[j])
    damping = float(0.0001)
    for iteration in range(160):
        p, r, a, origins = fk(
            q, base_p, base_r, offsets, rotations, anchors, axes, zeros, site_p, site_r
        )
        error = p - target
        cost = pose_cost(p, r, target, orientation)
        if cost < 1.0e-10:
            break
        normal, gradient, linear = Mat77(), Vec7(), Mat73()
        for j in range(7):
            axis = wp.vec3(a[j, 0], a[j, 1], a[j, 2])
            dp = wp.cross(
                axis, p - wp.vec3(origins[j, 0], origins[j, 1], origins[j, 2])
            )
            for k in range(3):
                linear[j, k] = dp[k]
            gradient[j] = wp.dot(dp, error) + weight * weight * (q[j] - reference[j])
            for column in range(3):
                actual = wp.vec3(r[0, column], r[1, column], r[2, column])
                desired = wp.vec3(
                    orientation[0, column],
                    orientation[1, column],
                    orientation[2, column],
                )
                gradient[j] += 0.0225 * wp.dot(wp.cross(axis, actual), actual - desired)
        for i in range(7):
            for j in range(7):
                normal[i, j] = wp.dot(
                    wp.vec3(linear[i, 0], linear[i, 1], linear[i, 2]),
                    wp.vec3(linear[j, 0], linear[j, 1], linear[j, 2]),
                )
                normal[i, j] += 0.045 * wp.dot(
                    wp.vec3(a[i, 0], a[i, 1], a[i, 2]),
                    wp.vec3(a[j, 0], a[j, 1], a[j, 2]),
                )
            normal[i, i] += damping + weight * weight
        # Cholesky solve of the positive definite 7x7 normal equations.
        lower, y, step = Mat77(), Vec7(), Vec7()
        for i in range(7):
            for j in range(i + 1):
                value = normal[i, j]
                for k in range(j):
                    value -= lower[i, k] * lower[j, k]
                if i == j:
                    lower[i, j] = wp.sqrt(wp.max(value, 1.0e-12))
                else:
                    lower[i, j] = value / lower[j, j]
            value = -gradient[i]
            for j in range(i):
                value -= lower[i, j] * y[j]
            y[i] = value / lower[i, i]
        for back in range(7):
            i = 6 - back
            value = y[i]
            for j in range(i + 1, 7):
                value -= lower[j, i] * step[j]
            step[i] = value / lower[i, i]
        trial = q
        for j in range(7):
            trial[j] = wp.clamp(q[j] + wp.clamp(step[j], -0.15, 0.15), lo[j], hi[j])
        tp, tr, ta, to = fk(
            trial,
            base_p,
            base_r,
            offsets,
            rotations,
            anchors,
            axes,
            zeros,
            site_p,
            site_r,
        )
        trial_cost = pose_cost(tp, tr, target, orientation)
        for j in range(7):
            cost += weight * weight * (q[j] - reference[j]) ** 2.0
            trial_cost += weight * weight * (trial[j] - reference[j]) ** 2.0
        if trial_cost < cost:
            q = trial
            damping = wp.max(1.0e-7, damping * 0.5)
        else:
            damping = wp.min(1.0, damping * 4.0)
    p, r, a, origins = fk(
        q, base_p, base_r, offsets, rotations, anchors, axes, zeros, site_p, site_r
    )
    angle_error = float(0.0)
    for i in range(3):
        for j in range(3):
            angle_error += 0.0225 * (r[i, j] - orientation[i, j]) ** 2.0
    score = float(0.0)
    for j in range(7):
        score += (q[j] - reference[j]) ** 2.0
    if wp.length(p - target) > tolerance or wp.sqrt(angle_error) > 0.02:
        score = 1.0e30
    solutions[index] = q
    scores[index] = score


@wp.kernel
def select_kernel(
    solutions: wp.array(dtype=Vec7),
    scores: wp.array(dtype=float),
    selected: wp.array(dtype=Vec7),
    status: wp.array(dtype=int),
):
    index = int(0)
    for j in range(1, 3):
        if scores[j] < scores[index]:
            index = j
    selected[0] = solutions[index]
    status[0] = int(scores[index] < 1.0e29)


@wp.kernel
def fk_kernel(
    q: wp.array(dtype=Vec7),
    base_p: wp.vec3,
    base_r: wp.mat33,
    offsets: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.mat33),
    anchors: wp.array(dtype=wp.vec3),
    axes: wp.array(dtype=wp.vec3),
    zeros: Vec7,
    site_p: wp.vec3,
    site_r: wp.mat33,
    positions: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    p, r, a, o = fk(
        q[i], base_p, base_r, offsets, rotations, anchors, axes, zeros, site_p, site_r
    )
    positions[i] = p
    orientations[i] = r


class ArmKinematics:
    def __init__(self, ik, device):
        self.ik, self.device = ik, device
        model = ik.model
        self.joints = ik.joints[ik.arm]
        bodies = model.jnt_bodyid[self.joints]
        self.parent = int(model.body_parentid[bodies[0]])
        if (
            np.any(model.jnt_type[self.joints] != mujoco.mjtJoint.mjJNT_HINGE)
            or not np.array_equal(model.body_parentid[bodies[1:]], bodies[:-1])
            or model.site_bodyid[ik.site] != bodies[-1]
            or np.any(model.body_jntnum[bodies] != 1)
        ):
            raise RuntimeError("CUDA IK requires the verified seven-hinge G1 arm chain")

        def mats(quats):
            result = np.empty((len(quats), 3, 3))
            for matrix, quat in zip(result, quats):
                mujoco.mju_quat2Mat(matrix.ravel(), quat)
            return result

        self.constants = [
            wp.array(model.body_pos[bodies], dtype=wp.vec3, device=device),
            wp.array(mats(model.body_quat[bodies]), dtype=wp.mat33, device=device),
            wp.array(model.jnt_pos[self.joints], dtype=wp.vec3, device=device),
            wp.array(model.jnt_axis[self.joints], dtype=wp.vec3, device=device),
            Vec7(*model.qpos0[ik.qadr[ik.arm]]),
            wp.vec3(*model.site_pos[ik.site]),
            wp.mat33(*mats([model.site_quat[ik.site]])[0].ravel()),
        ]
        self.seeds = wp.zeros(3, dtype=Vec7, device=device)
        self.solutions = wp.zeros(3, dtype=Vec7, device=device)
        self.scores = wp.zeros(3, dtype=float, device=device)
        self.selected = wp.zeros(1, dtype=Vec7, device=device)
        self.status = wp.zeros(1, dtype=int, device=device)

    def base(self):
        data = self.ik.data
        return [wp.vec3(*data.xpos[self.parent]), wp.mat33(*data.xmat[self.parent])]

    def forward(self, q):
        values = wp.array(np.atleast_2d(q), dtype=Vec7, device=self.device)
        p = wp.empty(len(values), dtype=wp.vec3, device=self.device)
        r = wp.empty(len(values), dtype=wp.mat33, device=self.device)
        wp.launch(
            fk_kernel,
            len(values),
            [values, *self.base(), *self.constants, p, r],
            device=self.device,
        )
        return p.numpy(), r.numpy()

    def solve(self, position, rotation, start, limits, *, local=False):
        position, rotation = np.asarray(position), np.asarray(rotation)
        if (
            position.shape != (3,)
            or rotation.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
        ):
            raise ValueError("Expected finite CUDA IK position and rotation")
        lo, hi = limits[:, 0].copy(), limits[:, 1].copy()
        if local:
            lo, hi = np.maximum(lo, start - 0.2), np.minimum(hi, start + 0.2)
            seeds = np.tile(start, (3, 1))
        else:
            seeds = np.array(
                [
                    start,
                    [-1.2, 0.2, 0, 1.2, 0, 0, 0],
                    [-0.5, 0.2, 1.3, 1.4, 0.7, -0.9, -0.5],
                ]
            )
        self.seeds.assign(seeds.astype(np.float32))
        wp.launch(
            solve_kernel,
            3,
            [
                self.seeds,
                Vec7(*lo),
                Vec7(*hi),
                Vec7(*start),
                *self.base(),
                *self.constants,
                wp.vec3(*position),
                wp.mat33(*rotation.ravel()),
                min(0.008, getattr(self.ik, "ik_position_tolerance", 0.005)),
                getattr(self.ik, "ik_reference_weight", 0.0),
                self.solutions,
                self.scores,
            ],
            device=self.device,
        )
        wp.launch(
            select_kernel,
            1,
            [self.solutions, self.scores, self.selected, self.status],
            device=self.device,
        )
        if not self.status.numpy()[0]:
            raise RuntimeError(
                "Continuous Cartesian lift IK cannot preserve the grasp pose (CUDA)"
            )
        return self.selected.numpy()[0].astype(float)
