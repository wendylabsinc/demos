"""Torque-only whole-body balance: unconstrained base, contact-force QP."""

import numpy as np
import mujoco
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation
from .sim import Plant, scene, ARM
from .bar_delivery import decorate, REST, BODY_JOINTS, MobilePlant


class BalancedPlant(Plant):
    pose_collision = MobilePlant.pose_collision
    move = MobilePlant.move

    def __init__(self, model):
        super().__init__(model)
        assert model.neq == 0 and model.nmocap == 0
        self.n = 49
        self.right = np.array(
            [model.actuator(n.replace("left_", "right_")).id for n in ARM]
        )
        self.body_act = np.array([model.actuator(n).id for n in BODY_JOINTS])
        rest = np.array([0, 0.25, 0, 1.45, 0, 0, 0])
        nominal = np.array(
            [-0.1, 0, 0, 0.3, -0.2, 0] * 2
            + [0] * 3
            + rest.tolist()
            + (rest * np.array([1, -1, 1, 1, 1, 1, 1])).tolist()
        )
        self.data.qpos[:7] = [0, 0, 0.784, 1, 0, 0, 0]
        self.data.qpos[self.qadr[self.body_act]] = nominal
        self.targets = self.data.qpos[self.qadr].copy()
        self.pelvis = model.body("pelvis").id
        self.feet = [model.body(s + "_ankle_roll_link").id for s in ["left", "right"]]
        self.corners = np.array(
            [
                [-0.05, 0.025, -0.03],
                [-0.05, -0.025, -0.03],
                [0.12, 0.03, -0.03],
                [0.12, -0.03, -0.03],
            ]
        )
        self.mass = np.empty((model.nv, model.nv))
        self.tick = 0
        self.effort = np.zeros(model.nu)
        self.previous = None
        self.failed = 0
        self.support = [True, True]
        self.contacts_log = []
        self.furniture = {
            model.geom(i).id
            for i in range(model.ngeom)
            if model.geom(i).name in ("table", "counter_front")
            or (
                model.body(model.geom_bodyid[i]).name == "delivery_table"
                and model.geom_contype[i]
            )
        }
        self.max_furniture_force = 0.0
        self.table_contacts = 0
        mujoco.mj_forward(model, self.data)
        self.anchors = self.foot_points().copy()
        self.anchors[:, :, 2] = 0.005
        self.foot_goals = self.anchors.copy()
        self.com_goal = self.data.subtree_com[self.pelvis].copy()
        self.com_goal[2] = 0.70
        self.yaw = 0.0
        self.robot_dofs = self.dadr
        self.default = self.targets.copy()
        self.solver = None
        self.min_height = 10.0
        self.min_upright = 1.0

    def foot_points(self):
        return np.array(
            [
                self.data.xpos[b] + self.corners @ self.data.xmat[b].reshape(3, 3).T
                for b in self.feet
            ]
        )

    def control(self):
        m, d = self.model, self.data
        n = self.n
        J = np.zeros((24, n))
        positions = self.foot_points()
        for side, b in enumerate(self.feet):
            for point in range(4):
                jp = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, None, positions[side, point], b)
                J[side * 12 + point * 3 : side * 12 + point * 3 + 3] = jp[:, :n]
        jc = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
        jc = jc[:, :n]
        jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, None, jr, self.pelvis)
        jr = jr[:, :n]
        allj = np.vstack([J, jc, jr])
        jdot = (
            np.zeros(30)
            if self.previous is None
            else ((allj - self.previous) / 0.01) @ d.qvel[:n]
        )
        self.previous = allj.copy()
        mujoco.mj_fullM(m, d, self.mass)
        M = self.mass[:n, :n]
        bias = d.qfrc_bias[:n]
        # Acceleration and force variables; only joint torques reach MuJoCo.
        total = n + 24
        H = np.eye(total) * 1e-5
        g = np.zeros(total)

        def task(A, b, weight):
            nonlocal H, g
            B = np.zeros((len(A), total))
            B[:, :n] = A
            H += weight * B.T @ B
            g -= weight * B.T @ b

        foot_acc = (
            350 * (self.foot_goals - positions).reshape(-1)
            - 35 * J @ d.qvel[:n]
            - jdot[:24]
        )
        task(J, foot_acc, 100)
        comacc = (
            60 * (self.com_goal - d.subtree_com[self.pelvis])
            - 25 * jc @ d.qvel[:n]
            - jdot[24:27]
        )
        task(jc, comacc, 200)
        R = d.xmat[self.pelvis].reshape(3, 3)
        desired = Rotation.from_euler("z", self.yaw).as_matrix()
        angular = Rotation.from_matrix(desired @ R.T).as_rotvec()
        task(jr, 100 * angular - 20 * jr @ d.qvel[:n] - jdot[27:], 100)
        jq = np.zeros((m.nu, n))
        jq[np.arange(m.nu), self.dadr] = 1
        gain = np.full(m.nu, 60.0)
        gain[self.body_act[:12]] = 8.0
        accel = (
            gain * (self.targets - d.qpos[self.qadr])
            - 2 * np.sqrt(gain) * d.qvel[self.dadr]
        )
        task(jq, accel, 0.05)
        waist = self.body_act[12:15]
        task(
            jq[waist],
            100 * (self.targets[waist] - d.qpos[self.qadr[waist]])
            - 20 * d.qvel[self.dadr[waist]],
            100,
        )
        upper = np.r_[self.arm, self.right, self.fingers]
        task(
            jq[upper],
            120 * (self.targets[upper] - d.qpos[self.qadr[upper]])
            - 24 * d.qvel[self.dadr[upper]],
            10,
        )
        H[n:, n:] += np.eye(24) * 1e-4
        rows = []
        lo = []
        hi = []
        dynamic = np.c_[M, -J.T]
        rows.extend(dynamic[:6])
        lo.extend(-bias[:6])
        hi.extend(-bias[:6])
        torque = dynamic[self.dadr]
        rows.extend(torque)
        lo.extend(-self.torque - bias[self.dadr])
        hi.extend(self.torque - bias[self.dadr])
        for point in range(8):
            force = n + point * 3
            normal = np.zeros(total)
            normal[force + 2] = 1
            rows.append(normal)
            lo.append(0)
            hi.append(500 if self.support[point // 4] else 0)
            for axis in (0, 1):
                for sign in (-1, 1):
                    row = np.zeros(total)
                    row[force + axis] = sign
                    row[force + 2] = -0.7
                    rows.append(row)
                    lo.append(-np.inf)
                    hi.append(0)
        A = sparse.csc_matrix(np.array(rows))
        P = sparse.csc_matrix(np.triu(H * 2))
        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=2 * g,
            A=A,
            l=np.array(lo),
            u=np.array(hi),
            verbose=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=4000,
            polishing=False,
        )
        result = solver.solve()
        if result.info.status_val not in (1, 2):
            self.failed += 1
            raise ValueError(f"Contact QP failed: {result.info.status}")
        self.effort = np.clip(
            torque @ result.x + bias[self.dadr], -self.torque, self.torque
        )
        # Small wrist/finger inertias need torque-space position feedback.
        upper = np.r_[self.arm, self.right, self.fingers]
        pd = (
            self.kp[upper] * (self.targets[upper] - d.qpos[self.qadr[upper]])
            - self.kd[upper] * d.qvel[self.dadr[upper]]
            + bias[self.dadr[upper]]
        )
        self.effort[upper] = np.clip(pd, -self.torque[upper], self.torque[upper])

    def step(self, seconds):
        for _ in range(round(seconds / self.model.opt.timestep)):
            if self.tick % 5 == 0:
                self.control()
            self.data.ctrl[:] = self.effort
            upper = np.r_[self.arm, self.right, self.fingers]
            d = self.data
            pd = (
                self.kp[upper] * (self.targets[upper] - d.qpos[self.qadr[upper]])
                - self.kd[upper] * d.qvel[self.dadr[upper]]
                + d.qfrc_bias[self.dadr[upper]]
            )
            d.ctrl[upper] = np.clip(pd, -self.torque[upper], self.torque[upper])
            mujoco.mj_step(self.model, self.data)
            self.tick += 1
            for index, c in enumerate(self.data.contact):
                if c.geom1 not in self.furniture and c.geom2 not in self.furniture:
                    continue
                other = c.geom2 if c.geom1 in self.furniture else c.geom1
                body = self.model.body(self.model.geom_bodyid[other]).name
                if not body.startswith(
                    ("left_", "right_", "waist_", "torso", "pelvis")
                ):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                self.max_furniture_force = max(
                    self.max_furniture_force, float(abs(force[0]))
                )
                if force[0] > 0.25:
                    self.table_contacts += 1
                    raise ValueError(
                        f"Unintended furniture contact: {body}, force={force[0]:.2f}N"
                    )
            z = self.data.qpos[2]
            upright = self.data.xmat[self.pelvis].reshape(3, 3)[2, 2]
            self.min_height = min(self.min_height, float(z))
            self.min_upright = min(self.min_upright, float(upright))
            if z < 0.55 or upright < 0.7:
                raise ValueError(f"Balance lost z={z:.3f}, upright={upright:.3f}")
            if self.frame_callback:
                self.frame_callback(self)

    def ready(self):
        from scipy.optimize import least_squares

        adr = self.qadr[self.arm]
        lo, hi = self.model.jnt_range[self.joints[self.arm]].T
        for target in ([0.10, 0.40, 0.59], [0.10, 0.40, 0.94], [0.12, 0.35, 1.15]):
            start = self.data.site_xpos[self.site].copy()
            for t in np.linspace(0, 1, 80):
                position = start + t * t * (3 - 2 * t) * (np.array(target) - start)
                reference = self.data.qpos[adr].copy()
                self.ik_data.qpos[:] = self.data.qpos

                def error(q):
                    self.ik_data.qpos[adr] = q
                    mujoco.mj_forward(self.model, self.ik_data)
                    return np.r_[
                        self.ik_data.site_xpos[self.site] - position,
                        0.02 * (q - reference),
                    ]

                solution = least_squares(
                    error,
                    np.clip(reference, lo + 0.025, hi - 0.025),
                    bounds=(lo + 0.025, hi - 0.025),
                    max_nfev=80,
                )
                error(solution.x)
                if self.pose_collision(self.ik_data):
                    collisions = [
                        (
                            self.model.body(self.model.geom_bodyid[c.geom1]).name,
                            self.model.body(self.model.geom_bodyid[c.geom2]).name,
                            float(c.dist),
                        )
                        for c in self.ik_data.contact
                        if c.dist < -0.002
                    ]
                    raise ValueError(
                        f"Pre-reach trajectory collision t={t} target={target} actual={self.data.site_xpos[self.site]} q={solution.x} {collisions}"
                    )
                self.targets[self.arm] = solution.x
                self.step(0.02)
            self.step(0.3)
            print("READY", self.data.site_xpos[self.site], flush=True)

    def shift(self, xy, duration=1.5):
        start = self.com_goal[:2].copy()
        xy = np.asarray(xy)
        for t in np.linspace(0, 1, round(duration / 0.02)):
            self.com_goal[:2] = start + t * t * (3 - 2 * t) * (xy - start)
            self.step(0.02)

    def footstep(self, side, delta):
        other = 1 - side
        center = self.anchors[other].mean(axis=0)
        self.shift(center[:2], 3.0)
        self.step(1.0)
        start = self.anchors[side].copy()
        finish = start.copy()
        finish[:, :2] += delta
        self.support[side] = False
        for t in np.linspace(0, 1, 60):
            alpha = t * t * (3 - 2 * t)
            self.foot_goals[side] = start + alpha * (finish - start)
            self.foot_goals[side, :, 2] += 0.055 * np.sin(np.pi * t) ** 2
            self.step(0.02)
        self.support[side] = True
        self.anchors[side] = finish
        self.foot_goals[side] = finish
        self.step(0.4)
        print(
            "STEP",
            side,
            self.data.qpos[:3].round(3),
            self.contacts(),
            self.can_position().round(3),
            flush=True,
        )

    def go(self, goal):
        goal = np.asarray(goal)
        side = 1
        steps = 0
        while (
            np.linalg.norm(goal - (self.anchors.mean(axis=(0, 1))[:2] + [0.03, 0]))
            > 0.06
        ):
            if steps >= 80:
                raise ValueError("Navigation step limit exceeded")
            center = self.anchors.mean(axis=(0, 1))[:2] + [0.03, 0]
            delta = goal - center
            delta *= min(1.0, 0.07 / np.linalg.norm(delta))
            self.footstep(side, delta * 2)
            side = 1 - side
            steps += 1
        self.shift(self.anchors.mean(axis=(0, 1))[:2], 1.5)
        return float(np.linalg.norm(goal - self.data.qpos[:2]))


def scenario_decorate(root, height=0.85, delivery_height=0.75):
    """Change physical surfaces and their decorations, keeping control fixed."""
    decorate(root)
    world = root.find("worldbody")
    delta = height - 0.85
    top = world.find("geom[@name='table']")
    pos = np.fromstring(top.get("pos"), sep=" ")
    pos[2] = height - 0.035
    top.set("pos", " ".join(map(str, pos)))
    front = world.find("geom[@name='counter_front']")
    size = np.fromstring(front.get("size"), sep=" ")
    size[2] = (height - 0.05) / 2
    front.set("size", " ".join(map(str, size)))
    front.set("pos", f".695 .18 {size[2]}")
    for geom in world.findall("geom"):
        if geom is top or geom is front:
            continue
        pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
        if len(pos) == 3 and 0.3 <= pos[0] <= 0.7 and 0.85 <= pos[2] <= 1.05:
            pos[2] += delta
            geom.set("pos", " ".join(map(str, pos)))
    table = world.find("body[@name='delivery_table']")
    for geom in table.findall("geom"):
        if geom.get("name") == "delivery_surface":
            geom.set("pos", f"1.87 -.42 {delivery_height-.025}")
        elif geom.get("name") == "delivery_coaster":
            geom.set("pos", f"1.62 -.43 {delivery_height+.004}")
        elif geom.get("fromto"):
            endpoints = np.fromstring(geom.get("fromto"), sep=" ")
            endpoints[5] = delivery_height - 0.05
            geom.set("fromto", " ".join(map(str, endpoints)))


def make(height=0.85, x=0.315, y=0.19, delivery_height=0.75):
    return BalancedPlant(
        scene(
            height,
            x,
            y,
            decorate=lambda root: scenario_decorate(root, height, delivery_height),
        ).model
    )


if __name__ == "__main__":
    p = make()
    p.step(3)
    print(
        "STAND",
        p.data.qpos[:7],
        p.data.subtree_com[p.pelvis],
        p.min_height,
        p.min_upright,
        flush=True,
    )
    from .perception import SimCamera
    from .task import execute

    p.ready()
    c = SimCamera(p)
    obs, _ = c.observe()
    try:
        print(
            execute(
                p,
                obs,
                lambda x: print(x, p.data.qpos[:3], flush=True),
                lambda: c.observe()[0],
            ),
            flush=True,
        )
        p.step(10)
        print("HOLD", p.contacts(), p.can_position(), flush=True)
    except Exception:
        print("BASE", p.data.qpos[:7], "HAND", p.data.site_xpos[p.site], flush=True)
        raise
