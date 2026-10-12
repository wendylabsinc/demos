"""Bounded joint-space staging candidates using the existing collision model.

This module plans only. It does not publish commands or bypass preflight.
A returned path still needs the physical plant's trajectory validation and
calibration before execution. No simulated object pose is read.
"""

import time
import mujoco
import numpy as np


def joint_path(
    plant, goal, limits, *, seed=7, max_nodes=1800, timeout_s=10.0, check=None
):
    """Find a bounded collision-free seven-joint route; return waypoints and stats."""
    model = plant.model
    data = mujoco.MjData(model)
    data.qpos[:] = plant.data.qpos
    adr = plant.qadr[plant.arm]
    start = data.qpos[adr].copy()
    goal = np.asarray(goal, dtype=float)
    bounds = np.asarray(limits, dtype=float)
    if goal.shape != (7,) or bounds.shape != (7, 2) or not np.isfinite(goal).all():
        raise ValueError("Expected seven-joint goal and limits")
    lo, hi = bounds.T
    deadline = time.monotonic() + timeout_s
    checks = 0
    rejected_candidates = 0
    edge_resolution = 0.004

    def valid(q):
        nonlocal checks
        if check is not None:
            check()
        if time.monotonic() > deadline:
            raise RuntimeError("Staging planning deadline reached")
        if np.any(q < lo) or np.any(q > hi):
            return False
        data.qpos[adr] = q
        # Staging checks consume collision geometry; skip velocity/dynamics.
        mujoco.mj_fwdPosition(model, data)
        checks += 1
        return not plant.pose_collision(data)

    def edge(a, b):
        return all(
            valid(a + (b - a) * alpha)
            for alpha in np.linspace(
                0, 1, max(2, int(np.ceil(np.max(np.abs(b - a)) / edge_resolution)) + 1)
            )
        )

    if not valid(start) or not valid(goal):
        raise RuntimeError("Staging endpoint collides or exceeds limits")
    if edge(start, goal):
        return [goal.tolist()], dict(
            collision_checks=checks,
            nodes=0,
            edge_resolution_rad=edge_resolution,
            rejected_candidates=0,
        )
    rng = np.random.default_rng(seed)
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, target):
        nodes, parents = tree
        if sum(len(t[0]) for t in trees) >= max_nodes:
            raise RuntimeError("Staging planning node limit reached")
        nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
        delta = target - nodes[nearest]
        length = np.linalg.norm(delta)
        new = target if length <= 0.35 else nodes[nearest] + delta * 0.35 / length
        if not edge(nodes[nearest], new):
            return None, False
        nodes.append(new)
        parents.append(nearest)
        return len(nodes) - 1, length <= 0.35

    def trace(tree, index):
        nodes, parents = tree
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    for iteration in range(max_nodes):
        side = iteration % 2
        a, b = trees[side], trees[1 - side]
        target = b[0][0] if iteration % 5 == 0 else rng.uniform(lo, hi)
        ai, _ = extend(a, target)
        if ai is None:
            continue
        for _ in range(80):
            bi, joined = extend(b, a[0][ai])
            if bi is None:
                break
            if joined:
                first, second = trace(a, ai), trace(b, bi)
                path = first + second[-2::-1]
                if side:
                    path = path[::-1]
                # Deterministic greedy shortcutting keeps only necessary turns.
                short = [path[0]]
                i = 0
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and not edge(path[i], path[j]):
                        j -= 1
                    short.append(path[j])
                    i = j
                if not all(edge(x, y) for x, y in zip(short, short[1:])):
                    # A rejected connection is not proof that every route is
                    # blocked. Keep searching within the same node/deadline
                    # budgets; cancellation still propagates from valid().
                    rejected_candidates += 1
                    break
                return [q.tolist() for q in short[1:]], dict(
                    collision_checks=checks,
                    nodes=sum(len(t[0]) for t in trees),
                    seed=seed,
                    edge_resolution_rad=edge_resolution,
                    rejected_candidates=rejected_candidates,
                )
    raise RuntimeError("No staging path within bounded search")
