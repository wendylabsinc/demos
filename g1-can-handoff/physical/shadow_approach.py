"""Read-only IK and full interpolated collision check for a mount hypothesis."""

import json, numpy as np, urllib.request, mujoco
from g1_can_handoff.sim import scene
from physical_core import PhysicalPlant


def shadow(hypothesis, limits):
    s = json.load(urllib.request.urlopen("http://127.0.0.1:8102/status"))["measured"]
    height = hypothesis["table_height_hypothesis_m"]
    p = scene(height=height)
    plant = PhysicalPlant(
        p,
        None,
        None,
        lambda: json.load(urllib.request.urlopen("http://127.0.0.1:8102/status"))[
            "measured"
        ],
        {"arm_limits_rad": limits},
        lambda *a, **k: None,
    )
    s = json.load(urllib.request.urlopen("http://127.0.0.1:8102/status"))["measured"]
    p.model.jnt_range[p.joints[p.arm]] = np.array(limits)[:7]
    center = np.r_[hypothesis["target_xy_hypothesis_m"], height + 0.061]
    attempts = []
    for pitch, yaw in [
        (0, 0),
        (0, 0.5),
        (0, 1),
        (-0.4, 0),
        (-0.4, 0.5),
        (0.4, 0),
        (0.4, 0.5),
        (0.7, 0),
        (0.7, 0.5),
        (0.7, 1),
    ]:
        cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
            [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        )
        target = center + R @ np.array([0, 0.05, 0]) + [0, 0, 0.16]
        try:
            goal = p.solve(target, R)
            start = np.array(s["body"]["q"][15:22])
            collision = False
            collision_detail = None
            p.ik_data.qpos[:] = p.data.qpos
            for t in np.linspace(0, 1, 300):
                p.ik_data.qpos[p.qadr[p.arm]] = start + (goal - start) * (
                    t * t * (3 - 2 * t)
                )
                mujoco.mj_forward(p.model, p.ik_data)
                if p.pose_collision(p.ik_data):
                    collision = True
                    collision_detail = {
                        "fraction": float(t),
                        "contacts": [
                            {
                                "dist": float(c.dist),
                                "bodies": [
                                    p.model.body(p.model.geom_bodyid[g]).name
                                    for g in (c.geom1, c.geom2)
                                ],
                            }
                            for c in p.ik_data.contact
                            if c.dist < -0.001
                        ],
                    }
                    break
            attempts.append(
                {
                    "pitch": pitch,
                    "yaw": yaw,
                    "target": target.tolist(),
                    "joint_goal": goal.tolist(),
                    "path_collision": collision,
                    "collision_detail": collision_detail,
                }
            )
        except Exception as e:
            attempts.append({"pitch": pitch, "yaw": yaw, "error": str(e)})
    return {"estimated_geometry_only": True, "actuation": False, "attempts": attempts}
