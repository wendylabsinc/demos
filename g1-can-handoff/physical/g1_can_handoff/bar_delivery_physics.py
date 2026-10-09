"""G1 delivery using torques and simulated contact, with no support constraints."""

import json
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .whole_body import make
from .bar_delivery import render
from . import bar_film as bar
from .perception import SimCamera
from .task import execute, OPEN

OUTPUT = bar.PROJECT / "results" / "bar-delivery-physics"


def simulate(
    resume_transfer=False,
    height=0.85,
    x=0.315,
    y=0.19,
    delivery_height=0.75,
    on_start=None,
):
    p = make(height, x, y, delivery_height)
    frames = []
    events = []
    phase = ["arms_at_hips"]
    last = [-1.0]
    if on_start is not None:
        on_start(p, frames, events)
    if resume_transfer:
        saved = np.load(OUTPUT / "transfer-state.npz")
        for key in ("qpos", "qvel", "ctrl", "qacc_warmstart"):
            getattr(p.data, key)[:] = saved[key]
        p.data.time = float(saved["time"])
        p.targets = saved["targets"]
        p.anchors = saved["anchors"]
        p.foot_goals = saved["foot_goals"]
        p.com_goal = saved["com_goal"]
        p.effort = saved["effort"]
        p.min_height = float(saved["min_height"])
        p.min_upright = float(saved["min_upright"])
        p.max_furniture_force = float(saved["max_furniture_force"])
        mujoco.mj_forward(p.model, p.data)
        archive = np.load(OUTPUT / "transfer-frames.npz")
        a = {k: archive[k] for k in archive.files}
        frames = [{k: v[i] for k, v in a.items()} for i in range(len(a["time"]))]
        events = json.loads((OUTPUT / "transfer-events.json").read_text())
        last[0] = frames[-1]["time"]
        phase[0] = "walk_along_table"

    def record(p):
        if p.data.time - last[0] < 0.05 - 0.0001:
            return
        last[0] = p.data.time
        frames.append(
            dict(
                time=float(p.data.time),
                qpos=p.data.qpos.copy(),
                mocap_pos=p.data.mocap_pos.copy(),
                mocap_quat=p.data.mocap_quat.copy(),
                phase=phase[0],
            )
        )

    def event(name):
        phase[0] = name
        events.append(
            dict(phase=name, time=float(p.data.time), base=p.data.qpos[:3].tolist())
        )
        print(name, p.data.qpos[:3].round(3).tolist(), flush=True)

    p.frame_callback = record
    if not resume_transfer:
        record(p)
        p.step(3)
        event("prepare_reach")
        p.ready()
        camera = SimCamera(p)
        o, rgb = camera.observe()
        pickup = execute(p, o, event, lambda: camera.observe()[0])
        camera.close()
        event("rotate_for_carry")
        p.move(
            p.data.site_xpos[p.site].copy(),
            3,
            Rotation.from_euler("x", np.pi / 2).as_matrix(),
        )
        route = []
        for name, goal in [
            ("walk_behind_counter", [0, -2.30]),
            ("round_counter_end", [1.27, -2.30]),
            ("approach_table_corner", [1.27, -1.13]),
            ("walk_along_table", [1.27, -0.62]),
        ]:
            event(name)
            error = p.go(goal)
            if len(p.contacts()) < 2 or p.can_position()[2] < height:
                raise ValueError("Can lost during physical carry")
            route.append(
                dict(goal=goal, error_m=error, actual=p.data.qpos[:2].tolist())
            )
        np.savez_compressed(
            OUTPUT / "transfer-state.npz",
            qpos=p.data.qpos,
            qvel=p.data.qvel,
            ctrl=p.data.ctrl,
            qacc_warmstart=p.data.qacc_warmstart,
            time=p.data.time,
            targets=p.targets,
            anchors=p.anchors,
            foot_goals=p.foot_goals,
            com_goal=p.com_goal,
            effort=p.effort,
            min_height=p.min_height,
            min_upright=p.min_upright,
            max_furniture_force=p.max_furniture_force,
        )
        np.savez_compressed(
            OUTPUT / "transfer-frames.npz",
            **{k: np.stack([f[k] for f in frames]) for k in frames[0]},
        )
        (OUTPUT / "transfer-events.json").write_text(json.dumps(events))
        (OUTPUT / "transfer-result.json").write_text(
            json.dumps(dict(pickup=pickup, route=route))
        )
    else:
        transport = json.loads((OUTPUT / "transfer-result.json").read_text())
        pickup = transport["pickup"]
        route = transport["route"]
    event("adjust_table_stance")
    correction = 1.28 - p.data.qpos[0]
    if correction > 0.025:
        for side in (1, 0):
            p.footstep(side, np.array([min(correction, 0.12), 0.0]))
        p.shift(p.anchors.mean(axis=(0, 1))[:2], 1.5)
        p.step(1.0)
    event("restore_upright")
    p.move(p.data.site_xpos[p.site].copy(), 3, np.eye(3))
    if len(p.contacts()) < 2:
        raise ValueError("Can lost during upright transition")
    event("align_over_table")
    rotation = p.data.site_xmat[p.site].reshape(3, 3).copy()
    can_rotation = p.data.xmat[p.can_id].reshape(3, 3).copy()
    relative = rotation.T @ can_rotation
    place_rotation = relative.T
    offset = rotation.T @ (p.can_position() - p.data.site_xpos[p.site])
    target = np.array([1.62, -0.43, delivery_height + 0.061])
    hand_target = target - place_rotation @ offset
    p.move(hand_target + [0, 0, 0.14], 2, place_rotation)
    event("lower_can")
    p.move(hand_target + [0, 0, 0.006], 3, place_rotation)
    event("release_on_table")
    p.hand(OPEN, 1.2)
    p.step(0.8)
    event("withdraw_hand")
    p.move(hand_target + [0, 0.07, 0.16], 2, place_rotation)
    p.step(2)
    pos = p.can_position()
    axis = p.data.xmat[p.can_id].reshape(3, 3)[:, 2]
    if (
        np.linalg.norm(pos[:2] - target[:2]) > 0.10
        or abs(pos[2] - target[2]) > 0.025
        or axis[2] < 0.95
    ):
        raise ValueError(f"Placement failed: {pos}, axis {axis}")
    event("finished")
    p.step(2)
    assert p.model.neq == 0 and p.model.nmocap == 0
    assert not np.any(p.data.qfrc_applied) and not np.any(p.data.xfrc_applied)
    result = dict(
        success=True,
        physics_only=True,
        base_support_constraints=0,
        object_attachments=0,
        mocap_bodies=0,
        external_forces=False,
        controller="Torque-only contact-force QP with joint feedback, quasi-static steps",
        scenario=dict(height=height, x=x, y=y, delivery_height=delivery_height),
        pickup=pickup,
        route=route,
        events=events,
        final_can_position=pos.tolist(),
        can_upright_cosine=float(axis[2]),
        min_pelvis_height=p.min_height,
        min_upright_cosine=p.min_upright,
        simulation_duration_s=float(p.data.time),
        unintended_furniture_contacts=p.table_contacts,
        max_robot_furniture_contact_force_N=p.max_furniture_force,
        hardware_validated=False,
    )
    np.savez_compressed(
        OUTPUT / "states.npz",
        **{k: np.stack([f[k] for f in frames]) for k in frames[0]},
    )
    (OUTPUT / "simulation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return p, frames


def edited_frames(frames):
    selected = []
    walking = (
        "walk_behind_counter",
        "round_counter_end",
        "approach_table_corner",
        "walk_along_table",
    )
    for name in dict.fromkeys(f["phase"] for f in frames):
        segment = [f for f in frames if f["phase"] == name]
        if name in walking:
            # Show consecutive real steps in each part of the route; cuts only.
            length = min(len(segment), 240)
            start = max(0, (len(segment) - length) // 2)
            segment = segment[start : start + length]
        selected += segment
    for index, f in enumerate(selected):
        f = f.copy()
        f["simulation_time"] = f["time"]
        f["time"] = index * 0.05
        selected[index] = f
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--reuse-states", action="store_true")
    parser.add_argument("--resume-transfer", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bar.textures()
    if args.reuse_states:
        archive = np.load(OUTPUT / "states.npz")
        a = {k: archive[k] for k in archive.files}
        frames = [{k: v[i] for k, v in a.items()} for i in range(len(a["time"]))]
        p = make()
    else:
        p, frames = simulate(args.resume_transfer)
    if not args.simulate_only:
        render(p, edited_frames(frames), OUTPUT, label="MUJOCO  ·  PHYSICS SIMULATION")


if __name__ == "__main__":
    main()
