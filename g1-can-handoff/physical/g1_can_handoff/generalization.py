"""Frozen-controller physics evaluation across table heights and can positions."""

import argparse, json, time, hashlib, traceback, contextlib, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .whole_body import make
from .task import execute
from .perception import SimCamera
from . import bar_film as bar

OUTPUT = bar.PROJECT / "results" / "generalization-physics"


def suite():
    cases = []
    for h in (0.75, 0.85, 0.95):
        for x in (0.28, 0.315, 0.35):
            for y in (0.10, 0.19, 0.28):
                cases.append(
                    dict(id=f"g{len(cases):02}", kind="grid", height=h, x=x, y=y)
                )
    rng = np.random.default_rng(20261001)
    for i in range(6):
        cases.append(
            dict(
                id=f"r{i:02}",
                kind="heldout",
                height=float(rng.uniform(0.75, 0.95)),
                x=float(rng.uniform(0.28, 0.35)),
                y=float(rng.uniform(0.10, 0.28)),
            )
        )
    for i, (h, x, y) in enumerate(
        [
            (0.65, 0.315, 0.19),
            (1.05, 0.315, 0.19),
            (0.85, 0.24, 0.19),
            (0.85, 0.48, 0.19),
            (0.85, 0.315, 0.04),
            (0.85, 0.315, 0.36),
        ]
    ):
        cases.append(dict(id=f"b{i:02}", kind="boundary", height=h, x=x, y=y))
    return cases


def trial(case):
    folder = OUTPUT / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    p = make(case["height"], case["x"], case["y"])
    frames = []
    phase = ["stand"]
    last = [-1.0]
    events = []
    camera = None
    result = dict(
        **case,
        success=False,
        physics_only=True,
        support_constraints=int(p.model.neq),
        mocap_bodies=int(p.model.nmocap),
        can_mass_kg=0.37,
        can_friction=1.0,
        stage="stand",
    )

    def record(p):
        if p.data.time - last[0] < 0.1 - 0.0001:
            return
        last[0] = p.data.time
        frames.append(
            dict(time=float(p.data.time), qpos=p.data.qpos.copy(), phase=phase[0])
        )

    def event(name):
        phase[0] = name
        result["stage"] = name
        events.append(dict(phase=name, time=float(p.data.time)))

    p.frame_callback = record
    record(p)
    log = io.StringIO()
    try:
        assert p.model.neq == 0 and p.model.nmocap == 0
        with contextlib.redirect_stdout(log):
            p.step(3)
            event("prepare_reach")
            p.ready()
            event("perception")
            camera = SimCamera(p)
            obs, rgb = camera.observe()
            from PIL import Image

            Image.fromarray(rgb).save(folder / "observation.png")
            result["observed_center"] = obs.center.tolist()
            result["observed_table_height"] = float(obs.table_height)
            result["observation_error_m"] = float(
                np.linalg.norm(obs.center - p.can_position())
            )
            result["pickup"] = execute(p, obs, event, lambda: camera.observe()[0])
            result["pickup_success"] = True
            camera.close()
            camera = None
            event("rotate_for_carry")
            p.move(
                p.data.site_xpos[p.site].copy(),
                3,
                Rotation.from_euler("x", np.pi / 2).as_matrix(),
            )
            if len(p.contacts()) < 2:
                raise ValueError("Grip lost during carry rotation")
            for side in (1, 0):
                event(f"carry_step_{side}")
                p.footstep(side, np.array([0, -0.08]))
                if len(p.contacts()) < 2 or p.can_position()[2] < case["height"] + 0.08:
                    raise ValueError("Grip lost during two-step carry")
            p.shift(p.anchors.mean(axis=(0, 1))[:2], 1.5)
            event("hold_after_steps")
            p.step(2)
            if len(p.contacts()) < 2 or p.can_position()[2] < case["height"] + 0.08:
                raise ValueError("Grip slipped after carry")
            assert not np.any(p.data.qfrc_applied) and not np.any(p.data.xfrc_applied)
            if p.max_furniture_force > 0:
                raise ValueError("Nonzero robot/furniture contact force")
            result.update(
                success=True,
                stage="passed_pickup_and_two_steps",
                contacts=p.contacts(),
                final_can_position=p.can_position().tolist(),
            )
    except Exception as error:
        result["failure"] = str(error)
        result["failure_type"] = type(error).__name__
    finally:
        if camera is not None:
            camera.close()
        result.update(
            simulation_time_s=float(p.data.time),
            wall_time_s=time.monotonic() - began,
            min_pelvis_height=p.min_height,
            min_upright_cosine=p.min_upright,
            unintended_furniture_contacts=p.table_contacts,
            max_furniture_contact_force_N=p.max_furniture_force,
            events=events,
        )
        np.savez_compressed(
            folder / "states.npz",
            **{k: np.stack([f[k] for f in frames]) for k in frames[0]},
        )
        (folder / "log.txt").write_text(log.getvalue())
        (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bar.textures()
    cases = suite()
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    files = ("whole_body.py", "sim.py", "task.py", "perception.py")
    manifest = dict(
        seed=20261001,
        criterion="Observed-pose pickup, physical lift and offer, horizontal carry, two unsupported steps, 2s hold; no furniture contact",
        cases=cases,
        controller_hashes={
            f: hashlib.sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
            for f in files
        },
        full_route_tested_for_every_case=False,
    )
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    results = []
    pending = []
    for c in cases:
        path = OUTPUT / c["id"] / "result.json"
        if args.resume and path.exists():
            results.append(json.loads(path.read_text()))
        else:
            pending.append(c)
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {pool.submit(trial, c): c for c in pending}
        for future in as_completed(futures):
            c = futures[future]
            try:
                r = future.result()
            except Exception as e:
                r = dict(**c, success=False, worker_error=str(e))
            results.append(r)
            results.sort(key=lambda x: x["id"])
            (OUTPUT / "results.json").write_text(json.dumps(results, indent=2))
            print(
                f"{len(results)}/{len(cases)} {r['id']} h={r['height']:.3f} x={r['x']:.3f} y={r['y']:.3f}: {'PASS' if r['success'] else 'FAIL'} {r.get('stage','worker')} {r.get('failure',r.get('worker_error',''))}",
                flush=True,
            )
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2))
    print("COMPLETE", sum(r["success"] for r in results), "/", len(results), flush=True)


if __name__ == "__main__":
    main()
