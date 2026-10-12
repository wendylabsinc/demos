"""Motion-free CUDA benchmark of frozen MJB + feedback bundles.

Create bundles with offline.planner_replay.fixture and mj_saveModel. This script
does not connect to a robot. Timings cover prepare, IK, command sampling and the
one collision batch; independent CPU audits run outside the timed region.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import copy
import math
import numpy as np
import mujoco
from g1_can_handoff.sim import Plant
from physical_core import PhysicalPlant
from gpu_planner import CudaPlanner, PHASES, warmup_cuda
from pickup import prepare


def load(directory, index):
    meta = json.loads((directory / f"trial-{index}.json").read_text())
    scene = Plant(
        mujoco.MjModel.from_binary_path(str(directory / f"trial-{index}.mjb"))
    )
    scene.table_clearance_m = meta["table_clearance"]
    scene.planning_table_height_offset_m = meta["table_offset"]

    def read():
        snapshot = copy.deepcopy(meta["snapshot"])
        for sample in snapshot.values():
            sample["time"] = 100.0
        return snapshot

    events = []
    config = dict(meta["config"], planning_backend="cuda", grasp_candidate_budget=1)
    plant = PhysicalPlant(
        scene,
        None,
        None,
        read,
        config,
        lambda phase, **data: events.append(dict(phase=phase, **data)),
        clock=SimpleNamespace(monotonic=lambda: 100.0),
    )
    observation = SimpleNamespace(
        center=np.array(meta["center"]), table_height=meta["table_height"], timestamp=0
    )

    def observe(*args):
        observation.timestamp = time.monotonic()
        return observation, dict(source="frozen_feedback_replay")

    return plant, observe, events


def audit(plant):
    """Independently audit every GPU-returned pose using original CPU geometry."""
    engine = plant.cuda_planner
    data, invalid, phases, commands = engine.buffers[engine.last_capacity]
    values = data.qpos.numpy()[: engine.last_count]
    phase_ids = phases.numpy()
    model, cpu = plant.ik.model, mujoco.MjData(plant.ik.model)
    table = model.geom("table").id
    left = [
        i
        for i in range(model.ngeom)
        if model.geom_contype[i]
        and model.body(model.geom_bodyid[i]).name.startswith("left_")
    ]
    minimum = 1.0
    old_phase = plant.ik.contact_phase
    try:
        for qpos, phase in zip(values, phase_ids):
            cpu.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, cpu)
            plant.ik.set_contact_phase(PHASES[phase])
            contact = plant.ik.collision_details(cpu)
            if contact:
                raise AssertionError(f"CPU audit rejected GPU-accepted pose: {contact}")
            minimum = min(
                minimum,
                *(
                    mujoco.mj_geomDistance(model, cpu, table, g, 0.1, None)
                    for g in left
                ),
            )
        if minimum < plant.ik.table_clearance_m - 1e-5:
            raise AssertionError(
                f"GPU path lacks independent table clearance: {minimum}"
            )
    finally:
        plant.ik.set_contact_phase(old_phase)
    return dict(samples=len(values), min_table_distance_m=minimum, passed=True)


def audit_rejections(plant):
    """GPU must reject the recorded table-crossing shortcut and a raw 2pi flip."""
    from g1_can_handoff.task import OPEN

    ik, engine = plant.ik, plant.cuda_planner
    limits = np.asarray(plant.config["arm_limits_rad"])[:7]
    start = plant.staging_start.copy()
    goals = dict(
        direct_table_crossing=plant.prepared_steps["approach"]["goal"].copy(),
        joint_flip=start + [0, 0, 0, 0, 0, 0, 2 * np.pi],
    )
    rejected = {}
    for name, goal in goals.items():
        travel = float(np.max(np.abs(goal - start)))
        count = max(2, math.ceil(1.5 * travel / 0.25 / 0.02) + 1)
        segment = dict(
            phase="approach",
            qpos=ik.data.qpos.copy(),
            start=start,
            goal=goal,
            finger_start=np.asarray(OPEN),
            finger_goal=np.asarray(OPEN),
            count=count,
        )
        try:
            engine.validate([segment], limits)
        except RuntimeError as exc:
            if "trajectory intersects" not in str(exc):
                raise
            rejected[name] = str(exc)
        else:
            raise AssertionError(f"GPU accepted unsafe input: {name}")
    return rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trials", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Include exact model/snapshot recording in timed prepare",
    )
    args = parser.parse_args()
    report = dict(
        actuation=False,
        backend="cuda",
        recording=args.record,
        source_base="28df52d0f22a72328585f88196fc050c0a465dd1",
        trials=[],
    )
    report["passive_warmup"] = warmup_cuda(load(args.bundle, args.trials[0])[0].ik)
    for index in args.trials:
        plant, observe, events = load(args.bundle, index)
        if args.record:
            from planning_snapshot import PlanningRecorder

            plant.planning_recorder = PlanningRecorder(
                args.output.parent / f"snapshots-{index}", plant.event
            )
        start = time.monotonic()
        plant.cuda_planner = CudaPlanner(plant.ik)
        initialized = time.monotonic() - start
        start = time.monotonic()
        try:
            prepare(plant, observe, plant.event)
            cold = time.monotonic() - start
            audit_result = audit(plant)
            times, passes = [], []
            for _ in range(args.repeats):
                before = plant.cuda_planner.validation_passes
                start = time.monotonic()
                prepare(plant, observe, plant.event)
                times.append(time.monotonic() - start)
                passes.append(plant.cuda_planner.validation_passes - before)
            item = dict(
                trial=index,
                passed=True,
                initialization_s=initialized,
                cold_prepare_s=cold,
                warm_prepare_s=times,
                validation_passes=passes,
                device=str(plant.cuda_planner.device),
                independent_cpu_audit=audit_result,
                rejected_candidates=sum(
                    e["phase"] == "grasp_candidate_rejected" for e in events
                ),
                unsafe_inputs_rejected=audit_rejections(plant),
            )
            if args.record:
                from planning_snapshot import replay

                snapshot = sorted(
                    plant.planning_recorder.directory.glob("*-preflight.json")
                )[-1]
                item["snapshot_replay"] = replay(snapshot)
                if not item["snapshot_replay"]["identical"]:
                    raise AssertionError("CUDA planning snapshot replay differs")
        except Exception as exc:
            item = dict(
                trial=index,
                passed=False,
                initialization_s=initialized,
                error=f"{type(exc).__name__}: {exc}",
            )
        report["trials"].append(item)
        print(json.dumps(item), flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    return 0 if all(t["passed"] for t in report["trials"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
