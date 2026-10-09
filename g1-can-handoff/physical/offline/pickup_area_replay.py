"""Replay the three native workspace failures without network, DDS or actuation."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from physical_core import PhysicalPlant, calibrated_scene
from offline.planner_replay import fixture, clearance_auditor
import pickup
from g1_can_handoff import task

BASE = Path(__file__).resolve().parent.parent
CASES = (
    "far-forward-20261008T003403Z",
    "near-forward-20261008T003718Z",
    "near-midline-clearance-trial02",
)


def load_case(index):
    if index == 2:
        root = BASE / "offline/evidence/clearance-demos-20261008-preserved"
        plant, observation, events, logged = fixture(2, root)
        files = [root / "trial-02/trial.json", root / "trial-02/events.jsonl.gz"]
    else:
        name = ("20261008T003403Z", "20261008T003718Z")[index]
        files = [
            BASE / "offline/trial-events.json.gz",
            BASE / f"offline/attempts/{name}.json",
        ]
        events = json.loads(gzip.decompress(files[0].read_bytes()))[name]["events"]
        snapshot = copy.deepcopy(
            json.loads(files[1].read_text())["after"]["status"]["measured"]
        )
        config = copy.deepcopy(
            next(
                e["calibration"]
                for e in reversed(events)
                if e["phase"] == "scene_refreshed"
            )
        )
        center = next(
            e["center_m"]
            for e in reversed(events)
            if e["phase"] == "observed_can_collision_proxy"
        )
        snapshot["body"]["q"][:15] = config["reference_base_q"]
        snapshot["body"]["imu_rpy"] = config["reference_imu_rpy"]
        logged = []

        def event(phase, **fields):
            logged.append(dict(phase=phase, **fields))

        def read():
            result = copy.deepcopy(snapshot)
            for sample in result.values():
                sample["time"] = 100.0
            return result

        plant = PhysicalPlant(
            calibrated_scene(config, x=center[0], y=center[1]),
            None,
            None,
            read,
            config,
            event,
            clock=SimpleNamespace(monotonic=lambda: 100.0),
        )
        observation = SimpleNamespace(
            center=np.array(center),
            table_height=config["table_height_m"],
            timestamp=time.monotonic(),
        )
        plant.observe_can(observation)
    observation.timestamp = time.monotonic()
    provenance = dict(
        name=CASES[index],
        center_m=observation.center.tolist(),
        table_height_m=observation.table_height,
        input_files={
            str(f.relative_to(BASE)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in files
        },
        original_failure=next(
            e["error"] for e in reversed(events) if e["phase"] == "primary_failure"
        ),
        feedback_source="captured terminal snapshot; base joints/IMU replaced by recorded exposure calibration",
        missing_fields=[
            "feedback at exact planning instant",
            "independent physical calibration metrology",
        ]
        + (["exact RGB-D planning frame"] if index < 2 else []),
        receipt_times_rebased_offline=True,
    )
    return plant, observation, logged, provenance


def run(
    index, diagnostic_bounds=False, historical_bounds=False, diagnostic_variant=None
):
    plant, observation, logged, result = load_case(index)
    report = clearance_auditor(plant)
    before = plant.ik.data.qpos.copy()
    planner = task.plan
    if diagnostic_bounds or historical_bounds or diagnostic_variant:
        # Diagnostic hypothesis only: production checks and immutable captures
        # remain unchanged. The genuine PhysicalPlant.preflight still runs.
        source = (
            BASE
            / (
                "offline/validation/issue-pickup-area-20261008/source_task_old_bounds.py"
                if historical_bounds
                else "g1_can_handoff/task.py"
            )
        ).read_text()
        if diagnostic_bounds or diagnostic_variant:
            source = source.replace(
                ".255<=c[0]<=.45 and .065<=c[1]<=.29",
                ".25<=c[0]<=.51 and .06<=c[1]<=.29",
            ).replace(
                ".27<=c[0]<=.45 and .09<=c[1]<=.29", ".25<=c[0]<=.51 and .06<=c[1]<=.29"
            )
        if diagnostic_variant == "high-grasp":
            source = source.replace("(0.,.01,.02)", "(.03,.04)")
        elif diagnostic_variant == "lower-approach":
            source = source.replace("offset+[0,0,.16]", "offset+[0,0,.12]")
        namespace = {}
        exec(compile(source, "diagnostic_workspace_task", "exec"), namespace)
        planner = namespace["plan"]
    started = time.monotonic()
    with patch.object(pickup, "plan", planner):
        try:
            _, _, points, rotation = pickup.prepare(
                plant,
                lambda *args: (observation, {"source": "captured_native_scene"}),
                plant.event,
            )
            assert plant.path_preflight_ok
            np.testing.assert_array_equal(plant.ik.data.qpos, before)
            result.update(
                passed=True,
                waypoints=[p.tolist() for p in points],
                rotation=rotation.tolist(),
            )
        except (ValueError, RuntimeError, AssertionError) as exc:
            result.update(passed=False, error=str(exc))
    if not result["passed"]:
        report.update(min_table_distance_m=None, min_elbow_torso_distance_m=None)
    result.update(
        elapsed_s=time.monotonic() - started,
        clearance=report,
        events=logged,
        actuation=False,
        physical_lift_verified=False,
        diagnostic_bounds=diagnostic_bounds,
        historical_bounds=historical_bounds,
        diagnostic_variant=diagnostic_variant,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--historical-bounds", action="store_true")
    parser.add_argument("--diagnostic-bounds", action="store_true")
    parser.add_argument(
        "--diagnostic-variant", choices=("high-grasp", "lower-approach")
    )
    parser.add_argument("--case", type=int, choices=range(3))
    args = parser.parse_args()
    results = [
        run(i, args.diagnostic_bounds, args.historical_bounds, args.diagnostic_variant)
        for _ in range(args.repeat)
        for i in (range(3) if args.case is None else [args.case])
    ]
    acceptance = (
        all(r["passed"] == (r["name"] in CASES[1:]) for r in results)
        if not (
            args.diagnostic_bounds or args.historical_bounds or args.diagnostic_variant
        )
        else all(r["passed"] for r in results)
    )
    print(
        json.dumps(
            dict(
                results=results,
                acceptance_passed=acceptance,
                passed=all(r["passed"] for r in results),
            ),
            indent=2,
        )
    )
    raise SystemExit(0 if acceptance else 1)
