"""Acceptance replay of all six native planner failures, without DDS or network.

Missing input fields are explicitly reported; a scene replay is not evidence of
an exact feedback replay or a physical pickup. Do not silently substitute states.
"""

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from physical_core import PhysicalPlant, calibrated_scene, SIM_TO_BRIDGE
from offline.planner_replay import fixture, clearance_auditor
from pickup import prepare

BASE = Path(__file__).resolve().parent.parent
EVIDENCE = BASE / "offline/evidence"
CASES = (
    ("orientation-timeout", EVIDENCE / "latest-planner-stop.json.gz", None),
    ("owned-staging-collision", EVIDENCE / "20261008T173157Z/events.jsonl.gz", None),
    (
        "owned-search-timeout",
        EVIDENCE / "ten-demos-20261008/trial-01/events.jsonl.gz",
        EVIDENCE / "ten-demos-20261008/trial-01/trial.json",
    ),
    (
        "additional-owned-search-timeout",
        BASE / "offline/validation/ten-demos-20261008/additional-attempt-observed.json",
        None,
    ),
    (
        "clearance-search-timeout-01",
        EVIDENCE / "clearance-demos-20261008-preserved/trial-01/events.jsonl.gz",
        EVIDENCE / "clearance-demos-20261008-preserved/trial-01/trial.json",
    ),
    (
        "clearance-search-timeout-03",
        EVIDENCE / "clearance-demos-20261008-preserved/trial-03/events.jsonl.gz",
        EVIDENCE / "clearance-demos-20261008-preserved/trial-03/trial.json",
    ),
)


def load_case(index, historical=False):
    name, path, trial_path = CASES[index]
    raw = path.read_bytes()
    if path.name == "latest-planner-stop.json.gz":
        obj = json.loads(gzip.decompress(raw))
        events = obj["events"]
        snapshot = copy.deepcopy(obj["status"]["measured"])
        missing = []
    elif path.suffix == ".json":
        obj = json.loads(raw)
        events = obj["recent_events"]
        snapshot = None
        missing = [
            "full measured body and both hand feedback",
            "exact RGB-D planning frame",
            "captured controller source",
        ]
        # Keep only the final failure's planning episode, not earlier attempts.
        last_release = max(
            (i for i, e in enumerate(events) if e["phase"] == "released"), default=-1
        )
        events = events[last_release + 1 :]
    else:
        events = [json.loads(x) for x in gzip.decompress(raw).decode().splitlines()]
        snapshot = (
            copy.deepcopy(
                json.loads(trial_path.read_text())["controller_after"]["measured"]
            )
            if trial_path
            else None
        )
        missing = [] if snapshot else ["full measured body and both hand feedback"]
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
    if snapshot is None:
        # Diagnostic scene-only fallback. Its provenance is returned, and it
        # cannot count towards exact captured-feedback acceptance.
        reference = json.loads(
            (EVIDENCE / "ten-demos-20261008/trial-03/trial.json").read_text()
        )
        snapshot = copy.deepcopy(reference["controller_before"]["measured"])
    snapshot["body"]["q"][:15] = config["reference_base_q"]
    snapshot["body"]["imu_rpy"] = config["reference_imu_rpy"]
    staging = next(
        (e for e in reversed(events) if e["phase"] == "staging_planned"), None
    )
    if staging:
        snapshot["body"]["q"][15:22] = staging["start_q"]
    owner = next(
        (e for e in reversed(events) if e["phase"] == "ownership_acquired"), None
    )
    if owner:
        snapshot["body"]["q"][22:29] = owner["right_arm_hold"]
        if "right_hand_hold" in owner:
            # The ownership event stores bridge order; snapshots use SDK
            # order. The thumb/index/middle permutation is self-inverse.
            snapshot["right"]["q"] = np.asarray(owner["right_hand_hold"])[
                SIM_TO_BRIDGE
            ].tolist()

    # Exported snapshot times belong to the old recording; receipt times are
    # rebased only offline. Native source timestamps stay in the input files.
    def read():
        result = copy.deepcopy(snapshot)
        for sample in result.values():
            sample["time"] = 100.0
        return result

    logged = []

    def event(phase, **fields):
        logged.append(dict(phase=phase, **fields))

    plant_class = PhysicalPlant
    source_path = path.parent / "source/physical_core.py"
    if historical:
        if not source_path.exists():
            raise ValueError("No captured physical_core source for this case")
        spec = importlib.util.spec_from_file_location(
            "recorded_physical_core", source_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        plant_class = mod.PhysicalPlant
    scene = calibrated_scene(config, x=center[0], y=center[1])
    plant = plant_class(
        scene,
        None,
        None,
        read,
        config,
        event,
        clock=SimpleNamespace(monotonic=lambda: 100.0),
    )
    observation = SimpleNamespace(
        center=np.asarray(center),
        table_height=config["table_height_m"],
        timestamp=time.monotonic(),
    )
    plant.observe_can(observation)
    provenance = dict(
        name=name,
        input_file=str(path.relative_to(BASE)),
        input_sha256=hashlib.sha256(raw).hexdigest(),
        original_failure=next(
            e["error"] for e in reversed(events) if e["phase"] == "primary_failure"
        ),
        missing_fields=missing,
        captured_feedback_available=not missing,
        feedback_source=(
            "captured terminal snapshot with recorded calibration/base and staging start"
            if not missing
            else "scene-only diagnostic with explicit fallback feedback"
        ),
        historical_source=str(source_path.relative_to(BASE)) if historical else None,
    )
    return plant, observation, logged, provenance


def run(index, historical=False):
    started = time.monotonic()
    try:
        plant, observation, logged, provenance = load_case(index, historical)
        report = clearance_auditor(plant)
        original = plant.ik.data.qpos.copy()
        try:
            _, _, points, rotation = prepare(
                plant,
                lambda *args: (observation, {"source": "captured_native_scene"}),
                plant.event,
            )
            assert plant.path_preflight_ok
            np.testing.assert_array_equal(plant.ik.data.qpos, original)
            result = dict(
                passed=True,
                waypoints=[p.tolist() for p in points],
                rotation=rotation.tolist(),
            )
        except (RuntimeError, ValueError, AssertionError) as exc:
            result = dict(passed=False, error=str(exc))
        result.update(provenance, clearance=report, events=logged)
    except (RuntimeError, ValueError, StopIteration) as exc:
        result = dict(name=CASES[index][0], passed=False, fixture_error=str(exc))
    result.update(
        elapsed_s=time.monotonic() - started,
        actuation=False,
        physical_lift_verified=False,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(6))
    parser.add_argument("--historical-source", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    results = [
        run(index, args.historical_source)
        for repeat in range(args.repeat)
        for index in (range(6) if args.case is None else [args.case])
    ]
    print(
        json.dumps(
            dict(
                results=results,
                passed=all(r["passed"] for r in results),
                captured_feedback_cases=sum(
                    r.get("captured_feedback_available", False) for r in results
                ),
            ),
            indent=2,
        )
    )
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)
