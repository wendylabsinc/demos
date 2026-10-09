#!/usr/bin/env python3
"""Reproducible paired full-pickup evaluation; isolated offline processes only."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cases():
    rows = [
        dict(
            id=f"nominal-{i:02d}",
            condition="fitted_noise",
            seed=1000 + i,
            scale=1.0,
            offset_xy_m=[0.0, 0.0],
        )
        for i in range(32)
    ]
    for i, (x, y) in enumerate(
        (x, y)
        for x in (-0.015, -0.005, 0.005, 0.015)
        for y in (-0.015, -0.005, 0.005, 0.015)
    ):
        rows.append(
            dict(
                id=f"placement-{i:02d}",
                condition="nearby_placement",
                seed=2000 + i,
                scale=1.0,
                offset_xy_m=[x, y],
            )
        )
    rows += [
        dict(
            id=f"stress-{i:02d}",
            condition="double_noise",
            seed=3000 + i,
            scale=2.0,
            offset_xy_m=[0.0, 0.0],
        )
        for i in range(16)
    ]
    return rows


def wilson(passed, total):
    if not total:
        return None
    z = 1.959963984540054
    p = passed / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    h = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0, c - h), min(1, c + h)]


def summarize(rows):
    groups = {}
    for condition in sorted(set(r["condition"] for r in rows)):
        group = {}
        for policy in ("legacy", "tuned"):
            selected = [
                r for r in rows if r["condition"] == condition and r["policy"] == policy
            ]
            passed = [r for r in selected if r["lift_verified"]]
            group[policy] = dict(
                trials=len(selected),
                lifts=len(passed),
                lift_fraction=len(passed) / len(selected) if selected else None,
                lift_fraction_wilson95=wilson(len(passed), len(selected)),
                errors=dict(
                    Counter(r["error"] for r in selected if not r["lift_verified"])
                ),
                failure_phases=dict(
                    Counter(
                        r.get("failure_phase")
                        for r in selected
                        if not r["lift_verified"]
                    )
                ),
                mean_success_simulated_time_s=(
                    sum(r["simulated_time_s"] for r in passed) / len(passed)
                    if passed
                    else None
                ),
                max_tracking_error_rad=max(
                    (r.get("max_tracking_error_rad", 0.0) for r in selected),
                    default=None,
                ),
                pregrasp_contact_trials=sum(
                    r.get("pregrasp_contact_steps", 0) > 0 for r in selected
                ),
                harness_exceptions=sum(bool(r.get("harness_error")) for r in selected),
            )
        groups[condition] = group
    pairs = {}
    for row in rows:
        pairs.setdefault(row["case_id"], {})[row["policy"]] = row
    counts = Counter()
    differences = []
    for pair in pairs.values():
        if set(pair) != set(("legacy", "tuned")):
            continue
        a, b = pair["legacy"]["lift_verified"], pair["tuned"]["lift_verified"]
        counts[
            (
                "both_pass"
                if a and b
                else "tuned_only" if b else "legacy_only" if a else "both_fail"
            )
        ] += 1
        if a and b:
            differences.append(
                pair["tuned"]["simulated_time_s"] - pair["legacy"]["simulated_time_s"]
            )
    return dict(
        completed=len(rows),
        groups=groups,
        paired_outcomes=dict(counts),
        mean_paired_success_duration_change_s=(
            sum(differences) / len(differences) if differences else None
        ),
        hardware_used=False,
        physical_pickup_verified=False,
        limitations=[
            "Same captured posture/calibration and approximate can dynamics across all cases.",
            "Fitted noise still has longer-lag and jump-tail mismatch; postgrasp noise is extrapolated.",
            "Equal seeds couple randomness, but different action timing changes sampled noise; not identical exposure sequences.",
            "Mixed stress/placement cases do not estimate a real-world success rate.",
            "Wilson intervals are conditional Monte Carlo intervals within this fixed simulation model.",
        ],
    )


def worker(folder, case, policy):
    if os.environ.get("HANDOFF_OFFLINE") != "1":
        raise RuntimeError("Worker requires HANDOFF_OFFLINE=1")
    from sim_runtime import run

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    setup = copy.deepcopy(manifest["setup_input"])
    for axis in range(2):
        setup["can_center_world_m"][axis] += case["offset_xy_m"][axis]
    started = time.monotonic()
    result = run(
        setup=setup,
        identity_mode="color_depth",
        position_noise="stochastic",
        noise_seed=case["seed"],
        noise_scale=case["scale"],
        planner_mode=policy,
    )
    result["campaign_case"] = dict(
        case, policy=policy, wall_time_s=time.monotonic() - started
    )
    payload = json.dumps(
        result, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    path = folder / "episodes" / f"{case['id']}-{policy}.json.gz"
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    compact = dict(
        case_id=case["id"],
        condition=case["condition"],
        seed=case["seed"],
        noise_scale=case["scale"],
        offset_xy_m=case["offset_xy_m"],
        policy=policy,
        lift_verified=result["lift_verified"],
        error=result.get("error"),
        failure_phase=result.get("failure_phase"),
        simulated_time_s=result["simulated_time_s"],
        wall_time_s=time.monotonic() - started,
        max_tracking_error_rad=result["max_tracking_error_rad"],
        command_count=result["command_count"],
        pregrasp_contact_steps=result["simulation_contact_audit"][
            "pregrasp_contact_steps"
        ],
        generated_jump_count=len(result["position_noise"]["generated_jumps"]),
        episode=path.relative_to(folder).as_posix(),
        episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        hardware_used=False,
        physical_pickup_verified=False,
    )
    path.with_suffix("").with_suffix(".summary.json").write_text(
        json.dumps(compact, indent=2) + "\n"
    )


def run_campaign(folder, jobs, timeout):
    from offline.run import integrity

    evidence = integrity()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "episodes").mkdir(exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    manifest_path = folder / "manifest.json"
    source_names = (
        "sim_runtime.py",
        "physical_core.py",
        "planner_tuning.py",
        "stochastic_position_noise.py",
        "g1_can_handoff/sim.py",
        "g1_can_handoff/task.py",
        "g1_can_handoff/stochastic_noise_profile.json",
        "g1_can_handoff/noise_aware_planner.json",
        "offline/sim_campaign.py",
    )
    manifest = dict(
        schema=1,
        cases=cases(),
        setup_input=json.loads(
            (ROOT / "offline/validation/position-noise-20261008/setup.json").read_text()
        ),
        source_sha256={
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in source_names
        },
        evidence_verified=len(evidence["files"]),
        hardware_used=False,
        physical_pickup_verified=False,
        design="32 fresh fitted-noise seeds, 16 nearby placements, 16 double-noise stress cases, each with legacy and tuned planner",
        workers=jobs,
        per_episode_wall_timeout_s=timeout,
    )
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        for key in ("cases", "setup_input", "source_sha256"):
            if prior[key] != manifest[key]:
                raise RuntimeError("Cannot resume campaign with changed " + key)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HANDOFF_", "G1_"))}
    env.update(
        HANDOFF_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT / "offline"), str(ROOT)]),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    if sys.platform != "darwin":
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")

    def evaluate(case, policy):
        summary = folder / "episodes" / f"{case['id']}-{policy}.summary.json"
        if summary.exists():
            saved = json.loads(summary.read_text())
            if saved.get("episode"):
                path = folder / saved["episode"]
                if (
                    not path.exists()
                    or hashlib.sha256(path.read_bytes()).hexdigest()
                    != saved["episode_sha256"]
                ):
                    raise RuntimeError("Missing/changed episode " + str(path))
            return saved
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(folder.resolve()),
            json.dumps(case),
            policy,
        ]
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            (folder / "logs" / f"{case['id']}-{policy}.txt").write_text(
                process.stdout + process.stderr
            )
            if process.returncode == 0 and summary.exists():
                return json.loads(summary.read_text())
            error = (
                "Worker failed: "
                + str(process.returncode)
                + "; "
                + (process.stderr or process.stdout)[-1000:]
            )
        except subprocess.TimeoutExpired as exc:
            error = f"Worker wall timeout after {timeout}s"
            (folder / "logs" / f"{case['id']}-{policy}.txt").write_text(str(exc))
        failed = dict(
            case_id=case["id"],
            condition=case["condition"],
            seed=case["seed"],
            noise_scale=case["scale"],
            policy=policy,
            lift_verified=False,
            error=error,
            failure_phase="campaign_worker",
            harness_error=True,
            simulated_time_s=0.0,
            hardware_used=False,
            physical_pickup_verified=False,
        )
        summary.write_text(json.dumps(failed, indent=2) + "\n")
        return failed

    rows = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(evaluate, case, policy)
            for case in manifest["cases"]
            for policy in ("legacy", "tuned")
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["case_id"], r["policy"]))
            (folder / "outcomes.json").write_text(json.dumps(rows, indent=2) + "\n")
            report = summarize(rows)
            report["wall_time_s"] = time.monotonic() - started
            temporary = folder / "summary.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(folder / "summary.json")
            print(
                json.dumps(
                    dict(
                        completed=len(rows),
                        total=128,
                        case=row["case_id"],
                        policy=row["policy"],
                        lift=row["lift_verified"],
                        error=row["error"],
                        seconds=round(report["wall_time_s"], 1),
                    )
                ),
                flush=True,
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        folder, case, policy = args.worker
        worker(folder, json.loads(case), policy)
    else:
        if not args.directory:
            parser.error("--directory is required")
        if not 1 <= args.jobs <= 8 or args.timeout <= 0:
            parser.error("Use 1-8 workers and a positive wall timeout")
        print(
            json.dumps(run_campaign(args.directory, args.jobs, args.timeout), indent=2)
        )
