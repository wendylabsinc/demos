"""Paired, seeded MuJoCo pickup runs at baseline and doubled tracking limits.

This command constructs only SimulationIO. It neither connects to Wendy nor
imports a robot SDK writer. Run from the physical project with python -m.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import time

from simulation_tracking_limits import profile

CASES = {
    "ideal": dict(position_noise="none"),
    "arms-down": dict(
        position_noise="none",
        initial_arm_q14=[
            0.246,
            0.265,
            -0.092,
            0.820,
            0.0029,
            0.00794,
            -0.00446,
            0.265,
            -0.286,
            0.089,
            0.798,
            -0.00014,
            0.0102,
            0.00156,
        ],
    ),
    "recorded-handover": dict(position_noise="recorded", noise_trace="handover"),
    "recorded-approach-drift": dict(
        position_noise="recorded", noise_trace="approach-drift"
    ),
    "stochastic-7": dict(position_noise="stochastic", noise_seed=7),
    "stochastic-23": dict(position_noise="stochastic", noise_seed=23),
    "stale-align": dict(
        position_noise="none", fault=dict(phase="align", kind="stale", age_s=0.6)
    ),
    "missing-can-align": dict(
        position_noise="none", fault=dict(phase="align", kind="missing_target")
    ),
}


def one(case, name, output):
    from sim_runtime import run

    started = time.monotonic()
    # Do not create gigabytes of planning snapshots for this tolerance-only
    # experiment. The full event/command trace and source hashes are retained.
    result = run(
        height=0.762,
        identity_mode="color_depth",
        planner_mode="tuned",
        tracking_profile=name,
        **CASES[case],
    )
    result.update(
        case=case, wall_seconds=time.monotonic() - started, hardware_used=False
    )
    path = Path(output) / f"{case}-{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return dict(
        case=case,
        profile=name,
        lift_verified=result["lift_verified"],
        error=result.get("error"),
        max_tracking_error_rad=result["max_tracking_error_rad"],
        simulated_time_s=result["simulated_time_s"],
        wall_seconds=result["wall_seconds"],
        command_count=result["command_count"],
        failure_phase=result.get("failure_phase"),
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def saved_reach(trial, output):
    from offline.replay_arm_tracking import load, replay

    snapshot, commands, physical = load(trial)
    results = [
        replay(snapshot, commands, 150.0, tracking_profile=name)
        for name in ("baseline", "double")
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    traces = gzip.compress(json.dumps(results, separators=(",", ":")).encode(), mtime=0)
    (output / "saved-reach.traces.json.gz").write_bytes(traces)
    recorded_error = max(abs(v) for v in physical["error_rad"]) if physical else None
    report = dict(
        hardware_used=False,
        simulation_only=True,
        trial=str(trial),
        trial_sha256=hashlib.sha256(Path(trial).read_bytes()).hexdigest(),
        same_recorded_commands_initial_state_feedforward_and_gains=True,
        results=[{k: v for k, v in r.items() if k != "samples"} for r in results],
        recorded_real_final_error_rad=recorded_error,
        recorded_real_settle_gate_only=(
            None
            if recorded_error is None
            else {
                name: recorded_error < profile(name)["arm_settle_error_rad"]
                for name in ("baseline", "double")
            }
        ),
        trace_file="saved-reach.traces.json.gz",
        trace_sha256=hashlib.sha256(traces).hexdigest(),
        limitations=[
            "Recorded terminal error is a gate comparison, not a counterfactual physical continuation.",
            "Saved reach is from an earlier trial; latest 0.15 rad failure recording is not available locally.",
            "Simulation contact and factory balance differ from hardware.",
        ],
    )
    (output / "saved-reach.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(output, cases=tuple(CASES), workers=2, trial=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(one, case, name, str(output)): (case, name)
            for case in cases
            for name in ("baseline", "double")
        }
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(json.dumps(row), flush=True)
    rows.sort(key=lambda r: (r["case"], r["profile"]))
    totals = {
        name: dict(
            runs=sum(r["profile"] == name for r in rows),
            verified_lifts=sum(
                r["profile"] == name and r["lift_verified"] for r in rows
            ),
        )
        for name in ("baseline", "double")
    }
    report = dict(
        schema="wendy.g1.sim-tracking-threshold-comparison.v1",
        hardware_used=False,
        simulation_only=True,
        physical_pickup_verified=False,
        profiles={name: profile(name) for name in ("baseline", "double")},
        totals=totals,
        runs=rows,
        unchanged=[
            "collision clearance",
            "joint position limits",
            "torque caps and gains",
            "joint velocity/acceleration limits",
            "feedback and camera freshness",
            "perception displacement limits",
            "base/calibration checks",
            "settling speed and timeout",
            "grasp and lift verification",
        ],
        limitations=[
            "Supported-base dynamics and simulated contact are approximate.",
            "Two noise traces replay earlier perception failures; this is not an exact replay of the latest arm-error stop.",
            "Passing a larger tolerance does not establish more accurate tracking or qualify intentional physical table contact.",
        ],
    )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if trial is not None:
        saved_reach(trial, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="offline/validation/tracking-limits-2x-20261009"
    )
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--saved-trial",
        help="Additionally replay this archived real reach with both profiles at P=150",
    )
    args = parser.parse_args()
    report = compare(
        args.output, tuple(args.case or CASES), args.workers, args.saved_trial
    )
    print(json.dumps(report["totals"], indent=2))
