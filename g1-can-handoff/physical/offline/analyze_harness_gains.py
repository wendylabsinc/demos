"""Compare verified physical replays of an identical saved joint sequence."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(directory):
    directory = Path(directory)
    results = []
    command_sha = None
    for path in sorted(directory.glob("physical-p*-*.json")):
        value = json.loads(path.read_text())
        metrics = value.get("metrics")
        if metrics is None or value.get("state") != "terminal":
            continue
        archive = path.with_suffix(".events.jsonl.gz")
        raw = archive.read_bytes()
        if hashlib.sha256(raw).hexdigest() != value["export"]["sha256"]:
            raise ValueError("Evidence archive checksum mismatch")
        events = [json.loads(row) for row in gzip.decompress(raw).splitlines()]
        samples = [row for row in events if row["phase"] == "joint_replay_sample"]
        if len(samples) != 50 + metrics["commands_replayed"] + 100:
            raise ValueError("Incomplete replay sample series")
        trajectory = samples[50 : 50 + metrics["commands_replayed"]]
        q = np.asarray([s["commanded_left_q"] for s in trajectory], dtype=np.float64)
        digest = hashlib.sha256(q.tobytes()).hexdigest()
        if command_sha is None:
            command_sha = digest
        if command_sha != digest:
            raise ValueError(
                "Physical trials did not receive the same requested joint sequence"
            )
        errors = np.asarray([s["error_rad"] for s in trajectory])
        rms = float(np.sqrt(np.mean(errors**2)))
        if not np.isclose(rms, metrics["rms_error_rad"], atol=1e-12):
            raise ValueError("Recorded metrics differ from command/feedback samples")
        last = samples[-1]
        wire_error = np.asarray(last["sdk_reference"]["q"][:7]) - np.asarray(
            last["commanded_left_q"]
        )
        results.append(
            dict(
                path=str(path),
                kp=value["kp"],
                kd=3.0,
                metrics=metrics,
                error=value.get("error"),
                phase_after=value["after"]["status"]["phase"],
                returned_to_origin=bool(
                    (value["after"]["status"].get("demo_result") or {}).get(
                        "returned_to_measured_origin"
                    )
                ),
                sdk_final_reference_error_rad=float(np.max(np.abs(wire_error))),
                fixture_sha256=value["fixture_sha256"],
                events_sha256=value["export"]["sha256"],
                trajectory_time_s=[
                    s["sample_monotonic_s"] - trajectory[0]["sample_monotonic_s"]
                    for s in trajectory
                ],
                trajectory_error_rad=errors.tolist(),
                all_sample_time_s=[
                    s["sample_monotonic_s"] - samples[0]["sample_monotonic_s"]
                    for s in samples
                ],
                all_shoulder_roll_target=[s["commanded_left_q"][1] for s in samples],
                all_shoulder_roll_measured=[s["measured_left_q"][1] for s in samples],
            )
        )
    results.sort(key=lambda r: (r["kp"], r["path"]))
    baseline = next(r for r in results if r["kp"] == 80.0)
    for result in results:
        result["rms_improvement_percent"] = 100 * (
            1
            - result["metrics"]["rms_error_rad"] / baseline["metrics"]["rms_error_rad"]
        )
        result["final_error_improvement_percent"] = 100 * (
            1
            - result["metrics"]["final_max_error_rad"]
            / baseline["metrics"]["final_max_error_rad"]
        )
    report = dict(
        physical_joint_replay=True,
        physical_pickup_verified=False,
        same_left_joint_requests=True,
        command_sequence_sha256=command_sha,
        all_feedback_and_command_events_checksum_verified=True,
        results=[
            {
                k: v
                for k, v in r.items()
                if not k.startswith(
                    (
                        "trajectory_time",
                        "trajectory_error",
                        "all_sample",
                        "all_shoulder",
                    )
                )
            }
            for r in results
        ],
        limitations=[
            "No can was grasped; fingers were not commanded.",
            "Right arm hold and starting posture are measured anew for each trial.",
            "Requested 50 Hz joint sequence is identical; physical timing jitter remains.",
            "Harness support changes base dynamics relative to a standing pickup.",
        ],
    )
    (directory / "physical-gain-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
    for result in results:
        label = f"P={result['kp']:g}, run {int(Path(result['path']).stem.rsplit('-',1)[-1])}"
        error = np.asarray(result["trajectory_error_rad"])
        axes[0].plot(result["trajectory_time_s"], error[:, 1], label=label, alpha=0.85)
        axes[1].scatter(result["kp"], result["metrics"]["rms_error_rad"], s=45)
        axes[2].scatter(
            result["kp"], result["metrics"]["final_window_max_error_rad"], s=45
        )
    axes[0].set(
        xlabel="Time through recorded reach (s)",
        ylabel="Shoulder roll: measured − requested (rad)",
        title="Same 418 joint requests",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="Shoulder/elbow proportional gain",
        ylabel="Trajectory RMS error (rad)",
        title="Measured tracking",
    )
    axes[2].axhline(0.04, color="red", ls="--", label="Existing settling limit")
    axes[2].set(
        xlabel="Shoulder/elbow proportional gain",
        ylabel="Final 10-sample maximum error (rad)",
        title="Measured final settling",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "NX2 harness replay · D=3; wrists P=40/D=1.5 · physical results", fontsize=13
    )
    fig.savefig(directory / "physical-gain-comparison.png", dpi=160)
    plt.close(fig)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    a = p.parse_args()
    report = analyze(a.directory)
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "kp",
                        "metrics",
                        "error",
                        "rms_improvement_percent",
                        "final_error_improvement_percent",
                        "sdk_final_reference_error_rad",
                    )
                }
                for r in report["results"]
            ],
            indent=2,
        )
    )
