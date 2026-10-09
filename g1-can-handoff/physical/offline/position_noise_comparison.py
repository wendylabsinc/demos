"""Compare paired noisy/ideal simulations with the preserved real XY estimates."""

import argparse
import hashlib
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def compare(directory):
    profile = json.loads(
        (ROOT / "g1_can_handoff/position_noise_profile.json").read_text()
    )
    payloads = {}
    for name in ("ideal", "approach-drift", "handover"):
        path = directory / (name + ".json")
        payloads[name] = (
            path.read_bytes()
            if path.exists()
            else gzip.decompress(path.with_suffix(".json.gz").read_bytes())
        )
    runs = {name: json.loads(payload) for name, payload in payloads.items()}
    summary = dict(
        simulation_only=True,
        physical_pickup_verified=False,
        noise_scale=1.0,
        limitations=profile["interpretation"] + " " + profile["unmeasured"],
        runs={},
    )
    for name, result in runs.items():
        cameras = [e for e in result["events"] if e["phase"] == "camera"]
        planning = [e for e in cameras if e["phase_name"] == "planning"]
        origin = np.array(planning[-1]["vision"]["center_xy_m"])
        reach = next(e for e in cameras if e["phase_name"] == "reach")
        summary["runs"][name] = dict(
            lift_verified=result["lift_verified"],
            error=result.get("error"),
            failure_phase=result["failure_phase"],
            simulated_time_s=result["simulated_time_s"],
            reach_estimate_displacement_mm=float(
                np.linalg.norm(np.array(reach["vision"]["center_xy_m"]) - origin) * 1000
            ),
            handover_estimate_displacement_mm=float(
                np.linalg.norm(
                    np.array(planning[-1]["vision"]["center_xy_m"])
                    - planning[0]["vision"]["center_xy_m"]
                )
                * 1000
            ),
            noise=result["position_noise"],
            command_count=result["command_count"],
            result_sha256=hashlib.sha256(payloads[name]).hexdigest(),
        )
    real = next(t for t in profile["traces"] if t["name"] == "approach-drift")[
        "samples"
    ]
    phases = [sample["phase"] for sample in real]
    real_delta = (
        np.array([s["center_xy_m"] for s in real]) - real[0]["center_xy_m"]
    ) * 1000
    summary["recorded_reach_estimate_displacement_mm"] = float(
        np.linalg.norm(real_delta[-1])
    )
    curves = {}
    for name in ("ideal", "approach-drift"):
        cameras = [e for e in runs[name]["events"] if e["phase"] == "camera"]
        origin = [
            e["vision"]["center_xy_m"] for e in cameras if e["phase_name"] == "planning"
        ][-1]
        curves[name] = (
            np.array(
                [
                    next(
                        e["vision"]["center_xy_m"]
                        for e in reversed(cameras)
                        if e["phase_name"] == phase
                    )
                    for phase in phases
                ]
            )
            - origin
        ) * 1000
    summary["maximum_relative_xy_replay_error_mm"] = float(
        np.max(np.linalg.norm(curves["approach-drift"] - real_delta, axis=1))
    )
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, coordinate in zip(axes, (0, 1)):
        axis.plot(
            phases,
            real_delta[:, coordinate],
            "o-",
            label="Real recorded estimate",
            color="#272727",
        )
        axis.plot(
            phases,
            curves["approach-drift"][:, coordinate],
            "x--",
            label="Sim: recorded drift",
            color="#1565c0",
        )
        axis.plot(
            phases,
            curves["ideal"][:, coordinate],
            ".:",
            label="Sim: ideal",
            color="#16885d",
        )
        axis.set_ylabel(("Forward X" if coordinate == 0 else "Left Y") + " shift (mm)")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper left")
    axes[-1].tick_params(axis="x", rotation=15)
    fig.suptitle(
        "Can position estimate drift relative to owned planning frame\nSame real-seeded scene; physical can truth stays separate"
    )
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=ROOT / "offline/validation/position-noise-20261008",
    )
    print(json.dumps(compare(parser.parse_args().directory), indent=2))
