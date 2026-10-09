"""Leave entire sessions out; sample fresh sequences at their exposure cadence."""

import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from offline.fit_stochastic_noise import fit, segments, MAX_GAP
from stochastic_position_noise import StochasticPositionNoise


def metrics(groups, threshold, gate_groups=None):
    steps = []
    duration = 0.0
    jumps = 0
    displacement = []
    phase_stops = []
    correlations = {".2s": [], "1s": [], "5s": []}
    for group in groups:
        t = np.array([r["t"] for r in group])
        xy = np.array([r["xy"] for r in group])
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        steps.extend(d * 1000)
        duration += t[-1] - t[0]
        jumps += int(np.count_nonzero(d > threshold))
        displacement.extend(np.linalg.norm(xy - xy[0], axis=1) * 1000)
        residual = xy - xy.mean(axis=0)
        for label, lag in ((".2s", 0.2), ("1s", 1.0), ("5s", 5.0)):
            for i in range(len(t) - 1):
                j = int(np.searchsorted(t, t[i] + lag))
                if j < len(t) and abs(t[j] - t[i] - lag) < max(0.05, lag * 0.2):
                    correlations[label].append((residual[i], residual[j]))
        baseline = xy[0]
        checked = False
        stopped = False
        for row, center in zip(group, xy):
            if row["phase"] == "planning":
                baseline = center
            if row["phase"] in (
                "plan-refresh",
                "pre-open",
                "pre-stage",
                "approach",
                "align",
                "reach",
                "pre-close",
            ):
                checked = True
                if np.linalg.norm(center - baseline) > 0.01:
                    stopped = True
        if checked:
            phase_stops.append(stopped)
    if gate_groups is not None:
        phase_stops = []
        for group in gate_groups:
            baseline = None
            checked = False
            stopped = False
            for row in group:
                center = np.asarray(row["xy"])
                if row["phase"] == "planning":
                    baseline = center
                if baseline is not None and row["phase"] in (
                    "plan-refresh",
                    "pre-open",
                    "pre-stage",
                    "approach",
                    "align",
                    "reach",
                    "pre-close",
                ):
                    checked = True
                    if np.linalg.norm(center - baseline) > 0.01:
                        stopped = True
                        break
            if checked:
                phase_stops.append(stopped)

    def quantiles(values):
        return (
            dict(
                zip(
                    ("p50", "p90", "p95", "p99"),
                    np.percentile(values, [50, 90, 95, 99]).tolist(),
                )
            )
            if values
            else None
        )

    correlation = {}
    for label, pairs in correlations.items():
        if len(pairs) >= 8:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
            correlation[label] = (
                float(np.sum(a * b) / denominator) if denominator else None
            )
        else:
            correlation[label] = None
    return dict(
        step_displacement_mm=quantiles(steps),
        baseline_displacement_mm=quantiles(displacement),
        correlation_over_time=correlation,
        large_step_rate_hz=jumps / duration if duration else None,
        large_step_count=jumps,
        observed_duration_s=duration,
        phase_boundary_stop_fraction=(
            float(np.mean(phase_stops)) if phase_stops else None
        ),
        checked_segments=len(phase_stops),
        censoring="Only recorded phase boundaries; recordings terminate on other failures. This is not full-demo success probability.",
    )


def generate(groups, profile, seed):
    generated = []
    session_scale = StochasticPositionNoise(seed=seed, profile=profile).session_scale
    for index, group in enumerate(groups):
        generator = StochasticPositionNoise(
            seed=seed * 1000 + index, profile=profile, session_scale=session_scale
        )
        samples = []
        for i, row in enumerate(group):
            generator.advance(
                row["t"] - group[0]["t"],
                moving=row["regime"] == "moving",
                occluded=row["regime"] == "occluded",
            )
            offset, _ = generator.offset(row["phase"], i + 1)
            samples.append(dict(row, xy=offset.tolist()))
        generated.append(samples)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=64)
    args = parser.parse_args()
    folder = args.directory
    data = json.loads(gzip.decompress((folder / "corpus.json.gz").read_bytes()))
    sessions = data["sessions"]
    reports = []
    for held in sessions:
        groups = list(segments([held]))
        if sum(len(g) for g in groups) < 10:
            if len(held["samples"]) >= 2:
                profile = fit([s for s in sessions if s["id"] != held["id"]])
                outcomes = []
                for seed in range(args.seeds):
                    generated = generate([held["samples"]], profile, seed + 1)
                    outcomes.append(
                        metrics([], profile["jump_threshold_m"], generated)[
                            "phase_boundary_stop_fraction"
                        ]
                    )
                values = [x for x in outcomes if x is not None]
                reports.append(
                    dict(
                        session=held["id"],
                        validated=False,
                        reason="Fewer than ten consecutive usable observations; phase checks only",
                        fit_session_ids=profile["fit_session_ids"],
                        real_phase_boundary_stop_fraction=metrics(
                            [], profile["jump_threshold_m"], [held["samples"]]
                        )["phase_boundary_stop_fraction"],
                        generated_phase_boundary_stop_probability=(
                            float(np.mean(values)) if values else None
                        ),
                    )
                )
            else:
                reports.append(
                    dict(
                        session=held["id"],
                        validated=False,
                        reason="Insufficient observations even for a phase-check comparison",
                    )
                )
            continue
        training = [s for s in sessions if s["id"] != held["id"]]
        profile = fit(training)
        generated = [
            generate([held["samples"]], profile, seed + 1)[0]
            for seed in range(args.seeds)
        ]
        real = metrics(groups, profile["jump_threshold_m"], [held["samples"]])
        sampled = [
            metrics(list(segments([dict(samples=g)])), profile["jump_threshold_m"], [g])
            for g in generated
        ]

        def envelope(key):
            values = [s[key] for s in sampled if s[key] is not None]
            return (
                np.percentile(values, [5, 50, 95], axis=0).tolist() if values else None
            )

        report = dict(
            session=held["id"],
            validated=True,
            held_out_sources=held["sources"],
            fit_session_ids=profile["fit_session_ids"],
            real=real,
            generated_step_percentiles_mm={
                q: np.percentile(
                    [m["step_displacement_mm"][q] for m in sampled], [5, 50, 95]
                ).tolist()
                for q in ("p50", "p90", "p95", "p99")
            },
            generated_baseline_percentiles_mm={
                q: np.percentile(
                    [m["baseline_displacement_mm"][q] for m in sampled], [5, 50, 95]
                ).tolist()
                for q in ("p50", "p90", "p95", "p99")
            },
            generated_correlation={
                k: (
                    np.percentile(
                        [
                            m["correlation_over_time"][k]
                            for m in sampled
                            if m["correlation_over_time"][k] is not None
                        ],
                        [5, 50, 95],
                    ).tolist()
                    if real["correlation_over_time"][k] is not None
                    else None
                )
                for k in real["correlation_over_time"]
            },
            generated_large_step_rate_hz=envelope("large_step_rate_hz"),
            generated_phase_boundary_stop_fraction=envelope(
                "phase_boundary_stop_fraction"
            ),
        )
        report["generated_phase_boundary_stop_probability"] = (
            float(
                np.mean(
                    [
                        m["phase_boundary_stop_fraction"]
                        for m in sampled
                        if m["phase_boundary_stop_fraction"] is not None
                    ]
                )
            )
            if any(m["phase_boundary_stop_fraction"] is not None for m in sampled)
            else None
        )
        report["p95_step_within_generated_90pct_interval"] = (
            report["generated_step_percentiles_mm"]["p95"][0]
            <= real["step_displacement_mm"]["p95"]
            <= report["generated_step_percentiles_mm"]["p95"][2]
        )
        reports.append(report)
        print(
            json.dumps(
                dict(
                    session=held["id"],
                    observations=sum(len(g) for g in groups),
                    real_p95_step=real["step_displacement_mm"]["p95"],
                    generated_p95=report["generated_step_percentiles_mm"]["p95"],
                    within=report["p95_step_within_generated_90pct_interval"],
                )
            ),
            flush=True,
        )
    result = dict(
        schema=1,
        method="leave-one-session-out; overlapping exports merged before fitting",
        seeds=args.seeds,
        data_limitations=data["limitations"],
        sessions=reports,
        hardware_used=False,
    )
    (folder / "held-out.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
