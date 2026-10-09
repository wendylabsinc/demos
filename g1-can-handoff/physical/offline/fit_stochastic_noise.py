"""Chronological corpus, session-held-out fitting and audit for estimate variability.

Only observed pregrasp XY is fitted: no independent can truth is available.
Reduced support is an occlusion proxy, not an annotated visibility label.
"""

import argparse
import bisect
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("stationary", "moving", "occluded")
MAX_GAP = 2.0
JUMP_FLOOR = 0.003


def source_logs(root):
    for path in sorted(root.rglob("*events.jsonl.gz")):
        yield str(path.relative_to(ROOT)), gzip.decompress(path.read_bytes())
    for path in sorted(root.rglob("*.tar.gz")):
        with tarfile.open(path) as archive:
            members = [m for m in archive if m.name.endswith("events.jsonl.gz")]
            for member in members:
                yield str(path.relative_to(ROOT)) + ":" + member.name, gzip.decompress(
                    archive.extractfile(member).read()
                )
    path = root / "trial-events.json.gz"
    if path.exists():
        for name, trial in json.loads(gzip.decompress(path.read_bytes())).items():
            yield str(path.relative_to(ROOT)) + ":" + name, (
                "\n".join(json.dumps(r, sort_keys=True) for r in trial["events"]) + "\n"
            ).encode()


def normalize(rows, identity, sources):
    rows = sorted(rows, key=lambda r: r.get("unix_ns", 0))
    targets = [
        r
        for r in rows
        if r.get("phase") == "target"
        and len(r.get("measured_arm", r.get("arm", []))) == 14
    ]
    target_times = np.array([r["monotonic_ns"] / 1e9 for r in targets])
    poses = [r for r in rows if r.get("phase") == "exposure_pose"]
    pose_times = [r["unix_ns"] for r in poses]
    closes = [r["unix_ns"] for r in rows if r.get("phase") == "close"]
    cutoff = min(closes) if closes else float("inf")
    samples = []
    seen = set()
    excluded = {}

    def reject(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for row in rows:
        if row.get("phase") != "perception_result":
            continue
        geometry = row.get("geometry") or {}
        if row.get("error") or "center_xy_m" not in geometry:
            reject("invalid_or_absent_geometry")
            continue
        if (
            row["unix_ns"] >= cutoff
            or row.get("phase_name") in ("pre-lift", "lift-verification")
            or row.get("phase_name", "").startswith("hold-verification")
        ):
            reject("postgrasp_actual_movement_expected")
            continue
        source = row.get("source") or {}
        frame = (
            row.get("frame_id"),
            geometry.get("image_sha256") or source.get("image_sha256"),
        )
        if frame in seen:
            reject("repeated_frame")
            continue
        seen.add(frame)
        center = np.asarray(geometry["center_xy_m"], dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            reject("nonfinite_geometry")
            continue
        stamp = source.get("timestamp")
        if stamp is None and poses:
            i = bisect.bisect_right(pose_times, row["unix_ns"]) - 1
            if i >= 0 and row["unix_ns"] - pose_times[i] < 50_000_000:
                stamp = poses[i]["frame_timestamp"]
        stamp_source = (
            "source_exposure" if stamp is not None else "receipt_minus_exposure_age"
        )
        if stamp is None:
            age = row.get("exposure_age_ms")
            if age is None:
                reject("exposure_time_missing")
                continue
            stamp = row["monotonic_ns"] / 1e9 - age / 1000
        speed = None
        if len(targets) > 1:
            a = int(np.searchsorted(target_times, stamp - 0.2))
            b = int(np.searchsorted(target_times, stamp + 0.05))
            if b - a >= 2:
                dt = target_times[b - 1] - target_times[a]
                if dt > 0:
                    speed = float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    targets[b - 1].get(
                                        "measured_arm", targets[b - 1]["arm"]
                                    )
                                )
                                - np.asarray(
                                    targets[a].get("measured_arm", targets[a]["arm"])
                                )
                            )
                        )
                        / dt
                    )
        motion_source = (
            "measured"
            if speed is not None and all("measured_arm" in r for r in targets[a:b])
            else "commanded_proxy" if speed is not None else "no_recent_arm_samples"
        )
        phase = row.get("phase_name")
        regime = "moving" if (speed is not None and speed > 0.04) else "stationary"
        samples.append(
            dict(
                t=float(stamp),
                unix_ns=row["unix_ns"],
                frame_id=row.get("frame_id"),
                xy=center.tolist(),
                pixels=geometry.get("pixels", 0),
                phase=phase,
                regime=regime,
                arm_speed_rad_s=speed,
                motion_source=motion_source,
                detector=geometry.get("detector"),
                fit_error_m=geometry.get("fit_error_m"),
                exposure_time_source=stamp_source,
                can_stationarity="pregrasp_candidate_unverified",
            )
        )
    samples.sort(key=lambda r: r["t"])
    support = float(np.median([s["pixels"] for s in samples[:5]])) if samples else 0.0
    for sample in samples:
        if support > 0 and sample["pixels"] / support < 0.65:
            sample["regime"] = "occluded"
        sample["support_fraction"] = sample["pixels"] / support if support else None
    return dict(
        id=identity,
        sources=sources,
        samples=samples,
        excluded=excluded,
        errors=[
            dict(unix_ns=r.get("unix_ns"), error=r.get("error"))
            for r in rows
            if r.get("phase") == "primary_failure"
        ],
    )


def corpus(root):
    logs = []
    by_event = {}
    audit = []
    for source, payload in source_logs(root):
        digest = hashlib.sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        keys = {
            (r.get("unix_ns"), r.get("frame_id"))
            for r in rows
            if r.get("phase") == "perception_result"
        }
        audit.append(
            dict(
                source=source,
                events_sha256=digest,
                event_count=len(rows),
                position_results=len(keys),
            )
        )
        if not keys:
            continue
        overlapping = {by_event[k] for k in keys if k in by_event}
        if overlapping:
            group = min(overlapping)
            for other in overlapping - {group}:
                logs[group]["rows"] += logs[other]["rows"]
                logs[group]["sources"] += logs[other]["sources"]
                for key, value in list(by_event.items()):
                    if value == other:
                        by_event[key] = group
                logs[other] = None
            logs[group]["rows"] += rows
            logs[group]["sources"].append(source)
        else:
            group = len(logs)
            logs.append(dict(rows=rows, sources=[source]))
        for key in keys:
            by_event[key] = group
    sessions = []
    for log in logs:
        if log is None:
            continue
        unique = {
            json.dumps(r, sort_keys=True, separators=(",", ":")): r for r in log["rows"]
        }
        rows = sorted(unique.values(), key=lambda r: r.get("unix_ns", 0))
        identity = hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()[:16]
        sessions.append(normalize(rows, identity, sorted(set(log["sources"]))))
    return dict(
        schema=1,
        sessions=sorted(sessions, key=lambda s: s["id"]),
        source_audit=audit,
        limitations="No independent can ground truth or operator stationary labels. All pregrasp data is combined estimate variability; postgrasp data is excluded. Arm speed (measured when available; commanded otherwise) and support ratio provide observable regime proxies.",
    )


def segments(sessions):
    for session in sessions:
        group = []
        for sample in session["samples"]:
            if group and (
                sample["t"] - group[-1]["t"] > MAX_GAP or sample["t"] <= group[-1]["t"]
            ):
                if len(group) > 1:
                    yield group
                group = []
            group.append(sample)
        if len(group) > 1:
            yield group


def psd(matrix):
    eigen, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * np.maximum(eigen, 1e-14)) @ vectors.T


def fit(sessions):
    groups = list(segments(sessions))
    deltas = np.array(
        [np.asarray(b["xy"]) - a["xy"] for g in groups for a, b in zip(g, g[1:])]
    )
    if len(deltas) < 10:
        raise ValueError(
            "Insufficient chronological pairs for a fitted noise distribution"
        )
    lengths = np.linalg.norm(deltas, axis=1)
    threshold = max(
        JUMP_FLOOR,
        float(
            np.median(lengths)
            + 6 * 1.4826 * np.median(np.abs(lengths - np.median(lengths)))
        ),
    )
    pairs = {name: [] for name in (*REGIMES, "pooled")}
    jumps = {name: [] for name in (*REGIMES, "pooled")}
    durations = {name: 0.0 for name in (*REGIMES, "pooled")}
    for group in groups:
        xy = np.array([s["xy"] for s in group])
        t = np.array([s["t"] for s in group])
        large = np.linalg.norm(np.diff(xy, axis=0), axis=1) > threshold
        blocks = np.r_[0, np.cumsum(large)]
        for i, (a, b) in enumerate(zip(group, group[1:])):
            regime = b["regime"]
            dt = b["t"] - a["t"]
            for key in ("pooled", regime):
                durations[key] += dt
                if large[i]:
                    jumps[key].append((xy[i + 1] - xy[i]).tolist())
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dt = t[j] - t[i]
                if dt > 10:
                    break
                if blocks[i] != blocks[j]:
                    continue
                delta = xy[j] - xy[i]
                pairs["pooled"].append((dt, delta))
                if all(s["regime"] == group[i]["regime"] for s in group[i : j + 1]):
                    pairs[group[i]["regime"]].append((dt, delta))
    bins = np.array([0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 10.01])

    def fitted(name):
        data = pairs[name]
        if len(data) < 30:
            return None
        times = np.array([p[0] for p in data])
        changes = np.array([p[1] for p in data])
        stats = []
        for low, high in zip(bins, bins[1:]):
            selected = (times >= low) & (times < high)
            if selected.sum() >= 8:
                # The non-jump second moment, not a truncated median, fits variance.
                stats.append(
                    (
                        float(np.mean(times[selected])),
                        float(np.mean(np.sum(changes[selected] ** 2, axis=1)) / 2),
                        int(selected.sum()),
                    )
                )
        if len(stats) < 3:
            return None
        lag = np.array([s[0] for s in stats])
        value = np.array([s[1] for s in stats])
        weight = np.sqrt(np.array([s[2] for s in stats]))
        candidates = []
        for tau in np.geomspace(0.05, 120, 90):
            design = np.c_[np.ones(len(lag)), 1 - np.exp(-lag / tau)]
            variance, _ = nnls(design * weight[:, None], value * weight)
            loss = float(np.sum(((design @ variance - value) * weight) ** 2))
            candidates.append((loss, tau, variance))
        loss, tau, variance = min(candidates, key=lambda r: r[0])
        short = changes[times < 0.4]
        long = changes[times > 1.6]

        def direction(vectors):
            matrix = (
                psd(vectors.T @ vectors / max(1, len(vectors)))
                if len(vectors)
                else np.eye(2)
            )
            return matrix / np.trace(matrix)

        jitter = direction(short) * variance[0]
        drift = direction(long) * variance[1]
        jump_rows = jumps[name]
        return dict(
            jitter_cov_m2=jitter.tolist(),
            drift_cov_m2=drift.tolist(),
            drift_tau_s=float(tau),
            jump_rate_hz=len(jump_rows) / durations[name] if durations[name] else 0.0,
            jump_vectors_m=jump_rows,
            observed_duration_s=durations[name],
            jump_count=len(jump_rows),
            pair_count=len(data),
            variogram_bins=stats,
            tau_at_search_boundary=bool(tau in (0.05, 120)),
            fit_loss=loss,
            fallback=None,
        )

    pooled = fitted("pooled")
    if pooled is None:
        raise ValueError(
            "Chronological recordings do not identify jitter and drift separately"
        )
    regimes = {}
    for name in REGIMES:
        fitted_regime = fitted(name)
        if fitted_regime is None:
            fitted_regime = dict(
                pooled,
                fallback="pooled: fewer than three populated lag bins",
                observed_duration_s=durations[name],
                jump_count=len(jumps[name]),
            )
        regimes[name] = fitted_regime
    session_ratios = []
    for session in sessions:
        actual = []
        predicted = []
        for group in segments([session]):
            for a, b in zip(group, group[1:]):
                delta = np.asarray(b["xy"]) - a["xy"]
                dt = b["t"] - a["t"]
                if np.linalg.norm(delta) > threshold:
                    continue
                parameter = regimes[b["regime"]]
                r = float(np.trace(parameter["jitter_cov_m2"]))
                v = float(np.trace(parameter["drift_cov_m2"]))
                tau = parameter["drift_tau_s"]
                expected = 2 * r + 2 * v * (
                    dt / tau
                    if parameter["tau_at_search_boundary"] and tau >= 120
                    else 1 - np.exp(-dt / tau)
                )
                actual.append(float(delta @ delta))
                predicted.append(expected)
        if len(actual) >= 10 and sum(predicted) > 0:
            session_ratios.append(
                dict(
                    session=session["id"],
                    scale=float(np.sqrt(sum(actual) / sum(predicted))),
                    pairs=len(actual),
                )
            )
    sigma = (
        float(np.std(np.log([entry["scale"] for entry in session_ratios]), ddof=1))
        if len(session_ratios) > 1
        else 0.0
    )
    return dict(
        schema=1,
        regimes=regimes,
        session_scale_log_sigma=sigma,
        session_scale_fit=session_ratios,
        session_scale_distribution="lognormal with log mean -sigma^2, preserving expected squared scale of 1",
        jump_threshold_m=threshold,
        fit_session_ids=sorted(s["id"] for s in sessions),
        training_observations=sum(len(s["samples"]) for s in sessions),
        max_contiguous_gap_s=MAX_GAP,
        model="Gaussian frame jitter + continuous-time OU drift + persistent compound-Poisson jump offsets",
        limitations="Fitted pregrasp estimate variability, not sensor accuracy; unverified can stationarity. Regimes use measured or commanded arm speed >0.04 rad/s and support <65% of initial support as proxies. Absolute initial bias, target loss, Z noise, latency and postgrasp behavior are not identified.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-json-gz",
        type=Path,
        help="Refit from the published chronological corpus instead of rescanning raw archives",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_directory
    folder.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(gzip.decompress(args.corpus_json_gz.read_bytes()))
        if args.corpus_json_gz
        else corpus(ROOT / "offline")
    )
    packed = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    (folder / "corpus.json.gz").write_bytes(gzip.compress(packed, mtime=0))
    profile = fit(data["sessions"])
    profile["corpus_sha256"] = hashlib.sha256(packed).hexdigest()
    (folder / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                sessions=len(data["sessions"]),
                observations=profile["training_observations"],
                sources=len(data["source_audit"]),
                regimes={
                    k: dict(
                        jitter_rms_mm=float(
                            np.sqrt(np.trace(v["jitter_cov_m2"])) * 1000
                        ),
                        drift_rms_mm=float(np.sqrt(np.trace(v["drift_cov_m2"])) * 1000),
                        tau=v["drift_tau_s"],
                        jumps=v["jump_count"],
                        duration=v["observed_duration_s"],
                        fallback=v["fallback"],
                    )
                    for k, v in profile["regimes"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
