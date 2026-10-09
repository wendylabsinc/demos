"""Extract pregrasp estimate drift from recorded phase observations, without DDS."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

PHASES = {
    "planning",
    "plan-refresh",
    "pre-open",
    "pre-stage",
    "approach",
    "align",
    "reach",
}


def extract(path, name):
    with tarfile.open(path) as archive:
        payload = archive.extractfile("events.jsonl.gz").read()
    events = gzip.decompress(payload)
    samples = []
    seen = set()
    for line in events.splitlines():
        row = json.loads(line)
        if (
            row.get("phase") != "perception_result"
            or row.get("phase_name") not in PHASES
            or row.get("error")
        ):
            continue
        frame = row["frame_id"]
        if frame in seen:
            continue
        seen.add(frame)
        samples.append(
            dict(
                phase=row["phase_name"],
                frame_id=frame,
                unix_ns=row["unix_ns"],
                center_xy_m=row["geometry"]["center_xy_m"],
                exposure_age_ms=row.get("exposure_age_ms"),
                fit_error_m=row["geometry"]["fit_error_m"],
            )
        )
    if not samples or samples[0]["phase"] != "planning":
        raise ValueError("Recorded trace must start at planning")
    return dict(
        name=name,
        source_archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_events_sha256=hashlib.sha256(events).hexdigest(),
        samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    traces = [
        extract(args.real_directory / "real/trial-01.tar.gz", "handover"),
        extract(args.real_directory / "continuation/trial-01.tar.gz", "approach-drift"),
    ]
    profile = dict(
        schema=1,
        axes="world XY metres",
        interpretation="Observed estimate variability; no independent can ground truth. Includes jitter and persistent drift, which may include actual can movement.",
        unmeasured="Z, depth pixels, camera latency, postgrasp variability and motion-conditioned noise probabilities are not fitted.",
        traces=traces,
    )
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
