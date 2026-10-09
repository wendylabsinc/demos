"""Bounded, passive camera-to-detection/geometry latency qualification.

Run inside the physical container with its pinned packages and shared monotonic
clock. Uses public exact-frame bundles only; never acquires/renews a lease,
opens USB/DDS, instantiates an adapter, or issues robot commands.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from camera_input import Camera, geometry
from yolo_client import YoloClient, fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-age-ms", type=float, default=400)
    parser.add_argument("--max-p95-ms", type=float, default=180)
    parser.add_argument(
        "--config",
        type=Path,
        help="Include full measured geometry from this calibration JSON",
    )
    args = parser.parse_args()
    if not 1 <= args.samples <= 32 or not 0 <= args.warmup <= 3:
        parser.error("Bounded probe requires samples1..32 and warmup0..3")
    camera = Camera()
    detector = YoloClient()
    request = camera.request
    camera.request = lambda path, data=None: (
        b"{}" if path == "/lease/renew" else request(path, data)
    )
    config = json.loads(args.config.read_text()) if args.config else None
    rows = []
    try:
        for index in range(args.warmup + args.samples):
            started = time.perf_counter()
            frame = None
            error = None
            vision = None
            try:
                frame = camera.frame(materialize=False)
                if config is None:
                    result = detector.detect(frame)
                    vision = {
                        "inference_ms": result["inference_ms"],
                        "confidence": max(
                            (b["confidence"] for b in result["detections"]), default=0
                        ),
                    }
                else:
                    _, vision = geometry(frame, config, detector=detector)
                fresh(frame)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = {
                "frame_id": frame["meta"]["frame_id"] if frame else None,
                "total_ms": (time.perf_counter() - started) * 1000,
                "exposure_age_ms": (
                    (time.monotonic() - frame["timestamp"]) * 1000 if frame else None
                ),
                "timings_ms": frame.get("timings_ms") if frame else None,
                "vision": vision,
                "error": error,
            }
            if index >= args.warmup:
                rows.append(row)
            time.sleep(0.067)
    finally:
        camera.close()
    p95 = float(np.percentile([r["total_ms"] for r in rows], 95))
    passed = (
        all(
            r["error"] is None and 0 <= r["exposure_age_ms"] <= args.max_age_ms
            for r in rows
        )
        and p95 <= args.max_p95_ms
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "passive": True,
                "motion_commands_sent": 0,
                "lease_acquired": False,
                "includes_geometry": config is not None,
                "p95_total_ms": p95,
                "max_age_ms": args.max_age_ms,
                "max_p95_ms": args.max_p95_ms,
                "samples": rows,
            },
            allow_nan=False,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
