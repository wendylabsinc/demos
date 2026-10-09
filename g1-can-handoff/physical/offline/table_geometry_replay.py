"""Replay exact native table-fault RGB-D with recorded CUDA semantic seed.

The most recent semantic reference and every recorded visual frame through the
original fault are replayed; earlier tracker history and motion are not replayed.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.ndimage import binary_erosion
from camera_input import Camera, geometry, world_points, fit_cylinder
from pose_alignment import corrected_config, PoseHistory
from table_plane import fit_table_plane
from target_tracker import TargetTracker
from yolo_client import SCHEMA, identity, validate_result

DIRECTORY = Path(__file__).parent / "validation/issue-table-geometry-20261008"


def inputs():
    capture = json.loads((DIRECTORY / "capture-manifest.json").read_text())
    for name, expected in capture["files"].items():
        if hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Changed captured table evidence: " + name)
    manifest = json.loads((DIRECTORY / "input-manifest.json").read_text())
    for name, row in manifest["files"].items():
        raw = (DIRECTORY / name).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise RuntimeError("Changed native table evidence: " + name)
    return manifest, json.loads((DIRECTORY / "native-poses.json").read_text())["poses"]


def frame(row):
    timing = (DIRECTORY / (row["stem"] + ".timing.npz")).read_bytes()
    compact = (DIRECTORY / (row["stem"] + ".source.zip")).read_bytes()
    with np.load(io.BytesIO(timing), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
    with np.load(
        Path(__file__).parent / "evidence/rgbd/rays.npz", allow_pickle=False
    ) as data:
        rays = data["rays"].copy()
    assert meta["clock"] == "host_monotonic" and meta["timestamp_valid"]
    for key in [
        "frame_id",
        "captured_at_unix_ns",
        "depth_time",
        "color_time",
        "stream_id",
        "camera_serial",
    ]:
        assert meta[key] == row["event"]["metadata"][key]
    assert hashlib.sha256(rays.tobytes()).hexdigest() == row["native"]["rays_sha256"]
    camera = Camera()
    camera.rays = rays
    camera.intrinsics = meta["intrinsics"]
    camera.serial = meta["camera_serial"]

    def request(path, data=None):
        if path == "/lease/renew":
            return b"{}"
        if path == "/health":
            return json.dumps(
                dict(frame_id=meta["frame_id"], serial=camera.serial)
            ).encode()
        if "policy-rgbd" in path:
            return timing
        if "recorder-rgbd" in path:
            return compact
        raise AssertionError("Unrecorded network endpoint: " + path)

    camera.request = request
    try:
        with patch("time.monotonic", return_value=meta["depth_time"] + 0.25):
            result = camera.frame(materialize=False)
    finally:
        camera.close()
    assert (
        hashlib.sha256(result["color_bgr"].tobytes()).hexdigest()
        == row["native"]["color_sha256"]
    )
    assert (
        hashlib.sha256(result["depth_m"].tobytes()).hexdigest()
        == row["native"]["depth_sha256"]
    )
    return result


def aligned_config(config, row, poses):
    pose = poses[str(row["native"]["metadata"]["frame_id"])]["pose"]
    h = PoseHistory()
    h.add(pose["timestamp"], pose["rpy"], pose["waist_q"])
    return corrected_config(config, h.nearest(row["native"]["metadata"]["depth_time"]))


def table_quality(calibration_frame):
    image = calibration_frame["color_bgr"][..., ::-1].astype(float)
    r, g, b = np.moveaxis(image, -1, 0)
    depth = calibration_frame["depth_m"]
    points = calibration_frame["rays"] * depth[..., None]
    yellow = binary_erosion(
        (r > 100)
        & (g > 100)
        & (b < 0.65 * np.minimum(r, g))
        & np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 2),
        iterations=2,
    )
    return fit_table_plane(
        points[yellow][::3],
        points[::4, ::4].reshape(-1, 3),
        points[2::4, 2::4].reshape(-1, 3),
    )[3]


def replay(case, poses, correct=True):
    seedrow = next(
        r
        for r in case["frames"]
        if r["native"]["metadata"]["frame_id"] == case["semantic_frame_id"]
    )
    faultrow = next(
        r
        for r in case["frames"]
        if r["native"]["metadata"]["frame_id"] == case["fault_frame_id"]
    )
    seed = frame(seedrow)
    fault = frame(faultrow)
    native = seedrow["perception_result"]
    frozen = dict(
        schema=SCHEMA,
        source=native["source"],
        model_sha256=native["model_sha256"],
        backend="pytorch_cuda",
        truncated=False,
        detections=native["detections"],
    )
    assert identity(seed) == frozen["source"]
    detector = SimpleNamespace(detect=lambda exact: validate_result(exact, frozen))
    tracker = TargetTracker(detector=detector)
    cfg = case["calibration"]
    results = []
    sequence = [
        r
        for r in case["frames"]
        if case["semantic_frame_id"]
        <= r["native"]["metadata"]["frame_id"]
        <= case["fault_frame_id"]
    ]
    for row in sequence:
        exact = seed if row is seedrow else fault if row is faultrow else frame(row)
        config = aligned_config(cfg, row, poses) if correct else cfg
        with patch("time.monotonic", return_value=exact["timestamp"] + 0.25):
            try:
                obs, vision = geometry(exact, config, tracker=tracker)
                mask = tracker.pending["mask"]
                tracker.accept(obs, vision, exact)
                points = world_points(exact, config)
                valid = (
                    np.isfinite(exact["depth_m"])
                    & (exact["depth_m"] > 0.1)
                    & (exact["depth_m"] < 2.5)
                )
                distance2 = (points[:, :, 0] - obs.center[0]) ** 2 + (
                    points[:, :, 1] - obs.center[1]
                ) ** 2
                z = points[valid & ~mask & (distance2 < 0.22**2), 2]
                selected = z[np.abs(z - cfg["table_height_m"]) < 0.008]
                results.append(
                    dict(
                        frame_id=exact["meta"]["frame_id"],
                        accepted=True,
                        vision=vision,
                        table_support=len(selected),
                        table_median_error_mm=float(
                            1000 * (np.median(selected) - cfg["table_height_m"])
                        ),
                        pose_skew_ms=1000
                        * poses[str(exact["meta"]["frame_id"])]["skew_s"],
                    )
                )
            except RuntimeError as error:
                failure = dict(
                    frame_id=exact["meta"]["frame_id"], accepted=False, error=str(error)
                )
                if (
                    str(error) == "Table plane differs by over 5 mm"
                    and tracker.pending is not None
                ):
                    points = world_points(exact, config)
                    mask = tracker.pending["mask"]
                    target = points[mask]
                    target = target[:: max(1, len(target) // 1500)]
                    fit = fit_cylinder(target[:, :2], cfg["can_radius_m"])
                    valid = (
                        np.isfinite(exact["depth_m"])
                        & (exact["depth_m"] > 0.1)
                        & (exact["depth_m"] < 2.5)
                    )
                    distance2 = (points[:, :, 0] - fit.x[0]) ** 2 + (
                        points[:, :, 1] - fit.x[1]
                    ) ** 2
                    z = points[valid & ~mask & (distance2 < 0.22**2), 2]
                    selected = z[np.abs(z - cfg["table_height_m"]) < 0.008]
                    failure.update(
                        table_median_error_mm=float(
                            1000 * (np.median(selected) - cfg["table_height_m"])
                        ),
                        table_support=len(selected),
                    )
                results.append(failure)
                break
    return dict(
        attempt=case["attempt"],
        corrected=correct,
        results=results,
        accepted=len(results) == len(sequence) and all(r["accepted"] for r in results),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncorrected", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    m, poses = inputs()
    runs = [
        replay(c, poses, not args.uncorrected)
        for _ in range(args.repeat)
        for c in m["cases"]
    ]
    quality = [
        dict(attempt=c["attempt"], quality=table_quality(frame(c["frames"][0])))
        for c in m["cases"]
    ]
    report = dict(
        hardware_used=False,
        physical_success_verified=False,
        original_full_rgbd=True,
        all_tracker_frames_after_semantic_seed=True,
        full_trial_tracker_sequence=False,
        recorded_semantic_seed=True,
        limits=dict(
            table_height_mm=5,
            cylinder_fit_mm=6,
            minimum_target_support=40,
            rgbd_skew_ms=12,
            pose_skew_ms=20,
            attitude_rad=0.04,
            waist_rad=0.002,
        ),
        runs=runs,
        calibration_table_validation=quality,
        accepted=all(r["accepted"] for r in runs),
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
