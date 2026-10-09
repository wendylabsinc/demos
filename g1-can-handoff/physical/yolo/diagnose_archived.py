"""Deterministic offline inference/depth diagnosis of archived NX2 failures.

No HTTP, DDS, camera leases, timestamp rewriting or hardware controls. Archived
exposures are never passed off as fresh observations. Reports do not qualify
hardware readiness; CPU FP32 numerical parity with CUDA FP16 is bounded evidence.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_input import target_coke_mask
from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates
from confidence import MIN_CONFIDENCE


def predict(network, image, threshold=MIN_CONFIDENCE):
    chw, scale, padx, pady = letterbox_bgr(image, 640)
    with torch.inference_mode():
        raw = network(torch.from_numpy(chw).float().div_(255).unsqueeze(0))
    if isinstance(raw, (tuple, list)):
        raw = raw[0]
    if tuple(raw.shape) != (1, 5, 8400):
        raise ValueError("Unexpected one-class prediction shape")
    scores, indices = raw[0, 4].topk(129, largest=True, sorted=True)
    decoded = decode_one_class_candidates(
        raw[0, :, indices[:128]].T.numpy(),
        confidence=min(0.05, threshold),
        nms_iou=0.7,
        max_detections=21,
        frame_width=image.shape[1],
        frame_height=image.shape[0],
        scale=scale,
        pad_x=padx,
        pad_y=pady,
    )
    boxes = [
        dict(
            xyxy=[b[k] for k in ("x1", "y1", "x2", "y2")],
            confidence=b["conf"],
            class_id=0,
            class_name="Coca-Cola",
        )
        for b in decoded
    ]
    return {
        "detections": boxes[:20],
        "truncated": len(boxes) > 20 or bool(scores[-1] > threshold),
        "max_raw_confidence": float(scores[0]),
        "top_raw_confidences": scores[:8].tolist(),
        "eligible_box_count": sum(b["confidence"] > threshold for b in boxes[:20]),
    }


def archive(path):
    with np.load(path, allow_pickle=False) as f:
        metadata = json.loads(str(f["metadata_json"]))
        frame = {
            "color_bgr": f["color_bgr"].copy(),
            "depth_m": f["depth_m"].copy(),
            "rays": f["rays"].copy(),
            "meta": metadata,
            "timestamp": metadata["depth_time"],
        }
    source = path.with_suffix(".source.zip")
    if source.exists():
        with zipfile.ZipFile(source) as z:
            cm = json.loads(z.read("metadata.json"))
            rgb = np.asarray(
                Image.open(io.BytesIO(z.read("color.jpg"))).convert("RGB")
            )[..., ::-1].copy()
            depth = (
                np.asarray(
                    Image.open(io.BytesIO(z.read("depth.png"))), dtype=np.float32
                )
                * cm["calibration"]["depth_scale_m"]
            )
        if (
            cm["source_frame_id"] != metadata["frame_id"]
            or cm["captured_at_unix_ns"] != metadata["captured_at_unix_ns"]
            or not np.array_equal(rgb, frame["color_bgr"])
            or not np.array_equal(depth, frame["depth_m"])
        ):
            raise ValueError("Archived NPZ/source ZIP identity or pixel mismatch")
    return frame, source.exists()


def scene_for_frame(events, frame_id):
    scene = None
    for row in events:
        if row["phase"] == "scene_refreshed":
            scene = row["calibration"]
        if (
            row["phase"] == "phase_camera_evidence"
            and row["metadata"]["frame_id"] == frame_id
        ):
            return scene, row.get("phase_name")
    return None, None


def depth_report(frame, config, result, model_sha):
    if config is None:
        return {"error": "No preceding scene calibration event"}
    t = np.asarray(config["camera_to_world"])
    world = (frame["rays"] * frame["depth_m"][..., None]) @ t[:3, :3].T + t[:3, 3]
    try:
        mask, _ = target_coke_mask(
            frame,
            config,
            world,
            result={
                **result,
                "model_sha256": model_sha,
                "backend": "cpu_offline_replay",
                "source": {
                    "image_sha256": hashlib.sha256(
                        frame["color_bgr"].tobytes()
                    ).hexdigest()
                },
            },
        )
    except Exception as exc:
        return {"error": str(exc)}
    points = world[mask]
    points = points[:: max(1, len(points) // 1500)]
    fit = least_squares(
        lambda c: np.linalg.norm(points[:, :2] - c, axis=1) - config["can_radius_m"],
        points[:, :2].mean(axis=0),
        loss="soft_l1",
        f_scale=0.002,
    )
    residual = float(np.sqrt(np.mean(fit.fun**2)))
    depth = frame["depth_m"]
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 2.5)
    distance2 = (world[:, :, 0] - fit.x[0]) ** 2 + (world[:, :, 1] - fit.x[1]) ** 2
    z = world[valid & ~mask & (distance2 < 0.22**2), 2]
    plane = z[np.abs(z - config["table_height_m"]) < 0.008]
    return {
        "mask_pixels": int(mask.sum()),
        "fit_points": len(points),
        "center_xy_m": fit.x.tolist(),
        "cylinder_residual_m": residual,
        "cylinder_gate_passed": residual <= 0.006,
        "top_z_m": float(np.quantile(points[:, 2], 0.95)),
        "bottom_z_m": float(np.quantile(points[:, 2], 0.05)),
        "table_plane_points": len(plane),
        "table_median_error_m": (
            float(abs(np.median(plane) - config["table_height_m"]))
            if len(plane)
            else None
        ),
        "left_hand_workspace_passed": bool(
            0.27 <= fit.x[0] <= 0.45 and 0.09 <= fit.x[1] <= 0.29
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).parent / "model/coke-can.pt"
    )
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument(
        "--confidence",
        type=float,
        default=MIN_CONFIDENCE,
        help="Offline comparison floor; use .30 for historical failure replay",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not MIN_CONFIDENCE <= args.confidence <= 0.99:
        parser.error("Offline comparison confidence must be within current floor..0.99")
    os.environ["HANDOFF_YOLO_MIN_CONFIDENCE"] = str(args.confidence)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    events = [json.loads(line) for line in args.events.read_text().splitlines()]
    images = [(p, *archive(p)) for p in args.frames]
    models = {}
    for weights in [args.model] + ([args.compare_model] if args.compare_model else []):
        model = YOLO(weights)
        if model.names != {0: "Coca-Cola"}:
            raise ValueError("Expected trained Coca-Cola checkpoint")
        sha = hashlib.sha256(weights.read_bytes()).hexdigest()
        network = model.model.fuse(verbose=False).eval().float().cpu()
        reports = []
        for path, frame, paired in images:
            result = predict(network, frame["color_bgr"], args.confidence)
            repeat = predict(network, frame["color_bgr"], args.confidence)
            if result != repeat:
                raise RuntimeError("Offline inference replay is not deterministic")
            config, phase = scene_for_frame(events, frame["meta"]["frame_id"])
            preview = args.output.parent / (path.stem + ".png")
            Image.fromarray(frame["color_bgr"][..., ::-1]).save(preview)
            reports.append(
                {
                    "archive": str(path),
                    "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "frame_id": frame["meta"]["frame_id"],
                    "phase": phase,
                    "source_zip_pixel_identity_verified": paired,
                    "exposure_timestamp_original": frame["timestamp"],
                    "timestamp_valid": frame["meta"]["timestamp_valid"],
                    "rgb_depth_skew_ms": abs(
                        frame["meta"]["color_time"] - frame["timestamp"]
                    )
                    * 1000,
                    "scene_source_frame_id": (
                        config.get("source_frame_id") if config else None
                    ),
                    "preview": str(preview),
                    **result,
                    "depth": depth_report(frame, config, result, sha),
                }
            )
        models[sha] = {"weights": str(weights), "frames": reports}
    report = {
        "schema": "wendy.handoff.archived-diagnosis.v1",
        "backend": "cpu_fp32_offline",
        "confidence_strictly_above": args.confidence,
        "deterministic_repeat_passed": True,
        "physical_readiness_verified": False,
        "timestamp_rewritten": False,
        "hardware_io": False,
        "models": models,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if any(
        row["eligible_box_count"] != 1 or row["truncated"]
        for row in models[next(iter(models))]["frames"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
