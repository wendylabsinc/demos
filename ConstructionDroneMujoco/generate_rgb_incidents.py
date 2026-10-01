#!/usr/bin/env python3
"""Generate fresh, labeled MuJoCo RGB crops without an embedding/model dependency.

Run on the deployment renderer, e.g. MUJOCO_GL=osmesa on Linux. Labels and
projected fixture locations select crops offline only. The incident CNN receives
saved RGB pixels, never positions, geometry IDs or environmental state. No live
test frames are loaded. Separate seeds and checkerboard release cells separate
training from validation; both still share the same yard and renderer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

from yard_drone.environment import GRID_X, GRID_Y, YardEnvironment
from yard_drone.missions import compare_images, register
from yard_drone.model import build_model
from yard_drone.sensors import SensorRig, scene_state

CLASSES = ("fire", "oil", "other")
CONTEXTS = (8, 18, 36)


def spatial_cells(environment, split):
    cells = {}
    for row, y in enumerate(GRID_Y):
        for col, x in enumerate(GRID_X):
            if environment.blocked[row, col] or abs(x) > 7.5 or abs(y) > 6.5:
                continue
            key = (math.floor(x), math.floor(y))
            if (key[0] + key[1]) % 2 == (0 if split == "train" else 1):
                cells.setdefault(key, []).append((float(x), float(y)))
    if not cells:
        raise RuntimeError(f"No available {split} release cells")
    return cells


def quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def rotation_matrix(q):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q)
    return matrix.reshape(3, 3)


def camera_pose(data, point, rng, mode):
    yaw_mode = rng.choice(("fixed", "cardinal", "random"), p=(.4, .25, .35))
    yaw = 0. if yaw_mode == "fixed" else float(rng.choice((0., math.pi / 2, math.pi, -math.pi / 2)))
    if yaw_mode == "random":
        yaw = float(rng.uniform(-math.pi, math.pi))
    roll, pitch = rng.uniform(-.06, .06, 2)
    q = quaternion(roll, pitch, yaw)
    height = float(rng.uniform(3.5, 5.7))
    limit = .9 if mode == "near" else 2.6
    offset = rng.uniform(-limit, limit, 2)
    if mode == "edge":
        # Aim the release at a real image boundary, including small clipping.
        # The camera remains the real model camera, not a synthetic projection.
        height = float(rng.uniform(3.5, 3.9))
        u = float(rng.uniform(115, 205))
        v = float(rng.uniform(-3, 12) if rng.random() < .5 else rng.uniform(227, 242))
        focal = 240 / (2 * math.tan(math.radians(65) / 2))
        direction = rotation_matrix(q) @ np.array([(u - 159.5) / focal, -(v - 119.5) / focal, -1.])
        camera_offset = rotation_matrix(q) @ np.array([0., 0., -.075])
        distance = (height + camera_offset[2]) / -direction[2]
        offset = -distance * direction[:2] - camera_offset[:2]
        offset = np.clip(offset, -2.6, 2.6)
    data.qpos[:3] = [point[0] + offset[0], point[1] + offset[1], height]
    data.qpos[3:7] = q
    data.qvel[:] = 0
    return {"mode": mode, "offset": offset.tolist(), "yawMode": str(yaw_mode),
            "euler": [float(roll), float(pitch), yaw]}


def jitter_pose(data, rng, enabled):
    shift = rng.uniform([-.025, -.025, -.01], [.025, .025, .01]) if enabled else np.zeros(3)
    angles = rng.uniform(-.002, .002, 3) if enabled else np.zeros(3)
    data.qpos[:3] += shift
    delta = quaternion(*angles)
    result = np.empty(4)
    mujoco.mju_mulQuat(result, data.qpos[3:7].copy(), delta)
    data.qpos[3:7] = result
    return {"translation": shift.tolist(), "rotationRadians": angles.tolist()}


def worker_pose(model, data, point):
    for name, value in zip(("worker_x", "worker_y"), (point[0] - 2., point[1])):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value


def capture(rig, model, data):
    return rig.capture(copy.copy(data), scene_state(model), "survey")


def project_point(frame, point):
    local = np.asarray(frame["cameraRotation"]).T @ (np.asarray(point) - frame["cameraPosition"])
    if local[2] >= -.1:
        return None
    k = frame["intrinsics"]
    return np.array([k["cx"] + k["fx"] * local[0] / -local[2],
                     k["cy"] - k["fy"] * local[1] / -local[2]])


def box_distance(point, box):
    x, y, width, height = box
    dx = max(x - point[0], 0, point[0] - (x + width - 1))
    dy = max(y - point[1], 0, point[1] - (y + height - 1))
    return math.hypot(dx, dy)


def true_change_support(reference, current, projected):
    """Find a genuine scene-change component at the projected fixture location.

    The reference has exactly the current camera pose and no injected incident
    (or moved worker). This additional actual render excludes unrelated static
    parallax edges from positive labels. It is not a runtime classifier input.
    """
    if projected is None:
        return None
    difference = np.max(np.abs(current["image"].astype(np.int16) - reference["image"].astype(np.int16)), axis=2)
    mask = (difference > 20).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        distance = box_distance(projected, [x, y, width, height])
        if area >= 30 and distance <= 14:
            candidates.append((distance, -area, index))
    if not candidates:
        return None
    index = min(candidates)[2]
    return labels == index


def compare_components(clean, current):
    """Attach exact connected pixels to the shared runtime comparison results.

    compare_images intentionally exposes boxes only. Reconstruct its mask with
    its registration/threshold/morphology operations, then verify both aligned
    render and component statistics before using a mask as an offline label.
    If a RANSAC pass disagrees, reject the ambiguous positive rather than label
    an unrelated structure enclosed by a component's bounding rectangle.
    """
    aligned, regions, registration = compare_images(clean["image"], current["image"], clean, current)
    checked, valid, _ = register(clean["image"], current["image"], clean, current)
    if not valid.any() or not np.array_equal(aligned, checked):
        return [], None, {**registration, "offlineComponentMasksVerified": False}
    before = cv2.GaussianBlur(aligned, (5, 5), 0).astype(np.float32)
    after = cv2.GaussianBlur(current["image"], (5, 5), 0).astype(np.float32)
    offset = np.median((after - before)[valid], axis=0)
    difference = np.max(np.abs(after - before - offset), axis=2)
    changed = ((difference > 35) & valid).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    changed &= valid.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
    lookup = {tuple(map(int, stats[index])): index for index in range(1, count) if stats[index, 4] >= 85}
    annotated = []
    for region in regions:
        key = (*region["box"], region["changedPixels"])
        if key not in lookup:
            return [], None, {**registration, "offlineComponentMasksVerified": False}
        annotated.append({**region, "connectedComponentLabel": lookup[key]})
    return annotated, labels, {**registration, "offlineComponentMasksVerified": True}


def matched_region(regions, labels, support, projected):
    if support is None or projected is None or labels is None:
        return None
    candidates = []
    support_area = int(support.sum())
    for region in regions:
        overlap = int(np.count_nonzero(support & (labels == region["connectedComponentLabel"])))
        distance = box_distance(projected, region["box"])
        # The selected runtime component must contain genuine changed pixels
        # near the known release; choosing the largest image change can instead
        # assign a fire/oil label to a static tank or fence parallax edge.
        if overlap >= max(25, math.ceil(support_area * .08)) and distance <= 14:
            candidates.append((overlap, -distance, region))
    if not candidates:
        return None
    overlap, _, region = max(candidates, key=lambda item: item[:2])
    return {**region, "trueChangeOverlapPixels": overlap,
            "trueChangeSupportPixels": support_area}


def crop_region(image, box, context):
    x, y, width, height = map(int, box)
    left, top = max(0, x - context), max(0, y - context)
    right, bottom = min(image.shape[1], x + width + context), min(image.shape[0], y + height + context)
    return image[top:bottom, left:right], [left, top, right - left, bottom - top]


def background_box(image, rng):
    height, width = image.shape[:2]
    side = int(rng.integers(20, min(200, height, width) + 1))
    mode = "random" if rng.random() < .5 else "image edge"
    center = rng.uniform([0, 0], [width - 1, height - 1])
    if mode == "image edge":
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 120)
        ys, xs = np.where(edges > 0)
        if len(xs):
            index = int(rng.integers(len(xs)))
            center = np.array([xs[index], ys[index]])
    left = int(np.clip(round(center[0] - side / 2), 0, width - side))
    top = int(np.clip(round(center[1] - side / 2), 0, height - side))
    return [left, top, side, side], mode


def generate_split(split, count, seed, model, data, environment, rig, directory, args):
    rng = np.random.default_rng(seed)
    cells = spatial_cells(environment, split)
    keys = sorted(cells)
    examples, rejected = [], Counter()
    started = time.monotonic()
    for example in range(count):
        label = CLASSES[example % 3]
        for attempt in range(1, args.max_attempts + 1):
            key = keys[int(rng.integers(len(keys)))]
            point = np.array(cells[key][int(rng.integers(len(cells[key])))]) + rng.uniform(-.08, .08, 2)
            mode = str(rng.choice(("near", "wide", "edge"), p=(.3, .4, .3)))
            margin = int(rng.choice(CONTEXTS, p=(.25, .5, .25)))
            environment.clear()
            worker_pose(model, data, (-20., -20.))
            pose_meta = camera_pose(data, point, rng, mode)
            data.time = 0.
            clean = capture(rig, model, data)
            jitter = jitter_pose(data, rng, rng.random() < args.jitter_probability)
            reference = capture(rig, model, data)
            region, registration, age = None, None, 0.
            if label in ("fire", "oil"):
                environment.inject(label, point)
                age = round(float(rng.uniform(0, args.max_age)) / .04) * .04
                for _ in range(round(age / .04)):
                    environment.step(.04)
                data.time = age
                current = capture(rig, model, data)
                projected = project_point(current, [*point, 0.])
                support = true_change_support(reference, current, projected)
                regions, labels, registration = compare_components(clean, current)
                region = matched_region(regions, labels, support, projected)
                if region is None:
                    rejected[f"{label}: no unambiguous release-matching component"] += 1
                    continue
                source = "Actual evolving incident diff component matched to offline release and same-pose pixel support"
            else:
                negative_mode = str(rng.choice(("worker", "parallax", "unchanged")))
                current = reference
                if negative_mode == "worker":
                    worker_point = point + rng.uniform(-.35, .35, 2)
                    worker_pose(model, data, worker_point)
                    current = capture(rig, model, data)
                    projected = project_point(current, [*worker_point, .9])
                    support = true_change_support(reference, current, projected)
                    regions, labels, registration = compare_components(clean, current)
                    region = matched_region(regions, labels, support, projected)
                    if region is None:
                        rejected["other: no unambiguous worker component"] += 1
                        continue
                    source = "Actual worker and shadow diff component; no hazard in scene"
                elif negative_mode == "parallax":
                    regions, _, registration = compare_components(clean, current)
                    if regions:
                        region = regions[int(rng.integers(len(regions)))]
                        source = "Actual no-hazard static-edge diff component from calibrated pose jitter"
                    else:
                        box, sampling = background_box(current["image"], rng)
                        region = {"box": box, "changedPixels": 0}
                        source = f"Actual no-hazard {sampling} crop; pose registration left no changed component"
                else:
                    box, sampling = background_box(current["image"], rng)
                    region = {"box": box, "changedPixels": 0}
                    source = f"Actual unchanged no-hazard {sampling} crop"
            crop, crop_box = crop_region(current["image"], region["box"], margin)
            if min(crop.shape[:2], default=0) < 16 or np.ptp(crop) <= 10:
                rejected[f"{label}: crop too small or visually constant"] += 1
                continue
            x, y, width, height = region["box"]
            boundary = x <= 1 or y <= 1 or x + width >= rig.width - 1 or y + height >= rig.height - 1
            path = directory / split / label / f"rgb-{example:05d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(crop).save(path)
            examples.append({"file": str(path.relative_to(directory)), "label": label,
                "locationCell": list(key), "position": point.tolist(), "fieldAge": age,
                "source": source, "cameraSampling": pose_meta, "poseJitter": jitter,
                "cropMargin": margin, "component": region, "cropBox": crop_box,
                "componentTouchesImageBoundary": bool(boundary), "registration": registration,
                "cameraPosition": current["cameraPosition"], "cameraRotation": current["cameraRotation"],
                "intrinsics": current["intrinsics"], "attempts": attempt})
            print(f"{split} {example + 1}/{count}: {label}, {mode}, {crop.shape[1]}x{crop.shape[0]}, context {margin}, boundary {boundary}", flush=True)
            break
        else:
            raise RuntimeError(f"No valid {split} {label} RGB crop for example {example} after {args.max_attempts} attempts; rejected={dict(rejected)}")
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example["label"].encode())
        digest.update((directory / example["file"]).read_bytes())
    manifest = {"version": 1, "generator": "generate_rgb_incidents.py", "seed": seed,
        "examples": examples, "rgbSha256": digest.hexdigest(), "rejectedAttempts": dict(rejected),
        "rendererBackend": os.environ.get("MUJOCO_GL", "MuJoCo platform default"),
        "platform": platform.system(), "mujocoVersion": mujoco.__version__,
        "camera": "Actual MuJoCo survey perspective RGB with metric calibration",
        "groundTruth": "Offline scene-generation labels; positive diff components must match projected release and same-pose rendered support",
        "cameraOffsets": "Near +/-0.9m, wide +/-2.6m and calibrated FOV-edge poses; fixed/cardinal/random yaw",
        "contextPixels": list(CONTEXTS), "contextProbabilities": [.25, .5, .25],
        "maxFieldAgeSeconds": args.max_age,
        "fieldAgeRangesSeconds": {"fire": [0., args.max_age], "oil": [0., args.max_age]},
        "poseJitter": {"probability": args.jitter_probability,
            "translationMaxMeters": [.025, .025, .01], "rotationMaxRadians": .002},
        "runtimeTestFramesUsed": False,
        "scope": "Fresh simulator crops only; shared yard/renderer, disjoint checkerboard release cells",
        "elapsedSeconds": round(time.monotonic() - started, 2)}
    (directory / f"{split}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {tuple(example["locationCell"]) for example in examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train", type=int, default=360)
    parser.add_argument("--validation", type=int, default=180)
    parser.add_argument("--train-seed", type=int, default=601117)
    parser.add_argument("--validation-seed", type=int, default=703219)
    parser.add_argument("--max-age", type=float, default=90.)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--jitter-probability", type=float, default=.65)
    args = parser.parse_args()
    if min(args.train, args.validation) < 3 or args.train % 3 or args.validation % 3:
        parser.error("Use positive train/validation counts divisible by three")
    if args.train_seed == args.validation_seed:
        parser.error("Use different training and validation seeds")
    if not math.isfinite(args.max_age) or args.max_age <= 0 or args.max_attempts < 1:
        parser.error("Maximum field age must be finite/positive and attempts at least one")
    if not 0 <= args.jitter_probability <= 1:
        parser.error("Jitter probability must be in [0,1]")
    if args.dataset.exists() and any(args.dataset.iterdir()):
        parser.error("Use a fresh empty dataset directory; existing crops/manifests are never overwritten")
    args.dataset.mkdir(parents=True, exist_ok=True)
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    environment = YardEnvironment(model, data)
    with SensorRig(model) as rig:
        train_cells = generate_split("train", args.train, args.train_seed, model, data, environment, rig, args.dataset, args)
        validation_cells = generate_split("validation", args.validation, args.validation_seed, model, data, environment, rig, args.dataset, args)
    if train_cells & validation_cells:
        raise RuntimeError("Training and validation release cells overlap")
    print(f"Saved {args.train} training and {args.validation} validation RGB crops; release cells disjoint.", flush=True)


if __name__ == "__main__":
    main()
