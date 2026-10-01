"""Append genuine no-hazard rendered RGB crops to an existing incident dataset.

This offline fixture samples camera poses and worker positions only to collect
training images. It never reads deployed test frames or changes existing crops.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from train_incident_classifier import camera_pose, spatial_cells, worker_pose
from yard_drone.environment import YardEnvironment
from yard_drone.model import build_model
from yard_drone.sensors import SensorRig, scene_state


def crop_box(image, rng, mode, worker_pixel=None):
    height, width = image.shape[:2]
    side = int(rng.integers(20, min(200, height, width) + 1))
    if mode == "worker" and worker_pixel is not None:
        center = np.asarray(worker_pixel) + rng.uniform(-side * .25, side * .25, 2)
    elif mode == "edge":
        # Sample visible geometric boundaries from genuine camera pixels.
        # This chooses diverse negative crops; it is not an incident detector.
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 120)
        ys, xs = np.where(edges > 0)
        if len(xs):
            index = int(rng.integers(len(xs)))
            center = np.array([xs[index], ys[index]], dtype=float)
        else:
            center = rng.uniform([0, 0], [width - 1, height - 1])
    else:
        center = rng.uniform([0, 0], [width - 1, height - 1])
    left = int(np.clip(round(center[0] - side / 2), 0, width - side))
    top = int(np.clip(round(center[1] - side / 2), 0, height - side))
    return [left, top, side, side]


def project_point(frame, point):
    local = np.asarray(frame["cameraRotation"]).T @ (np.asarray(point) - frame["cameraPosition"])
    if local[2] >= -.1:
        return None
    k = frame["intrinsics"]
    return [float(k["cx"] + k["fx"] * local[0] / -local[2]),
            float(k["cy"] - k["fy"] * local[1] / -local[2])]


def append_split(dataset, split, count, seed, model, data, environment, rig):
    manifest_path = dataset / f"{split}.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("negativeAugmentation"):
        raise ValueError(f"{split} already has negative augmentation; refusing duplicate examples")
    rng = np.random.default_rng(seed)
    cells = spatial_cells(environment, split)
    keys = sorted(cells)
    additions = []
    directory = dataset / split / "other"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        key = keys[int(rng.integers(len(keys)))]
        point = np.array(cells[key][int(rng.integers(len(cells[key])))]) + rng.uniform(-.08, .08, 2)
        environment.clear()
        camera_pose(data, point, rng)
        # Workers and their shadows are ordinary negative evidence. Their
        # position is an offline scene fixture and never a CNN input.
        worker_point = point + rng.uniform(-.65, .65, 2)
        worker_pose(model, data, worker_point)
        mujoco.mj_forward(model, data)
        frame = rig.capture(copy.copy(data), scene_state(model), "survey")
        mode = ("random", "edge", "worker")[index % 3]
        worker_pixel = project_point(frame, [*worker_point, .9])
        left, top, width, height = crop_box(frame["image"], rng, mode, worker_pixel)
        crop = frame["image"][top:top+height, left:left+width]
        filename = directory / f"background-v1-{index:04d}.png"
        if filename.is_file() and any(row["file"] == str(filename.relative_to(dataset)) for row in manifest["examples"]):
            raise ValueError(f"Refusing to replace existing dataset image {filename}")
        Image.fromarray(crop).save(filename)
        additions.append({"file": str(filename.relative_to(dataset)), "label": "other",
            "locationCell": list(key), "position": point.tolist(), "fieldAge": 0.,
            "source": "Actual no-hazard RGB camera crop: yard structures, fences, ground, worker and shadows",
            "cropSampling": mode, "cropBox": [left, top, width, height],
            "cameraPosition": frame["cameraPosition"], "cameraRotation": frame["cameraRotation"],
            "intrinsics": frame["intrinsics"], "negativeAugmentationSeed": seed})
        print(f"{split}: negative {index+1}/{count} {mode}, actual RGB {crop.shape[1]}x{crop.shape[0]}", flush=True)
    manifest["rgbSha256BeforeNegativeAugmentation"] = manifest["rgbSha256"]
    manifest["examples"].extend(additions)
    digest = hashlib.sha256()
    for row in manifest["examples"]:
        digest.update(row["label"].encode()); digest.update((dataset / row["file"]).read_bytes())
    manifest["rgbSha256"] = digest.hexdigest()
    manifest["negativeAugmentation"] = {"version": 1, "examples": count, "seed": seed,
        "cameras": "Actual survey RGB at random height 3.5..5.7m and yaw -pi..pi",
        "sampling": "Equal random, image-edge and visible worker-centered crops; 20..200 pixels",
        "runtimeTestFramesUsed": False, "originalExamplesUnchanged": True}
    temporary = manifest_path.with_suffix(".json.new")
    temporary.write_text(json.dumps(manifest, indent=2)+"\n")
    os.replace(temporary, manifest_path)
    return {tuple(row["locationCell"]) for row in manifest["examples"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train", type=int, default=90)
    parser.add_argument("--validation", type=int, default=45)
    parser.add_argument("--train-seed", type=int, default=81173)
    parser.add_argument("--validation-seed", type=int, default=92279)
    args = parser.parse_args()
    if args.train < 1 or args.validation < 1 or args.train_seed == args.validation_seed:
        parser.error("Use positive counts and distinct split seeds")
    # Validate both targets before generating either split.
    for split in ("train", "validation"):
        manifest = json.loads((args.dataset / f"{split}.json").read_text())
        if manifest.get("negativeAugmentation"):
            parser.error(f"{split} already has appended negatives")
    model = build_model(); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    environment = YardEnvironment(model, data)
    with SensorRig(model) as rig:
        train_cells = append_split(args.dataset, "train", args.train, args.train_seed, model, data, environment, rig)
        validation_cells = append_split(args.dataset, "validation", args.validation, args.validation_seed, model, data, environment, rig)
    if train_cells & validation_cells:
        raise RuntimeError("Negative generation leaked location cells between splits")
    print(f"Added {args.train} train and {args.validation} validation no-hazard RGB images, with disjoint location cells.", flush=True)


if __name__ == "__main__":
    main()
