#!/usr/bin/env python3
"""Train a yard-specific incident probe from actual MuJoCo camera crops.

Scene labels supervise dataset generation only. The exported classifier accepts
frozen MobileCLIP image embeddings at runtime, with no coordinates, flags,
geometry names, hazard parameters or simulator state among its inputs.

The validation split uses different ground cells and a separate random seed.
Reported metrics describe this simulator's visual domain, not real-world fire
or spill recognition. Run on the same renderer backend used for deployment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

from yard_drone.environment import GRID_X, GRID_Y, YardEnvironment
from yard_drone.missions import compare_images
from yard_drone.model import build_model
from yard_drone.perception import VisionService
from yard_drone.sensors import SensorRig, scene_state

CLASSES = ("fire", "oil", "other")


def image_embedding(vision, image):
    """Use the shared runtime encoder, or its current identical preprocessing.

    The fallback lets dataset generation run while the runtime probe adapter is
    being added. It matches VisionService.classify's frozen encoder exactly;
    only image pixels enter the network. Prefer VisionService.embed when present.
    """
    if hasattr(vision, "embed"):
        return np.asarray(vision.embed(image), dtype=np.float32).reshape(-1)
    frame = Image.fromarray(image).convert("RGB")
    config = vision._clip_config
    size = config["size"]["shortest_edge"]
    ratio = size / min(frame.size)
    frame = frame.resize((round(frame.width * ratio), round(frame.height * ratio)), Image.Resampling.BILINEAR)
    crop = config["crop_size"]
    left, top = (frame.width - crop["width"]) // 2, (frame.height - crop["height"]) // 2
    frame = frame.crop((left, top, left + crop["width"], top + crop["height"]))
    pixels = np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)[None] / 255.
    with vision._lock:
        names = [output.name for output in vision._vision.get_outputs()]
        features = vision._vision.run(["image_embeds"] if "image_embeds" in names else None,
                                      {"pixel_values": pixels})[0]
    features = np.asarray(features, dtype=np.float32).reshape(-1)
    return features / max(float(np.linalg.norm(features)), 1e-10)


def softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def fit_probe(features, labels, *, regularization=.04, epochs=1200, rate=.025):
    """Fit a regularized multinomial logistic probe with full-batch Adam.

    Centering and one global scale are learned from the training split only and
    baked into the final coefficients. A global scale preserves the pretrained
    encoder's feature geometry instead of amplifying low-variance noise axes.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Provide finite feature rows and matching class labels")
    if set(y.tolist()) != set(range(len(CLASSES))):
        raise ValueError("Training examples must include fire, oil and other")
    mean = x.mean(axis=0)
    scale = max(float(np.sqrt(np.mean(x.var(axis=0)))), .015)
    x = (x - mean) / scale
    targets = np.eye(len(CLASSES))[y]
    weights = np.zeros((x.shape[1], len(CLASSES)), dtype=np.float64)
    bias = np.zeros(len(CLASSES), dtype=np.float64)
    mw = np.zeros_like(weights); vw = np.zeros_like(weights)
    mb = np.zeros_like(bias); vb = np.zeros_like(bias)
    for epoch in range(1, epochs + 1):
        probabilities = softmax(x @ weights + bias)
        residual = (probabilities - targets) / len(x)
        dw = x.T @ residual + regularization * weights
        db = residual.sum(axis=0)
        mw = .9 * mw + .1 * dw; vw = .999 * vw + .001 * dw * dw
        mb = .9 * mb + .1 * db; vb = .999 * vb + .001 * db * db
        weights -= rate * (mw / (1 - .9 ** epoch)) / (np.sqrt(vw / (1 - .999 ** epoch)) + 1e-8)
        bias -= rate * (mb / (1 - .9 ** epoch)) / (np.sqrt(vb / (1 - .999 ** epoch)) + 1e-8)
    final_weights = weights / scale
    final_bias = bias - mean @ final_weights
    return final_weights.astype(np.float32), final_bias.astype(np.float32)


def evaluate(features, labels, weights, bias):
    logits = np.asarray(features) @ weights + bias
    predictions = np.argmax(logits, axis=1)
    labels = np.asarray(labels, dtype=int)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, predicted in zip(labels, predictions):
        matrix[truth, predicted] += 1
    recalls = {label: float(matrix[i, i] / max(1, matrix[i].sum()))
               for i, label in enumerate(CLASSES)}
    probabilities = softmax(logits)
    sorted_scores = np.sort(probabilities, axis=1)
    accepted = (sorted_scores[:, -1] >= .70) & (sorted_scores[:, -1] - sorted_scores[:, -2] >= .20)
    runtime_matrix = np.zeros((len(CLASSES), len(CLASSES) + 1), dtype=int)
    for truth, predicted, accept in zip(labels, predictions, accepted):
        runtime_matrix[truth, predicted if accept else len(CLASSES)] += 1
    runtime = {"confidenceThreshold": .70, "marginThreshold": .20,
        "accepted": int(accepted.sum()), "abstained": int((~accepted).sum()),
        "confusionColumns": [*CLASSES, "abstained"], "confusionMatrix": runtime_matrix.tolist(),
        "acceptedAccuracy": float(np.mean(predictions[accepted] == labels[accepted])) if accepted.any() else None,
        "acceptedHazardRecall": {label: float(np.count_nonzero(accepted & (predictions == i) & (labels == i)) /
            max(1, np.count_nonzero(labels == i))) for i, label in enumerate(CLASSES[:2])}}
    return {"examples": len(labels), "accuracy": float(np.mean(predictions == labels)),
            "classes": list(CLASSES), "confusionMatrix": matrix.tolist(),
            "confusionRows": "true class", "confusionColumns": "predicted class",
            "recall": recalls, "runtimeEvaluation": runtime}


def save_probe(path, weights, bias, metadata, validation_features=None, validation_labels=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"weights": np.asarray(weights, np.float32), "bias": np.asarray(bias, np.float32),
              "classes": np.array(CLASSES), "metadata": np.array(json.dumps(metadata, sort_keys=True))}
    if validation_features is not None:
        arrays.update(validation_features=np.asarray(validation_features, np.float32),
                      validation_labels=np.asarray(validation_labels, np.int64))
    np.savez_compressed(path, **arrays)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_cells(environment, split):
    """Disjoint checkerboard metre cells, excluding measured obstacles."""
    cells = {}
    for row, y in enumerate(GRID_Y):
        for col, x in enumerate(GRID_X):
            if environment.blocked[row, col] or abs(x) > 7.5 or abs(y) > 6.5:
                continue
            cell = (int(np.floor(x)), int(np.floor(y)))
            parity = (cell[0] + cell[1]) % 2
            if parity == (0 if split == "train" else 1):
                cells.setdefault(cell, []).append((float(x), float(y)))
    return cells


def camera_pose(data, point, rng):
    # Known fixture camera calibration poses; application control never assigns
    # qpos. Both compared frames share exactly this actual camera transform.
    data.qpos[:3] = [point[0] + rng.uniform(-.9, .9), point[1] + rng.uniform(-.9, .9), rng.uniform(3.5, 5.7)]
    roll, pitch, yaw = rng.uniform(-.10, .10), rng.uniform(-.10, .10), rng.uniform(-np.pi, np.pi)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    data.qpos[3:7] = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
    data.qvel[:] = 0


def worker_pose(model, data, point):
    for name, value in zip(("worker_x", "worker_y"), (point[0] - 2., point[1])):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value


def region_crop(clean, current):
    _, regions, _ = compare_images(clean["image"], current["image"], clean, current)
    if not regions:
        return None
    x, y, width, height = max(regions, key=lambda region: region["changedPixels"])["box"]
    margin = 18  # Same crop context as MissionTracker.prepare.
    return current["image"][max(0, y - margin):min(current["image"].shape[0], y + height + margin),
                            max(0, x - margin):min(current["image"].shape[1], x + width + margin)]


def ground_crop(frame, point, rng):
    rotation = np.asarray(frame["cameraRotation"])
    camera_point = rotation.T @ (np.array([*point, 0.]) - frame["cameraPosition"])
    depth = -camera_point[2]
    if depth <= .1:
        return None
    k = frame["intrinsics"]
    u = int(k["cx"] + k["fx"] * camera_point[0] / depth)
    v = int(k["cy"] - k["fy"] * camera_point[1] / depth)
    side = int(rng.integers(64, 130))
    image = frame["image"]
    left, top = max(0, u - side // 2), max(0, v - side // 2)
    crop = image[top:min(image.shape[0], top + side), left:min(image.shape[1], left + side)]
    return crop if min(crop.shape[:2], default=0) >= 32 else None


def generate_split(split, count, seed, model, data, environment, rig, vision, output_dir,
                   *, fire_max_age=10., oil_max_age=25.):
    rng = np.random.default_rng(seed)
    cells = spatial_cells(environment, split)
    keys = sorted(cells)
    features, labels, observations = [], [], []
    digest = hashlib.sha256()
    attempts = 0
    for example in range(count):
        label_index = example % len(CLASSES)
        label = CLASSES[label_index]
        crop = None
        for _ in range(20):
            attempts += 1
            key = keys[int(rng.integers(len(keys)))]
            point = np.array(cells[key][int(rng.integers(len(cells[key])))]) + rng.uniform(-.08, .08, 2)
            environment.clear()
            worker_pose(model, data, (-20, -20))
            camera_pose(data, point, rng)
            mujoco.mj_forward(model, data)
            clean = rig.capture(copy.copy(data), scene_state(model))
            if label == "other" and example % 2 == 0:
                # The moved worker and its shadow are negative examples for
                # incident classification, while remaining true RGB changes.
                worker_pose(model, data, point)
                mujoco.mj_forward(model, data)
                current = rig.capture(copy.copy(data), scene_state(model))
                crop = region_crop(clean, current)
                source = "physical worker and shadow change"
                age = 0.
            elif label == "other":
                current = clean
                crop = ground_crop(clean, point, rng)
                source = "unchanged yard camera crop"
                age = 0.
            else:
                try:
                    environment.inject(label, point)
                except ValueError:
                    continue
                age = float(rng.uniform(0., fire_max_age)) if label == "fire" else float(rng.uniform(0., oil_max_age))
                for _ in range(round(age / .04)):
                    environment.step(.04)
                mujoco.mj_forward(model, data)
                current = rig.capture(copy.copy(data), scene_state(model))
                crop = region_crop(clean, current)
                source = "rendered evolving incident change component"
            if crop is not None and min(crop.shape[:2]) >= 16 and np.ptp(crop) > 10:
                break
        else:
            raise RuntimeError(f"Could not obtain a visible actual {label} crop after 20 attempts")
        encoded = image_embedding(vision, crop)
        if encoded.shape != (512,) or not np.isfinite(encoded).all():
            raise RuntimeError(f"Expected a finite 512-value MobileCLIP image embedding, got {encoded.shape}")
        features.append(encoded); labels.append(label_index)
        crop_dir = output_dir / split / label
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"{example:04d}.png"
        Image.fromarray(crop).save(crop_path)
        digest.update(label.encode()); digest.update(crop_path.read_bytes())
        observation = {"file": str(crop_path.relative_to(output_dir)), "label": label,
                       "locationCell": list(key), "position": point.tolist(), "source": source,
                       "fieldAge": age, "cameraPosition": current["cameraPosition"],
                       "cameraRotation": current["cameraRotation"], "intrinsics": current["intrinsics"]}
        observations.append(observation)
        print(f"{split}: {example + 1}/{count} {label} actual RGB -> frozen embedding", flush=True)
    manifest = {"seed": seed, "examples": observations, "attempts": attempts, "rgbSha256": digest.hexdigest(),
                "rendererBackend": os.environ.get("MUJOCO_GL", "platform default"),
                "fieldAgeRangesSeconds": {"fire": [0., fire_max_age], "oil": [0., oil_max_age]}}
    (output_dir / f"{split}.json").write_text(json.dumps(manifest, indent=2))
    return np.stack(features), np.array(labels), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="/opt/models/vision")
    parser.add_argument("--output", type=Path, default=Path("yard_drone/incident_classifier.npz"))
    parser.add_argument("--dataset", type=Path, default=Path("/tmp/yard-incident-training"))
    parser.add_argument("--train", type=int, default=90)
    parser.add_argument("--validation", type=int, default=45)
    parser.add_argument("--train-seed", type=int, default=1729)
    parser.add_argument("--validation-seed", type=int, default=2718)
    parser.add_argument("--fire-max-age", type=float, default=10.,
                        help="Maximum simulated fire age in seconds for actual RGB training crops")
    parser.add_argument("--oil-max-age", type=float, default=25.,
                        help="Maximum simulated oil age in seconds for actual RGB training crops")
    args = parser.parse_args()
    if args.train < 12 or args.validation < 12 or args.train % 3 or args.validation % 3:
        parser.error("Use at least twelve examples per split, with counts divisible by three")
    if args.train_seed == args.validation_seed:
        parser.error("Use independent train and validation seeds")
    if not all(np.isfinite(value) and value > 0 for value in (args.fire_max_age, args.oil_max_age)):
        parser.error("Maximum field ages must be finite positive seconds")
    started = time.monotonic()
    args.dataset.mkdir(parents=True, exist_ok=True)
    vision = VisionService(args.models)
    if not vision.status()["ready"]:
        raise RuntimeError(vision.status()["error"])
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    environment = YardEnvironment(model, data)
    with SensorRig(model) as rig:
        train_x, train_y, train_manifest = generate_split("train", args.train, args.train_seed,
            model, data, environment, rig, vision, args.dataset,
            fire_max_age=args.fire_max_age, oil_max_age=args.oil_max_age)
        validation_x, validation_y, validation_manifest = generate_split("validation", args.validation,
            args.validation_seed, model, data, environment, rig, vision, args.dataset,
            fire_max_age=args.fire_max_age, oil_max_age=args.oil_max_age)
    train_cells = {tuple(item["locationCell"]) for item in train_manifest["examples"]}
    validation_cells = {tuple(item["locationCell"]) for item in validation_manifest["examples"]}
    if train_cells & validation_cells:
        raise RuntimeError("Location cells leaked between training and validation")
    weights, bias = fit_probe(train_x, train_y)
    encoder_path = Path(args.models) / "mobileclip-s0/vision_model.onnx"
    encoder_hash = file_hash(encoder_path)
    metadata = {"version": 1, "architecture": "Frozen MobileCLIP-S0 normalized 512D embedding + regularized multinomial linear probe",
        "encoderSha256": encoder_hash, "classes": list(CLASSES), "trainingSeed": args.train_seed,
        "preprocessorSha256": file_hash(Path(args.models) / "mobileclip-s0/preprocessor_config.json"),
        "validationSeed": args.validation_seed, "regularization": .04, "epochs": 1200,
        "training": evaluate(train_x, train_y, weights, bias),
        "validation": evaluate(validation_x, validation_y, weights, bias),
        "trainingRgbSha256": train_manifest["rgbSha256"], "validationRgbSha256": validation_manifest["rgbSha256"],
        "releaseCellsDisjoint": True, "trainingCells": sorted(map(list, train_cells)),
        "splitScope": "Disjoint incident release-center cells and independent random seeds; backgrounds and adjacent camera footprints share this yard",
        "validationCells": sorted(map(list, validation_cells)), "elapsedSeconds": round(time.monotonic() - started, 2),
        "mujocoVersion": mujoco.__version__, "cameraResolution": [320, 240],
        "fieldAgeRangesSeconds": {"fire": [0., args.fire_max_age], "oil": [0., args.oil_max_age]},
        "cropMethod": "Image-change connected components with 18px context, matching MissionTracker",
        "scope": "Held-out rendered construction-yard views only; no real-world recognition validation",
        "runtimeInputs": "Normalized image embedding only; simulator metadata is never a feature",
        "scoreMeaning": "Linear-probe softmax score; uncalibrated outside this simulator",
        "abstention": {"confidenceThreshold": .70, "marginThreshold": .20,
                       "basis": "Fixed conservative settings chosen before held-out evaluation; not calibrated"}}
    save_probe(args.output, weights, bias, metadata, validation_x, validation_y)
    np.savez_compressed(args.dataset / "embeddings.npz", train_x=train_x, train_y=train_y,
                        validation_x=validation_x, validation_y=validation_y)
    report = args.output.with_suffix(".metrics.json")
    report.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved learned classifier to {args.output} and metrics to {report}", flush=True)


if __name__ == "__main__":
    main()
