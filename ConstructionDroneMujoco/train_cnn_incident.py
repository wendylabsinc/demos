"""Train a compact CNN on existing actual camera crops; evaluate held-out views once."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn

from yard_drone.cnn import file_hash, image_tensor

CLASSES = ("fire", "oil", "other")


class IncidentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Linear(64 * 2 * 2, 3)

    def forward(self, rgb):
        # The exported graph owns normalization. Its input is float RGB pixels
        # in [0,255], matching the image_tensor used for training and runtime.
        features = self.features(rgb / 255.)
        return self.classifier(torch.flatten(features, 1))


def read_split(directory):
    rows = []
    for label, name in enumerate(CLASSES):
        for path in sorted((directory / name).glob("*.png")):
            rows.append((path, label))
    if len(rows) < 12 or any(not any(label == index for _, label in rows) for index in range(3)):
        raise ValueError(f"Insufficient labeled RGB images in {directory}")
    return rows


def internal_split(rows, seed):
    rng = np.random.default_rng(seed)
    fit, selection = [], []
    for label in range(3):
        indices = [index for index, (_, category) in enumerate(rows) if category == label]
        rng.shuffle(indices)
        count = max(2, round(len(indices) * .2))
        selection.extend(rows[index] for index in indices[:count])
        fit.extend(rows[index] for index in indices[count:])
    return fit, selection


def class_weights(rows):
    counts = np.bincount([label for _, label in rows], minlength=len(CLASSES)).astype(float)
    if np.any(counts == 0):
        raise ValueError("CNN training requires every class")
    weights = 1. / counts
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def batch_images(rows, size, *, augment=False, rng=None):
    images = []
    for path, _ in rows:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        if augment:
            if rng.random() < .5:
                image = ImageOps.mirror(image)
            image = image.rotate(rng.uniform(-25, 25), resample=Image.Resampling.BILINEAR,
                                 fillcolor=tuple(int(v) for v in np.asarray(image).mean(axis=(0, 1))))
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(.80, 1.20))
        images.append(image_tensor(image, size)[0])
    return torch.from_numpy(np.stack(images)), torch.tensor([label for _, label in rows], dtype=torch.long)


def probabilities(logits, temperature=1.):
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    values = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def evaluate(logits, labels, temperature=1., confidence=.0, margin=.0, domain_mask=None):
    scores = probabilities(logits, temperature)
    predictions = scores.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=int)
    for truth, prediction in zip(labels, predictions):
        matrix[int(truth), int(prediction)] += 1
    ordered = np.sort(scores, axis=1)
    accepted = (ordered[:, -1] >= confidence) & (ordered[:, -1] - ordered[:, -2] >= margin)
    if domain_mask is not None:
        accepted &= np.asarray(domain_mask, dtype=bool)
    return {"examples": len(labels), "accuracy": float(np.mean(predictions == labels)),
            "classes": list(CLASSES), "confusionMatrix": matrix.tolist(),
            "acceptedExamples": int(accepted.sum()), "coverage": float(accepted.mean()),
            "outOfDomainExamples": int(np.count_nonzero(~np.asarray(domain_mask, dtype=bool))) if domain_mask is not None else 0,
            "acceptedAccuracy": float(np.mean(predictions[accepted] == labels[accepted])) if accepted.any() else None,
            "perClassRecall": {name: float(matrix[i, i] / max(1, matrix[i].sum())) for i, name in enumerate(CLASSES)}}


def calibration(logits, labels):
    choices = [.5, .75, 1., 1.5, 2., 3.]
    loss = lambda t: float(-np.log(np.maximum(probabilities(logits, t)[np.arange(len(labels)), labels], 1e-12)).mean())
    temperature = min(choices, key=loss)
    choices = []
    for confidence in (.50, .55, .60, .65, .70, .75, .80, .85, .90):
        for margin in (.05, .10, .15, .20, .30):
            metrics = evaluate(logits, labels, temperature, confidence, margin)
            if metrics["acceptedAccuracy"] is not None and metrics["acceptedAccuracy"] >= .90 and metrics["acceptedExamples"] >= 6:
                choices.append((metrics["coverage"], -confidence, -margin, confidence, margin))
    if choices:
        _, _, _, confidence, margin = max(choices)
        basis = "Maximum coverage with at least 90% accepted accuracy on train-only internal selection images"
    else:
        confidence, margin = .90, .30
        basis = "Internal selection too weak for calibration; conservative fallback, no accuracy guarantee"
    return temperature, confidence, margin, {"basis": basis,
        "temperatureSelection": "Minimum NLL on train-only internal selection images",
        "selection": evaluate(logits, labels, temperature, confidence, margin)}


def rgb_hash(rows):
    digest = hashlib.sha256()
    for path, label in rows:
        digest.update(CLASSES[label].encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def crop_aspects(rows):
    aspects = []
    for path, _ in rows:
        with Image.open(path) as image:
            width, height = image.size
        aspects.append(max(width / height, height / width))
    return np.asarray(aspects, dtype=float)


def dataset_metadata(training, validation):
    """Carry offline generation conditions into the deployed model artifact."""
    field_ages = training.get("fieldAgeRangesSeconds")
    if field_ages is None and training.get("maxFieldAgeSeconds") is not None:
        field_ages = {name: [0., training["maxFieldAgeSeconds"]] for name in CLASSES[:2]}
    return {"generator": training.get("generator", "train_incident_classifier.py"),
        "rendererBackend": training.get("rendererBackend", training.get("renderBackend")),
        "fieldAgeRangesSeconds": field_ages,
        "trainingSeed": training.get("seed"), "validationSeed": validation.get("seed"),
        "cameraOffsets": training.get("cameraOffsets"),
        "contextPixels": training.get("contextPixels"),
        "contextProbabilities": training.get("contextProbabilities"),
        "poseJitter": training.get("poseJitter"),
        "groundTruth": training.get("groundTruth"),
        "runtimeTestFramesUsed": training.get("runtimeTestFramesUsed"),
        "scope": training.get("scope")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("yard_drone/incident_cnn.onnx"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--size", type=int, choices=(64, 96), default=96)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    if device == "auto": device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; use --device cpu")
    training_rows = read_split(args.dataset / "train")
    validation_rows = read_split(args.dataset / "validation")
    train_manifest = json.loads((args.dataset / "train.json").read_text())
    validation_manifest = json.loads((args.dataset / "validation.json").read_text())
    train_cells = {tuple(row["locationCell"]) for row in train_manifest["examples"]}
    validation_cells = {tuple(row["locationCell"]) for row in validation_manifest["examples"]}
    if train_cells & validation_cells:
        raise ValueError("Incident release cells overlap between training and validation")
    fit_rows, selection_rows = internal_split(training_rows, args.seed)
    selection_x, selection_y = batch_images(selection_rows, args.size)
    selection_x, selection_y = selection_x.to(device), selection_y.to(device)
    model = IncidentCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=.0003)
    balancing = class_weights(fit_rows).to(device)
    criterion = nn.CrossEntropyLoss(weight=balancing)
    rng = random.Random(args.seed)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(args.epochs):
        model.train()
        order = list(fit_rows); rng.shuffle(order)
        losses = []
        for offset in range(0, len(order), 18):
            features, labels = batch_images(order[offset:offset+18], args.size, augment=True, rng=rng)
            optimizer.zero_grad(); logits = model(features.to(device))
            loss = criterion(logits, labels.to(device)); loss.backward(); optimizer.step()
            losses.append(loss.detach())
        schedule.step(); model.eval()
        with torch.no_grad():
            selection_logits = model(selection_x)
            selection_loss = float(criterion(selection_logits, selection_y))
        if selection_loss < best_loss:
            best_loss, best_epoch = selection_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            accuracy = float((selection_logits.argmax(1) == selection_y).float().mean())
            mean_loss = float(torch.stack(losses).mean())
            print(f"Epoch {epoch+1}/{args.epochs}: train loss {mean_loss:.4f}; train-only selection loss {selection_loss:.4f}, accuracy {accuracy:.3f}", flush=True)
    model.load_state_dict(best_state); model.eval(); model.to("cpu")
    with torch.no_grad():
        selection_logits = model(selection_x.cpu()).numpy()
    labels = selection_y.cpu().numpy()
    temperature, confidence, margin, calibration_info = calibration(selection_logits, labels)
    training_aspects = crop_aspects(training_rows)
    aspect_max = float(np.ceil(float(training_aspects.max()) * 10.) / 10.)
    # The external validation split is first inferred after checkpoint and
    # calibration selection finish. Its scores never affect either decision.
    train_x, train_y = batch_images(training_rows, args.size)
    validation_x, validation_y = batch_images(validation_rows, args.size)
    validation_aspects = crop_aspects(validation_rows)
    with torch.no_grad():
        train_logits = model(train_x).numpy()
        validation_logits = model(validation_x).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size), str(args.output),
        input_names=["rgb"], output_names=["logits"], opset_version=17, dynamo=False,
        dynamic_axes={"rgb": {0: "batch"}, "logits": {0: "batch"}})
    import onnx
    onnx.checker.check_model(str(args.output))
    metadata = {"version": 1, "classes": list(CLASSES), "imageSize": args.size,
        "architecture": "Conv16/32/64 + ReLU + MaxPool + adaptive 2x2 average + linear3",
        "normalization": "Float RGB [0,255], divide by 255 inside exported graph",
        "modelSha256": file_hash(args.output), "seed": args.seed, "trainingEpochs": args.epochs,
        "selectedEpoch": best_epoch, "checkpointSelection": "Minimum loss on train-only internal selection images",
        "trainingDevice": device, "learningRate": .001,
        "fitExamples": len(fit_rows), "selectionExamples": len(selection_rows),
        "classBalance": {"method": "Inverse class frequency in training fit images",
            "counts": {name: sum(label == i for _, label in training_rows) for i, name in enumerate(CLASSES)},
            "lossWeights": balancing.cpu().numpy().tolist()},
        "negativeAugmentation": {"train": train_manifest.get("negativeAugmentation"),
            "validation": validation_manifest.get("negativeAugmentation")},
        "dataset": dataset_metadata(train_manifest, validation_manifest),
        "temperature": temperature, "abstention": {"confidenceThreshold": confidence,
            "marginThreshold": margin, "basis": calibration_info["basis"]},
        "calibration": calibration_info,
        "training": evaluate(train_logits, train_y.numpy(), temperature, confidence, margin, training_aspects <= aspect_max),
        "validation": evaluate(validation_logits, validation_y.numpy(), temperature, confidence, margin, validation_aspects <= aspect_max),
        "inputDomain": {"aspectMax": aspect_max, "trainingAspectMax": float(training_aspects.max()),
            "basis": "Observed training and internal-selection crop aspect ratios, rounded upward to 0.1 before external validation"},
        "trainingRgbSha256": rgb_hash(training_rows), "validationRgbSha256": rgb_hash(validation_rows),
        "releaseCellsDisjoint": True, "trainingCells": sorted(map(list, train_cells)),
        "validationCells": sorted(map(list, validation_cells)),
        "augmentation": "Training fit only: horizontal reflection, rotation +/-25 degrees, brightness factor 0.8..1.2",
        "runtimeInputs": "RGB pixels only; no simulator state or pose features",
        "scope": "Held-out actual MuJoCo construction-yard camera crops; no real-world accuracy validation",
        "elapsedSeconds": round(time.monotonic()-started, 2), "torchVersion": torch.__version__}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
