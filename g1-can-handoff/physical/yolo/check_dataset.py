"""Measure fixed-floor one-class precision/recall on original held-out splits."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from ultralytics import YOLO

from recognition_check import iou
from confidence import MIN_CONFIDENCE


def measure(model, dataset, split, size, confidence=MIN_CONFIDENCE):
    tp = fp = fn = 0
    images = sorted((dataset / split / "images").glob("*"))
    if not images:
        raise ValueError(f"No held-out images in {dataset/split}")
    for image in images:
        width, height = Image.open(image).size
        labels = dataset / split / "labels" / (image.stem + ".txt")
        expected = []
        for line in labels.read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            if cls != 0:
                raise ValueError("Expected one Coke class")
            expected.append(
                [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ]
            )
        result = model.predict(
            str(image),
            device="cpu",
            imgsz=size,
            rect=False,
            conf=confidence,
            iou=0.7,
            verbose=False,
        )[0]
        remaining = set(range(len(expected)))
        for box in result.boxes:
            if float(box.conf[0]) <= confidence:
                continue
            overlaps = [(iou(box.xyxy[0].numpy(), expected[i]), i) for i in remaining]
            overlap, index = max(overlaps, default=(0, None))
            if overlap >= 0.5:
                tp += 1
                remaining.remove(index)
            else:
                fp += 1
        fn += len(remaining)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else 0,
        recall=tp / (tp + fn) if tp + fn else 0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=Path(__file__).parent / "model/coke-can.pt"
    )
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=MIN_CONFIDENCE)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not MIN_CONFIDENCE <= args.confidence <= 0.99:
        parser.error("Confidence must be within current floor..0.99")
    rows = {"confidence_strictly_above": args.confidence}
    for name, weights, size in [
        ("baseline640", args.baseline, 640),
        ("baseline_same_size", args.baseline, args.size),
        ("candidate", args.candidate, args.size),
    ]:
        model = YOLO(weights)
        rows[name] = {
            split: measure(model, args.dataset, split, size, args.confidence)
            for split in ("valid", "test")
        }
        print(json.dumps({name: rows[name]}), flush=True)
    passed = all(
        rows["candidate"][split][metric] >= rows["baseline640"][split][metric] - 0.05
        for split in ("valid", "test")
        for metric in ("precision", "recall")
    )
    rows["regression_gate_passed"] = passed
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
