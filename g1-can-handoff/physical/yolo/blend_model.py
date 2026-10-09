"""Shrink the adapted class branch toward its rehearsal checkpoint."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapted-weight", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.adapted_weight <= 1:
        raise ValueError("Adapted weight must be in [0,1]")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing artifact")
    base = YOLO(args.rehearsal)
    adapted = YOLO(args.adapted)
    result = deepcopy(base)
    original = base.model.state_dict()
    other = adapted.model.state_dict()
    if original.keys() != other.keys():
        raise ValueError("Checkpoint architectures differ")
    for key, value in result.model.state_dict().items():
        if not key.startswith("model.22.cv3."):
            if not torch.equal(original[key], other[key]):
                raise ValueError("Features or box regression changed: " + key)
        elif value.is_floating_point():
            value.copy_(
                (1 - args.adapted_weight) * original[key]
                + args.adapted_weight * other[key]
            )
    result.save(args.output)


if __name__ == "__main__":
    main()
