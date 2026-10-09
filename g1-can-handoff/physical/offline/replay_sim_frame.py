"""Replay one synthetic RGB-D failure through real color/depth reacquisition.

This pins the clock to the original exposure. It does not reconstruct earlier
tracker history, camera transport, or physical feedback. It never authorizes IO.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from camera_input import geometry
from target_tracker import TargetTracker


def replay(path):
    with np.load(path, allow_pickle=False) as archive:
        saved = json.loads(str(archive["metadata_json"]))
        if saved.get("synthetic_fixture") is not True:
            raise ValueError("Expected a labelled synthetic simulation frame")
        frame = {key: archive[key].copy() for key in ("color_bgr", "depth_m", "rays")}
        frame.update(timestamp=saved["timestamp"], meta=saved["meta"])
    result = dict(
        source=str(path),
        phase=saved["phase"],
        simulation_only=True,
        clock_pinned=True,
        scope="single-frame color/depth reacquisition; no previous tracker history",
        physical_motion=False,
    )
    with (
        patch("yolo_client.time.monotonic", return_value=frame["timestamp"] + 0.05),
        patch(
            "yolo_client.YoloClient.detect",
            side_effect=AssertionError("Unexpected YOLO call"),
        ),
    ):
        try:
            observation, vision = geometry(
                frame,
                saved["config"],
                require_table=saved["require_table"],
                expected_center_xy=saved["expected_center_xy"],
                tracker=TargetTracker(identity_mode="color_depth"),
            )
            result.update(accepted=True, vision=vision)
        except (RuntimeError, ValueError) as exc:
            result.update(accepted=False, error=str(exc))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    args = parser.parse_args()
    result = replay(args.frame)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["accepted"] else 1)
