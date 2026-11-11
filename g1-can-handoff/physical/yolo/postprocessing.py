"""Low-overhead one-class detection decoding for the fixed TensorRT output."""
from __future__ import annotations

import numpy as np


def box_iou(first: dict, second: dict) -> float:
    """Return intersection-over-union for two ``x1,y1,x2,y2`` mappings."""
    x1 = max(float(first["x1"]), float(second["x1"]))
    y1 = max(float(first["y1"]), float(second["y1"]))
    x2 = min(float(first["x2"]), float(second["x2"]))
    y2 = min(float(first["y2"]), float(second["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first["x2"]) - float(first["x1"])) * max(
        0.0, float(first["y2"]) - float(first["y1"])
    )
    second_area = max(0.0, float(second["x2"]) - float(second["x1"])) * max(
        0.0, float(second["y2"]) - float(second["y1"])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def decode_one_class_candidates(
    candidates: np.ndarray,
    *,
    confidence: float,
    nms_iou: float,
    max_detections: int,
    frame_width: int,
    frame_height: int,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> list[dict]:
    """Decode top ``cx,cy,w,h,score`` rows and perform small CPU NMS.

    The GPU has already reduced the fixed 8,400-anchor output to a bounded
    candidate set. CPU NMS avoids the several-launch overhead of generalized
    GPU NMS while retaining distinct Coke cans in the same frame.
    """
    values = np.asarray(candidates, dtype=np.float32)
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError(f"expected candidate shape [N,5], got {values.shape}")
    finite = np.isfinite(values).all(axis=1)
    values = values[finite & (values[:, 4] >= float(confidence))]
    if not len(values):
        return []

    xyxy = np.empty((len(values), 4), dtype=np.float32)
    xyxy[:, 0] = values[:, 0] - values[:, 2] / 2.0
    xyxy[:, 1] = values[:, 1] - values[:, 3] / 2.0
    xyxy[:, 2] = values[:, 0] + values[:, 2] / 2.0
    xyxy[:, 3] = values[:, 1] + values[:, 3] / 2.0
    scores = values[:, 4]
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(xyxy[current, 0], xyxy[remaining, 0])
        yy1 = np.maximum(xyxy[current, 1], xyxy[remaining, 1])
        xx2 = np.minimum(xyxy[current, 2], xyxy[remaining, 2])
        yy2 = np.minimum(xyxy[current, 3], xyxy[remaining, 3])
        intersections = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        current_area = max(0.0, xyxy[current, 2] - xyxy[current, 0]) * max(
            0.0, xyxy[current, 3] - xyxy[current, 1]
        )
        remaining_areas = np.maximum(0.0, xyxy[remaining, 2] - xyxy[remaining, 0]) * np.maximum(
            0.0, xyxy[remaining, 3] - xyxy[remaining, 1]
        )
        unions = current_area + remaining_areas - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections),
            where=unions > 0.0,
        )
        order = remaining[ious <= float(nms_iou)]

    detections: list[dict] = []
    for index in keep:
        x1 = max(0.0, min(frame_width, (float(xyxy[index, 0]) - pad_x) / scale))
        y1 = max(0.0, min(frame_height, (float(xyxy[index, 1]) - pad_y) / scale))
        x2 = max(0.0, min(frame_width, (float(xyxy[index, 2]) - pad_x) / scale))
        y2 = max(0.0, min(frame_height, (float(xyxy[index, 3]) - pad_y) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": float(scores[index]),
            }
        )
    return detections
