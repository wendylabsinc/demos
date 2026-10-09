"""Fixed-shape image preprocessing shared by runtime and regression tests."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox_bgr(
    frame: np.ndarray, image_size_px: int
) -> tuple[np.ndarray, float, int, int]:
    """Match Ultralytics' centered square letterbox for a fixed-size engine."""
    height, width = frame.shape[:2]
    scale = min(image_size_px / width, image_size_px / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = (
        frame
        if (resized_width, resized_height) == (width, height)
        else cv2.resize(
            frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
    )
    pad_x = round((image_size_px - resized_width) / 2.0 - 0.1)
    pad_y = round((image_size_px - resized_height) / 2.0 - 0.1)
    canvas = np.full((image_size_px, image_size_px, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    chw_rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    return chw_rgb, scale, pad_x, pad_y
