"""Validated camera runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


@dataclass(frozen=True)
class CameraSettings:
    serial: str
    width_px: int
    height_px: int
    fps: int
    jpeg_quality: int
    frame_timeout_limit: int
    retry_delay_s: float
    hardware_reset_after_failures: int
    freshness_limit_s: float
    lease_timeout_s: float
    port: int
    rgbd_enabled: bool = False
    rgbd_buffer_frames: int = 16

    @classmethod
    def from_env(cls) -> CameraSettings:
        return cls(
            serial=os.environ.get("G1_REALSENSE_SERIAL", "337122071091").strip(),
            width_px=_integer("G1_REALSENSE_WIDTH", 640, 160, 1920),
            height_px=_integer("G1_REALSENSE_HEIGHT", 480, 120, 1080),
            fps=_integer("G1_REALSENSE_FPS", 15, 1, 60),
            jpeg_quality=_integer("G1_REALSENSE_JPEG_QUALITY", 90, 50, 100),
            frame_timeout_limit=_integer("G1_REALSENSE_FRAME_TIMEOUT_LIMIT", 3, 1, 30),
            retry_delay_s=_number("G1_REALSENSE_RETRY_DELAY_S", 2.0, 0.1, 30.0),
            hardware_reset_after_failures=_integer(
                "G1_REALSENSE_HARDWARE_RESET_AFTER_FAILURES", 0, 0, 100
            ),
            freshness_limit_s=_number("G1_REALSENSE_FRESHNESS_LIMIT_S", 1.5, 0.1, 10.0),
            lease_timeout_s=_number("G1_REALSENSE_LEASE_TIMEOUT_S", 5.0, 1.0, 30.0),
            port=_integer("CAMERA_PORT", 8001, 1, 65535),
            rgbd_enabled=bool(_integer("G1_REALSENSE_RGBD_ENABLED", 0, 0, 1)),
            rgbd_buffer_frames=_integer("G1_REALSENSE_RGBD_BUFFER_FRAMES", 16, 1, 64),
        )
