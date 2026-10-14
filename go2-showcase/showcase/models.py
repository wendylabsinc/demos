"""Measured observations and bounded robot commands.

Coordinates are metres in an odometry/map frame; angles are radians. The
caller owns timestamps, sensor freshness and the final motor watchdog.
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    label: str
    x: float
    y: float
    confidence: float = 1.0
    source: str = ""
    timestamp: Optional[float] = None


@dataclass(frozen=True)
class Observation:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    fresh: bool = False
    obstacles: tuple = ()  # world x, y, obstacle radius
    detections: tuple = ()
    gesture: Optional[str] = None
    front_clearance: Optional[float] = None
    scan_fresh: bool = False
    sectors: tuple = ()  # measured robot-relative bearing, clearance
    front_coverage_complete: bool = True
    scan_coverage_complete: bool = True
    scan_coverage: float = 1.0
    scan_gaps_allowed: bool = False


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    posture: str = "stand"
    reason: str = "stopped"

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.vx, self.vy, self.yaw_rate)):
            raise ValueError("command must be finite")
        if not (0 <= self.vx <= 0.55 and self.vy == 0 and abs(self.yaw_rate) <= 0.6):
            raise ValueError("command exceeds tested limits")
