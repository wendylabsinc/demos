"""Units shared by the simulator, planner and real Ackermann driver."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Command:
    speed: float = 0.0  # metres per second, negative means reverse
    steering: float = 0.0  # radians, positive means left


@dataclass
class Observation:
    pose: Pose
    ranges: list[float]
    angle_min: float = -math.pi
    angle_increment: float = 2 * math.pi / 180
    range_max: float = 6.0
    captured_at: float = 0.0  # local monotonic receipt/capture time
    people: list[dict] = field(default_factory=list)
    collision: bool = False
