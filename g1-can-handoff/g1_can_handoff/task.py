"""Observed-pose grasp, lift verification, and offer. Release requires acceptance."""

import time
import numpy as np

OPEN = [0, 0, 0, 0, 0, 0, 0]
CLOSED = [0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0]


def plan(plant, observation):
    c = np.asarray(observation.center, dtype=float)
    if (
        c.shape != (3,)
        or not np.all(np.isfinite(c))
        or not np.isfinite(observation.table_height)
    ):
        raise ValueError("Invalid observed can pose")
    if time.monotonic() - observation.timestamp > 0.5:
        raise ValueError("Observation expired before planning")
    if not (0.645 <= observation.table_height <= 0.955):
        raise ValueError("Table height is outside the supported 65–95 cm range")
    if not (0.27 <= c[0] <= 0.45 and 0.09 <= c[1] <= 0.29):
        raise ValueError(
            "Can is outside the left-hand pickup area: "
            f"measured forward={c[0]*100:.1f} cm, robot-left={c[1]*100:.1f} cm; "
            "requires forward=27–45 cm and robot-left=9–29 cm. "
            "Reposition the can inside that area and recheck perception before starting."
        )
    # The cylinder permits yaw changes. A downward approach expands the
    # reachable workspace on lower tables without changing the can location.
    for pitch, yaw in [
        (0, 0),
        (0, 0.5),
        (0, 1.0),
        (-0.4, 0),
        (-0.4, 0.5),
        (0.4, 0),
        (0.4, 0.5),
        (0.7, 0),
        (0.7, 0.5),
        (0.7, 1.0),
    ]:
        cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
            [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        )
        offset = rotation @ np.array([0, 0.05, 0])
        waypoints = [
            c + offset + [0, 0, 0.16],
            c + offset,
            c,
            c + [0, 0, 0.15],
            c + [0.02, 0, 0.15],
        ]
        try:
            for target in waypoints:
                plant.solve(target, rotation)
        except ValueError:
            continue
        return waypoints, rotation
    raise ValueError("No reachable grasp and offer path for this can position")


def execute(plant, observation, event=lambda phase: None, observe=None):
    waypoints, rotation = plan(plant, observation)
    # Planning may take longer than the camera age limit. Observe again before
    # moving, and reject a moved can rather than executing an old plan.
    if observe:
        current = observe()
        if (
            np.linalg.norm(current.center - observation.center) > 0.01
            or abs(current.table_height - observation.table_height) > 0.005
        ):
            raise ValueError(
                "Can or table moved during planning; a new plan is required"
            )
        observation = current
    elif time.monotonic() - observation.timestamp > 0.5:
        raise ValueError("Observation expired during planning; refresh before moving")
    plant.hand(OPEN, 0.3)
    for phase, target in zip(("approach", "align", "reach"), waypoints[:3]):
        event(phase)
        plant.move(target, rotation=rotation)
    event("close")
    plant.hand(CLOSED, 1.2)
    if len(plant.contacts()) < 2:
        raise ValueError("Grasp failed: fewer than two fingers contact the can")
    event("lift")
    plant.move(waypoints[3], 2.0, rotation)
    plant.step(0.5)
    if (
        plant.can_position()[2] < observation.table_height + 0.14
        or len(plant.contacts()) < 2
    ):
        raise ValueError("Grasp failed: can did not remain held above the table")
    event("offer")
    plant.move(waypoints[4], 2.0, rotation)
    plant.step(1.0)
    if (
        plant.can_position()[2] < observation.table_height + 0.14
        or len(plant.contacts()) < 2
    ):
        raise ValueError("Can slipped during offer")
    event("waiting_for_acceptance")
    return {
        "status": "waiting_for_acceptance",
        "can_height": float(plant.can_position()[2]),
        "contacts": plant.contacts(),
    }


def accept(plant, recipient_supports_can=False):
    if not recipient_supports_can:
        raise ValueError("Release requires confirmed recipient support")
    plant.hand(OPEN, 1.0)
