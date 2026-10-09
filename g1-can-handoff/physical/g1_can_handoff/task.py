"""Observed-pose grasp, lift verification, and offer. Release requires acceptance."""

import time
import numpy as np

OPEN = [0, 0, 0, 0, 0, 0, 0]
CLOSED = [0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0]


def plan(plant, observation, validate=None, include_offer=True):
    """Select a grasp; validate must check the complete arm and finger path.

    The physical caller supplies PhysicalPlant.preflight through prepare.
    A caller without that full-path contract must leave validate unset.
    """
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
    table_offset = getattr(plant, "planning_table_height_offset_m", 0.0)
    # Apply the assumed table correction to targets, without altering the
    # camera observation used by stationary-can and table movement checks.
    if not np.isfinite(table_offset) or not 0 <= table_offset <= 0.03:
        raise ValueError("Invalid planning table height offset")
    if not 0.645 <= observation.table_height + table_offset <= 0.955:
        raise ValueError(
            "Adjusted planning table height is outside the supported workspace"
        )
    # Recorded 25.7 cm forward and 6.8 cm left scenes pass complete
    # measured-start physical preflight.
    # Physical admission stays bounded and still requires IK,
    # commissioned joint limits, positive table clearance and both finger paths.
    if not (0.255 <= c[0] <= 0.45 and 0.065 <= c[1] <= 0.29):
        raise ValueError(
            "Can is outside the left-hand pickup area: "
            f"measured forward={c[0]*100:.1f} cm, robot-left={c[1]*100:.1f} cm; "
            "requires forward=25.5–45 cm and robot-left=6.5–29 cm. "
            "Reposition the can inside that area and recheck perception before starting."
        )
    # The cylinder permits yaw changes. A downward approach expands the
    # reachable workspace on lower tables without changing the can location.
    deadline = time.monotonic() + 70
    orientations = [
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
    ]
    step = getattr(plant, "grasp_orientation_step_rad", 0.0)
    if step:
        orientations = [(0, 0), (0, step), (0, -step)] + orientations[1:]
    # Prefer the cylinder centre; raising the grasp increases slip leverage.
    # Higher candidates are still checked against the positive table margin.
    heights = (0.0,) if include_offer else (0.0, 0.01, 0.02)
    candidate_budget = (
        getattr(plant, "grasp_candidate_budget", 1) if validate is not None else 1
    )
    feasible = []
    seen_orientations = set()
    for pitch, yaw, height in [(p, y, h) for p, y in orientations for h in heights]:
        if time.monotonic() > deadline:
            raise ValueError("Grasp orientation search time limit reached")
        if candidate_budget > 1 and (pitch, yaw) in seen_orientations:
            continue
        cp, sp, cy, sy = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array(
            [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
        )
        offset = rotation @ np.array([0, 0.10, 0])
        grasp = c + np.array([0, 0, height + table_offset])
        waypoints = [
            grasp + offset + [0, 0, 0.16],
            grasp + offset,
            grasp,
            grasp + [0, 0, 0.15],
            grasp + [0.02, 0, 0.15],
        ]
        # A full-path validator solves sequentially from the measured start,
        # including Cartesian transfers, fingers and every joint segment.
        # Independent endpoint solves both duplicate that work and can select
        # a different IK branch, discarding a feasible continuous path before
        # it is checked. Backends without a validator retain endpoint checks.
        if validate is None:
            try:
                for phase, target in zip(
                    ("approach", "align", "reach", "lift", "offer"),
                    waypoints[: 5 if include_offer else 4],
                ):
                    plant.set_contact_phase(phase)
                    plant.solve(target, rotation)
            except ValueError:
                continue
        plant.set_contact_phase("pregrasp")
        if validate is not None and not validate(waypoints, rotation):
            continue
        if candidate_budget == 1:
            return waypoints, rotation
        feasible.append((plant.grasp_noise_score(), waypoints, rotation))
        seen_orientations.add((pitch, yaw))
        if len(feasible) >= candidate_budget:
            break
    if feasible:
        if time.monotonic() > deadline:
            raise ValueError("Grasp orientation search time limit reached")
        score, waypoints, rotation = min(feasible, key=lambda candidate: candidate[0])
        # A later candidate may have replaced staging/collision state. Revalidate
        # the selected path before returning; no unvalidated cached path is reused.
        if not validate(waypoints, rotation):
            raise ValueError("Selected noise-aware grasp failed final path validation")
        if time.monotonic() > deadline:
            raise ValueError("Grasp orientation search time limit reached")
        if hasattr(plant, "event"):
            plant.event(
                "noise_aware_grasp_selected",
                candidate_count=len(feasible),
                predicted_joint_noise_rms_rad=score,
            )
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
        plant.set_contact_phase(phase)
        event(phase)
        plant.move(target, rotation=rotation)
    plant.set_contact_phase("close")
    event("close")
    plant.hand(CLOSED, 1.2)
    if len(plant.contacts()) < 2:
        raise ValueError("Grasp failed: fewer than two fingers contact the can")
    plant.set_contact_phase("lift")
    event("lift")
    plant.move(waypoints[3], 2.0, rotation)
    plant.step(0.5)
    if (
        plant.can_position()[2] < observation.table_height + 0.14
        or len(plant.contacts()) < 2
    ):
        raise ValueError("Grasp failed: can did not remain held above the table")
    plant.set_contact_phase("offer")
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
