"""Stationary pickup sequence shared by physical and simulated G1 backends.

Perception and plant feedback are supplied by the backend. No object pose or
simulation contacts bypass the measured-state or camera verification gates.
"""

import numpy as np
from g1_can_handoff.task import plan, OPEN, CLOSED


def prepare(plant, observe, event):
    """Observe, select and fully validate a grasp; refresh before using the plan."""
    observation, baseline = observe("planning", True)
    plant.observe_can(observation)
    rejected = []

    def validate_candidate(points, rotation):
        try:
            plant.preflight(points, rotation)
            return True
        except (RuntimeError, ValueError) as exc:
            # Only geometric/search failures permit another grasp candidate.
            # Telemetry, stop, temperature and posture errors abort immediately.
            geometric = (
                "Interpolated physical trajectory intersects",
                "Planned finger trajectory intersects",
                "Continuous Cartesian lift IK",
                "Target pose collision",
                "Target cannot be reached",
                "Staging endpoint collides",
                "Staging planning deadline reached",
                "Staging planning node limit reached",
                "No staging path within bounded search",
            )
            if not str(exc).startswith(geometric):
                raise
            reason = str(exc).split(":", 1)[0]
            if reason not in rejected:
                rejected.append(reason)
            event(
                "grasp_candidate_rejected", rotation=rotation.tolist(), reason=str(exc)
            )
            return False

    try:
        points, rotation = plan(
            plant, observation, validate=validate_candidate, include_offer=False
        )
    except ValueError as exc:
        if rejected:
            raise ValueError(
                str(exc) + "; preflight rejected: " + "; ".join(rejected)
            ) from exc
        raise
    refreshed, track = observe("plan-refresh", True)
    if (
        np.linalg.norm(refreshed.center - observation.center) > 0.01
        or abs(refreshed.table_height - observation.table_height) > 0.005
    ):
        raise RuntimeError("Can or table moved during planning")
    event(
        "plan",
        waypoints=[p.tolist() for p in points],
        rotation=rotation.tolist(),
        observation=track,
        measured_table_height_m=float(observation.table_height),
        planning_table_height_offset_m=float(
            getattr(plant, "planning_table_height_offset_m", 0.0)
        ),
        planning_table_height_m=float(
            observation.table_height
            + getattr(plant, "planning_table_height_offset_m", 0.0)
        ),
    )
    return observation, baseline, points, rotation


def execute(plant, prepared, observe, event):
    """Run the checked pickup phases and verify the lift with fresh camera data."""
    observation, baseline, points, rotation = prepared
    plant.validate_start()
    plant.set_contact_phase("pregrasp")

    def checkpoint(phase, message):
        fresh, vision = observe(phase, True)
        if (
            np.linalg.norm(fresh.center - observation.center) > 0.01
            or abs(fresh.table_height - observation.table_height) > 0.005
        ):
            raise RuntimeError(message)
        plant.observe_can(fresh)

    checkpoint("pre-open", "Can or table moved before opening; replanning required")
    plant.hand(OPEN, 0.8)
    checkpoint("pre-stage", "Can or table moved before staging; replanning required")
    plant.stage()
    for phase, point in zip(("approach", "align", "reach"), points[:3]):
        checkpoint(phase, "Can or table moved during approach; replanning required")
        plant.set_contact_phase(phase)
        event(phase)
        plant.move(point, rotation=rotation)
    checkpoint("pre-close", "Can or table moved during final reach; refusing to close")
    plant.set_contact_phase("close")
    event("close")
    plant.hand(CLOSED, 1.2)
    plant.step(0.5)
    # Closing contact can still be settling after the nominal half second.
    # Require the same two-finger stationary evidence for three consecutive
    # samples, with a bounded additional two seconds. Never relax the guard.
    stationary = 0
    for attempt in range(21):
        stationary = stationary + 1 if len(plant.contacts()) >= 2 else 0
        if stationary >= 3:
            break
        if attempt < 20:
            plant.step(0.1)
    else:
        raise RuntimeError("Insufficient stable finger obstruction candidates")
    # Once the fingers close, use the existing post-grasp tracking contract:
    # the can may rise/settle in the fingers, but must remain within 40 mm in XY.
    # The pregrasp 10 mm full-position/table check no longer describes this state.
    fresh, vision = observe(
        "pre-lift", False, expected_center_xy=observation.center[:2]
    )
    if np.linalg.norm(fresh.center[:2] - observation.center[:2]) > 0.04:
        raise RuntimeError(
            "Grasped can moved outside the checked pickup path before lifting"
        )
    plant.observe_can(fresh)
    plant.set_contact_phase("lift")
    event("lift")
    plant.move(points[3], rotation=rotation)
    plant.step(0.5)
    rises = []
    previous_frame = None
    for sample in range(5):
        if sample:
            plant.step(0.25)
        phase = "lift-verification" if sample == 0 else f"hold-verification-{sample}"
        lifted, vision = observe(phase, False, expected_center_xy=points[3][:2])
        frame_id = vision.get("frame_id")
        if frame_id is not None and frame_id == previous_frame:
            raise RuntimeError("Camera repeated a frame during lift hold verification")
        previous_frame = frame_id
        rise = vision["top_quantile_z_m"] - baseline["top_quantile_z_m"]
        if (
            rise < 0.11
            or np.linalg.norm(np.array(vision["center_xy_m"]) - points[3][:2]) > 0.04
        ):
            raise RuntimeError("Independent camera did not verify the lift and hold")
        rises.append(float(rise))
        event(
            "lift_observation", sample=sample, visual_rise_m=float(rise), vision=vision
        )
    event("lift_verified", visual_rise_m=min(rises), hold_verified_s=1.0, vision=vision)
    return {"lift_verified": True, "visual_rise_m": min(rises), "hold_verified_s": 1.0}
