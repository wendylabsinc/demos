"""Perception scheduling, independent of continuously checked robot feedback."""

import os


def post_grasp_camera_checks():
    """0 finishes on robot feedback; 1 retains diagnostic visual lift checks."""
    value = os.getenv("HANDOFF_POST_GRASP_CAMERA_CHECKS", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_POST_GRASP_CAMERA_CHECKS must be 0 or 1")
    return value == "1"


def perception_policy():
    value = os.getenv("HANDOFF_PERCEPTION_POLICY", "phase_boundary")
    if value not in ("phase_boundary", "continuous"):
        raise ValueError(
            "HANDOFF_PERCEPTION_POLICY must be phase_boundary or continuous"
        )
    return value


def semantic_recheck_interval_s():
    return (
        None
        if perception_policy() == "phase_boundary"
        or target_identity_mode() != "yolo_reacquire"
        else 1.0
    )


def target_identity_mode():
    value = os.getenv("HANDOFF_TARGET_IDENTITY", "yolo_reacquire")
    if value not in ("color_depth", "yolo_once", "yolo_reacquire"):
        raise ValueError(
            "HANDOFF_TARGET_IDENTITY must be color_depth, yolo_once or yolo_reacquire"
        )
    return value
