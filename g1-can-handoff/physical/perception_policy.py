"""Perception scheduling, independent of continuously checked robot feedback."""
import os


def perception_policy():
    value=os.getenv('HANDOFF_PERCEPTION_POLICY','phase_boundary')
    if value not in ('phase_boundary','continuous'):
        raise ValueError('HANDOFF_PERCEPTION_POLICY must be phase_boundary or continuous')
    return value


def semantic_recheck_interval_s():
    return None if perception_policy()=='phase_boundary' or target_identity_mode()!='yolo_reacquire' else 1.0


def target_identity_mode():
    value=os.getenv('HANDOFF_TARGET_IDENTITY','yolo_reacquire')
    if value not in ('color_depth','yolo_once','yolo_reacquire'):
        raise ValueError('HANDOFF_TARGET_IDENTITY must be color_depth, yolo_once or yolo_reacquire')
    return value
