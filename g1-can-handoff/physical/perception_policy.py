"""Perception scheduling, independent of continuously checked robot feedback."""
import os


def perception_policy():
    value=os.getenv('HANDOFF_PERCEPTION_POLICY','phase_boundary')
    if value not in ('phase_boundary','continuous'):
        raise ValueError('HANDOFF_PERCEPTION_POLICY must be phase_boundary or continuous')
    return value


def semantic_recheck_interval_s():
    return None if perception_policy()=='phase_boundary' else 1.0
