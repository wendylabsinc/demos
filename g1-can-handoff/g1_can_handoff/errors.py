"""Geometric IK rejection, distinct from feedback and operator faults."""


class IKError(ValueError):
    """A target has no pose-accurate, collision-free IK candidate."""
