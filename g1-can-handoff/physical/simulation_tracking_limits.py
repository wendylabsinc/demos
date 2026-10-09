"""Tracking tolerance experiments for SimulationIO only, never hardware I/O."""

import math
import os
from pathlib import Path
from physical_core import PhysicalPlant

FIELDS = (
    "arm_tracking_error_rad",
    "arm_settle_error_rad",
    "starting_arm_error_rad",
    "target_position_error_m",
    "settling_arm_span_rad",
)
ARM_ORDER = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
GLOBAL_ENV = "SIM_TRACKING_GLOBAL_MULTIPLIER"
ENV_FIELDS = {
    "SIM_TRACKING_ARM_ERROR_MULTIPLIERS": ("arm_tracking_error_rad", 7),
    "SIM_TRACKING_ARM_SETTLE_MULTIPLIERS": ("arm_settle_error_rad", 7),
    "SIM_TRACKING_START_POSTURE_MULTIPLIERS": ("starting_arm_error_rad", 7),
    "SIM_TRACKING_SETTLE_SPAN_MULTIPLIERS": ("settling_arm_span_rad", 14),
    "SIM_TRACKING_MOTOR_LEASH_MULTIPLIERS": ("motor_reference_leash_rad", 14),
    "SIM_TRACKING_FK_MULTIPLIER": ("target_position_error_m", 1),
}


def environment_file(path=None):
    values = dict(os.environ)
    if path is not None:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            key, value = key.strip(), value.split("#", 1)[0].strip()
            if not separator or key not in (*ENV_FIELDS, GLOBAL_ENV):
                raise ValueError(
                    f"Invalid simulation tracking environment entry: {key}"
                )
            values[key] = value
    return values


def multipliers(value, count, key):
    try:
        numbers = [float(part.strip()) for part in value.split(",")]
        if len(numbers) == 1:
            numbers *= count
        if len(numbers) != count or any(
            not math.isfinite(x) or not 0 < x <= 2 for x in numbers
        ):
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"{key} requires one or {count} finite multipliers, each 0 < value <= 2"
        ) from exc
    return numbers


def profile(name, environ=None):
    if name not in ("baseline", "double", "env"):
        raise ValueError("Simulation tracking profile must be baseline, double or env")
    scale = 2.0 if name == "double" else 1.0
    limits = dict(
        name=name,
        scale=scale,
        simulation_only=True,
        **{field: getattr(PhysicalPlant, field) * scale for field in FIELDS},
        motor_reference_leash_rad=math.radians(10) * scale,
    )
    if name == "env":
        values = os.environ if environ is None else environ
        limits["scale"] = None
        limits["multipliers"] = {}
        global_multiplier = multipliers(values.get(GLOBAL_ENV, "1"), 1, GLOBAL_ENV)[0]
        limits["global_multiplier"] = global_multiplier
        limits["combined_multipliers"] = {}
        for key, (field, count) in ENV_FIELDS.items():
            factors = multipliers(values.get(key, "1"), count, key)
            combined = [global_multiplier * factor for factor in factors]
            if any(factor > 2 for factor in combined):
                raise ValueError(
                    f"{GLOBAL_ENV} × {key} must not exceed 2 for any joint"
                )
            base = (
                math.radians(10)
                if field == "motor_reference_leash_rad"
                else getattr(PhysicalPlant, field)
            )
            effective = [base * factor for factor in combined]
            limits[field] = effective[0] if count == 1 else effective
            limits["multipliers"][key] = factors
            limits["combined_multipliers"][key] = combined
        limits["joint_order"] = dict(
            left_arm=list(ARM_ORDER),
            both_arms=[
                f"{side}_{joint}" for side in ("left", "right") for joint in ARM_ORDER
            ],
        )
    return limits


def plant_class(name, environ=None):
    limits = profile(name, environ)

    class SimulationTrackingPlant(PhysicalPlant):
        def __init__(self, model_plant, adapter, hand, read, config, event, **kwargs):
            # An expanded profile cannot wrap a SDK/DDS writer. Fail before
            # constructing the plant or allowing any command.
            from sim_runtime import SimulationIO

            if (
                not isinstance(adapter, SimulationIO)
                or hand is not adapter
                or kwargs.get("clock") is not adapter
            ):
                raise ValueError(
                    "Tracking experiments require SimulationIO for arm, hand and clock"
                )
            super().__init__(model_plant, adapter, hand, read, config, event, **kwargs)

    for field in FIELDS:
        setattr(SimulationTrackingPlant, field, limits[field])
    return SimulationTrackingPlant
