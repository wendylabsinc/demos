"""Bounded IK/planner tuning; defaults preserve the deployed policy."""

import os
import json
from pathlib import Path
import numpy as np


def configure(plant, config):
    """Validate planner settings and save effective values for deterministic replay."""
    profile = config.get(
        "planner_profile", os.environ.get("HANDOFF_PLANNER_PROFILE", "legacy")
    )
    if profile not in ("legacy", "tuned"):
        raise ValueError("Planner profile must be legacy or tuned")
    if profile == "tuned":
        defaults = json.loads(
            (
                Path(__file__).parent / "g1_can_handoff/noise_aware_planner.json"
            ).read_text()
        )
        config.update({**defaults, **config})
    config["planner_profile"] = profile
    backend = config.get("planning_backend", os.environ.get("HANDOFF_PLANNING_BACKEND", "cpu"))
    if backend not in ("cpu", "cuda"):
        raise ValueError("Planning backend must be cpu or cuda")
    plant.planning_backend = backend
    config["planning_backend"] = backend
    weight = float(
        config.get(
            "ik_reference_weight_m_per_rad",
            os.environ.get("HANDOFF_IK_REFERENCE_WEIGHT_M_PER_RAD", "0"),
        )
    )
    tolerance = float(
        config.get(
            "ik_position_tolerance_m",
            float(os.environ.get("HANDOFF_IK_POSITION_TOLERANCE_MM", "12")) / 1000,
        )
    )
    scaling = config.get(
        "ik_variable_scale", os.environ.get("HANDOFF_IK_VARIABLE_SCALE", "unit")
    )
    orientation_step = float(
        config.get(
            "grasp_orientation_step_rad",
            os.environ.get("HANDOFF_GRASP_ORIENTATION_STEP_RAD", "0"),
        )
    )
    if scaling not in ("unit", "jac"):
        raise ValueError("IK variable scale must be unit or jac")
    if not np.isfinite(orientation_step) or not 0 <= orientation_step <= 0.3:
        raise ValueError("Grasp orientation step must be between 0 and 0.3 rad")
    budget = config.get(
        "grasp_candidate_budget",
        int(os.environ.get("HANDOFF_GRASP_CANDIDATE_BUDGET", "1")),
    )
    covariance = np.asarray(
        config.get("planning_position_cov_m2", np.diag([1e-6, 1e-6, 0]).tolist()),
        dtype=float,
    )
    if not np.isfinite(weight) or not 0 <= weight <= 0.005:
        raise ValueError("IK reference weight must be between 0 and 0.005 m/rad")
    if not np.isfinite(tolerance) or not 0.001 <= tolerance <= 0.012:
        raise ValueError("IK position tolerance must be between 1 and 12 mm")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 3:
        raise ValueError("Grasp candidate budget must be an integer between 1 and 3")
    if (
        covariance.shape != (3, 3)
        or not np.isfinite(covariance).all()
        or not np.allclose(covariance, covariance.T)
        or np.linalg.eigvalsh(covariance).min() < -1e-12
    ):
        raise ValueError(
            "Planning position covariance must be finite symmetric positive semidefinite"
        )
    plant.ik.ik_reference_weight = weight
    plant.ik.ik_position_tolerance = tolerance
    plant.ik.IK_SETTINGS = dict(
        plant.ik.IK_SETTINGS, x_scale=1.0 if scaling == "unit" else "jac"
    )
    plant.grasp_orientation_step_rad = orientation_step
    plant.grasp_candidate_budget = 1 if backend == "cuda" else budget
    plant.planning_position_covariance = covariance
    # Store effective values so planning snapshots replay settings rather than
    # depending on the future operator environment.
    config.update(
        ik_reference_weight_m_per_rad=weight,
        ik_position_tolerance_m=tolerance,
        ik_variable_scale=scaling,
        grasp_orientation_step_rad=orientation_step,
        grasp_candidate_budget=plant.grasp_candidate_budget,
        planning_position_cov_m2=covariance.tolist(),
    )


def grasp_noise_score(plant):
    """Estimate worst-case joint RMS noise from the fitted position covariance."""
    import mujoco

    model = plant.ik.model
    data = mujoco.MjData(model)
    data.qpos[:] = plant.ik.data.qpos
    scores = []
    for q in plant.candidate_joint_goals:
        data.qpos[plant.ik.qadr[plant.ik.arm]] = q
        mujoco.mj_forward(model, data)
        linear = np.zeros((3, model.nv))
        angular = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, linear, angular, plant.ik.site)
        dofs = model.jnt_dofadr[plant.ik.joints[plant.ik.arm]]
        jacobian = np.vstack((linear[:, dofs], np.sqrt(2) * 0.15 * angular[:, dofs]))
        inverse = np.linalg.pinv(jacobian, rcond=1e-5)[:, :3]
        covariance = inverse @ plant.planning_position_covariance @ inverse.T
        scores.append(float(np.sqrt(max(0, np.trace(covariance)))))
    return max(scores)
