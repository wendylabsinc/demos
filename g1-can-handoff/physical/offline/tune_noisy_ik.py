"""Offline sensitivity sweep on fresh draws; never creates hardware adapters."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import mujoco
from g1_can_handoff.perception import Observation
from g1_can_handoff.task import plan
from physical_core import PhysicalPlant
from planner_tuning import configure
from pickup import prepare
from sim_runtime import setup_scene, SimulationIO, apply_measured_state
from stochastic_position_noise import StochasticPositionNoise

ROOT = Path(__file__).resolve().parents[1]


def plant_for(setup, settings):
    world = setup_scene(setup)
    io = SimulationIO(world)
    apply_measured_state(world, io, setup["measured"])
    config = dict(setup["config"], **settings)
    plant = PhysicalPlant(
        setup_scene(setup), io, io, io.read, config, lambda *a, **kw: None, clock=io
    )
    return plant, io


def sweep(
    setup,
    weights,
    seeds,
    output,
    *,
    budgets=(1,),
    scales=("unit",),
    orientation_steps=(0.0,),
):
    # The same fresh draw bank is used across weights and never fitted to outputs.
    offsets = []
    for seed in seeds:
        generator = StochasticPositionNoise(seed=seed)
        generator.advance(3.0, moving=True)
        offsets.append(generator.offset("planning", 1)[0])
    observations = []
    nominal = np.array(setup["can_center_world_m"])
    profile = json.loads(
        (ROOT / "g1_can_handoff/stochastic_noise_profile.json").read_text()
    )
    covariance = np.zeros((3, 3))
    total = sum(p["observed_duration_s"] for p in profile["regimes"].values())
    for parameter in profile["regimes"].values():
        tau = parameter["drift_tau_s"]
        v = np.asarray(parameter["drift_cov_m2"])
        r = np.asarray(parameter["jitter_cov_m2"])
        transition = (
            10 / tau
            if parameter["tau_at_search_boundary"] and tau >= 120
            else 1 - np.exp(-10 / tau)
        )
        covariance[:2, :2] += (
            (2 * r + transition * v) * parameter["observed_duration_s"] / total
        )
    for weight, budget, scaling, step in [
        (w, b, s, d)
        for w in weights
        for b in budgets
        for s in scales
        for d in orientation_steps
    ]:
        settings = dict(
            ik_reference_weight_m_per_rad=weight,
            ik_variable_scale=scaling,
            grasp_orientation_step_rad=step,
            ik_position_tolerance_m=0.005,
            grasp_candidate_budget=budget,
            planning_position_cov_m2=covariance.tolist(),
        )
        plant, io = plant_for(setup, settings)
        observation = Observation(
            nominal.copy(),
            setup["config"]["table_height_m"],
            time.monotonic(),
            0.003,
            100,
        )
        plant.observe_can(observation)

        def observe(*args, **kwargs):
            current = Observation(
                nominal.copy(), observation.table_height, time.monotonic(), 0.003, 100
            )
            return current, dict(
                center_xy_m=nominal[:2].tolist(), top_quantile_z_m=nominal[2] + 0.055
            )

        _, _, points, rotation = prepare(plant, observe, lambda *a, **kw: None)
        plant.set_contact_phase("approach")
        q0 = plant.solve(points[0], rotation)
        trials = []
        for seed, offset in zip(seeds, offsets):
            target = np.array(points[0])
            target[:2] += offset
            try:
                q = plant.solve(target, rotation)
                adr = plant.ik.qadr[plant.ik.arm]
                plant.ik.ik_data.qpos[adr] = q
                mujoco.mj_forward(plant.ik.model, plant.ik.ik_data)
                trials.append(
                    dict(
                        seed=seed,
                        offset_xy_m=offset.tolist(),
                        joint_change_rad=(q - q0).tolist(),
                        position_error_m=float(
                            np.linalg.norm(
                                plant.ik.ik_data.site_xpos[plant.ik.site] - target
                            )
                        ),
                        error=None,
                    )
                )
            except ValueError as exc:
                trials.append(
                    dict(seed=seed, offset_xy_m=offset.tolist(), error=str(exc))
                )
        successful = [t for t in trials if not t["error"]]
        rms = (
            float(
                np.sqrt(
                    np.mean(np.array([t["joint_change_rad"] for t in successful]) ** 2)
                )
            )
            if successful
            else None
        )
        observations.append(
            dict(
                settings=settings,
                trials=trials,
                successful=len(successful),
                joint_change_rms_rad=rms,
                max_position_error_m=(
                    max(t["position_error_m"] for t in successful)
                    if successful
                    else None
                ),
                commands=io.commands,
                nominal_points=[p.tolist() for p in points],
                rotation=rotation.tolist(),
            )
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in observations[-1].items()
                    if k not in ("trials", "nominal_points", "rotation")
                }
            ),
            flush=True,
        )
    result = dict(
        simulation_only=True,
        hardware_used=False,
        seeds=seeds,
        sweep=observations,
        metric="RMS joint target displacement from nominal for identical noisy approach requests; all starts are the same measured held state",
    )
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jac-sweep", action="store_true")
    parser.add_argument("--orientation-sweep", action="store_true")
    parser.add_argument("--planner-budget-sweep", action="store_true")
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=24)
    args = parser.parse_args()
    sweep(
        json.loads(args.setup_json.read_text()),
        (
            [0.0]
            if args.planner_budget_sweep or args.jac_sweep or args.orientation_sweep
            else [0.0, 0.0003, 0.001, 0.003]
        ),
        list(range(args.seed_start, args.seed_start + args.seeds)),
        args.output,
        budgets=(
            (1, 2, 3)
            if args.planner_budget_sweep
            else (3,) if args.orientation_sweep else (1,)
        ),
        scales=("unit", "jac") if args.jac_sweep else ("unit",),
        orientation_steps=(0.1, 0.2, 0.3) if args.orientation_sweep else (0.0,),
    )
