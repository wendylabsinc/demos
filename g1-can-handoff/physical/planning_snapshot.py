"""Exact planner inputs and outputs, without constructing hardware I/O.

The MJB contains meshes, inertias, collision masks and solver options. Mutable
model poses and complete integration state are saved at each decision. Measured
feedback is kept separately: FK intentionally ignores measured velocities.
"""

import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from functools import wraps
from pathlib import Path

import mujoco
import numpy as np

SOURCE_NAMES = (
    "physical_core.py",
    "pickup.py",
    "staging.py",
    "planning_snapshot.py",
    "planner_tuning.py",
    "g1_can_handoff/noise_aware_planner.json",
    "g1_can_handoff/sim.py",
    "g1_can_handoff/task.py",
    "pose_alignment.py",
    "adapter/arm_reference.py",
    "adapter/g1_arm.py",
    "adapter/arm_gains.py",
)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def recorded(kind):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            recorder = getattr(self, "planning_recorder", None)
            if recorder is None:
                return fn(self, *args, **kwargs)
            return recorder.call(
                self,
                kind,
                dict(args=plain(args), kwargs=plain(kwargs)),
                lambda: fn(self, *args, **kwargs),
            )

        return wrapped

    return decorate


class PlanningRecorder:
    def __init__(self, directory, event):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.event = event
        self.models = {}
        self.sequence = 0
        root = Path(__file__).parent
        self.provenance = dict(
            python=platform.python_version(),
            machine=platform.machine(),
            dependencies={
                n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "scipy")
            },
            source_sha256={
                n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                for n in SOURCE_NAMES
            },
        )

    def call(self, plant, kind, request, fn):
        model, data = plant.ik.model, plant.ik.data
        # Serialize each model once; subsequent changes are captured explicitly.
        if model not in self.models:
            raw = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
            mujoco.mj_saveModel(model, buffer=raw)
            digest = hashlib.sha256(raw).hexdigest()
            path = self.directory / (digest + ".mjb")
            if not path.exists():
                path.write_bytes(raw.tobytes())
            self.models[model] = dict(path=str(path), sha256=digest)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        self.sequence += 1
        name = f"{time.time_ns()}-{self.sequence:04d}-{kind}.json"
        snapshot = dict(
            schema="wendy.g1.planning-snapshot.v1",
            kind=kind,
            request=request,
            captured_monotonic_s=plant.clock.monotonic(),
            model=self.models[model],
            provenance=self.provenance,
            integration_state_spec=spec,
            integration_state=state.tolist(),
            qpos=data.qpos.tolist(),
            qvel=data.qvel.tolist(),
            model_body_pos=model.body_pos.tolist(),
            model_body_quat=model.body_quat.tolist(),
            config=copy.deepcopy(plant.config),
            measured=copy.deepcopy(plant.last_synced_snapshot),
            collision_phase=plant.ik.contact_phase,
            table_clearance_m=getattr(plant.ik, "table_clearance_m", 0.0),
            held_right_arm=plant.right_arm.tolist(),
            held_right_hand=plant.right_hand.tolist(),
            held_left_hand=plant.left_hand.tolist(),
            solver_settings=dict(
                ik=copy.deepcopy(plant.ik.IK_SETTINGS),
                ik_seed_q_rad=[
                    np.clip(
                        seed,
                        model.jnt_range[plant.ik.joints[plant.ik.arm], 0] + 0.025,
                        model.jnt_range[plant.ik.joints[plant.ik.arm], 1] - 0.025,
                    ).tolist()
                    for seed in (
                        data.qpos[plant.ik.qadr[plant.ik.arm]],
                        [-1.2, 0.2, 0, 1.2, 0, 0, 0],
                        [-0.5, 0.2, 1.3, 1.4, 0.7, -0.9, -0.5],
                    )
                ],
                ik_bound_margin_rad=0.025,
                ik_orientation_weight=0.15,
                continuous_ik=dict(
                    max_nfev=250,
                    local_bound_rad=0.2,
                    orientation_weight=0.15,
                    torso_gap_m=0.003,
                    torso_residual_weight=5.0,
                    nullspace_weight=0.003,
                ),
                staging=dict(
                    seed=7, max_nodes=1800, timeout_s=10.0, edge_resolution_rad=0.004
                ),
                interpolation=dict(
                    period_s=0.02, max_velocity_rad_s=0.25, max_acceleration_rad_s2=0.5
                ),
            ),
            limitations=[
                "Staging/grasp wall-clock deadlines can differ across CPUs.",
                "Measured base translation and IMU angular velocity are unavailable in this feedback source.",
            ],
        )
        path = self.directory / name
        # Persist before planning so failed decisions retain their exact inputs.
        path.write_text(json.dumps(plain(snapshot), separators=(",", ":")) + "\n")
        self.event(
            "planning_snapshot", kind=kind, path=str(path), model=self.models[model]
        )
        try:
            result = fn()
            snapshot["result"] = plain(result)
            if kind == "preflight":
                snapshot["result"] = dict(
                    staging_start=plain(plant.staging_start),
                    staging_waypoints=plain(plant.staging_waypoints),
                    path_preflight_ok=plant.path_preflight_ok,
                )
            return result
        except Exception as exc:
            snapshot["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            path.write_text(json.dumps(plain(snapshot), separators=(",", ":")) + "\n")
            self.event(
                "planning_snapshot_complete",
                kind=kind,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                error=snapshot.get("error"),
            )


def replay(path):
    """Replay one numerical decision with its compiled model and frozen clock."""
    from types import SimpleNamespace
    from physical_core import PhysicalPlant
    from g1_can_handoff.sim import Plant

    path = Path(path)
    snapshot = json.loads(path.read_text())
    model_path = path.parent / Path(snapshot["model"]["path"]).name
    raw = model_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot["model"]["sha256"]:
        raise ValueError("Planning model checksum mismatch")
    current = {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "scipy")}
    if current != snapshot["provenance"]["dependencies"]:
        raise ValueError("Planner dependency versions differ")
    root = Path(__file__).parent
    if any(
        hashlib.sha256((root / n).read_bytes()).hexdigest() != v
        for n, v in snapshot["provenance"]["source_sha256"].items()
    ):
        raise ValueError("Planner source differs from captured decision")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    ik = Plant(model)
    measured = copy.deepcopy(snapshot["measured"])
    plant = PhysicalPlant(
        ik,
        None,
        None,
        lambda: copy.deepcopy(measured),
        snapshot["config"],
        lambda *a, **k: None,
        clock=SimpleNamespace(
            monotonic=lambda: max(v["time"] for v in measured.values())
        ),
    )
    model.body_pos[:] = snapshot["model_body_pos"]
    model.body_quat[:] = snapshot["model_body_quat"]
    mujoco.mj_setState(
        model,
        ik.data,
        np.asarray(snapshot["integration_state"]),
        snapshot["integration_state_spec"],
    )
    mujoco.mj_forward(model, ik.data)
    ik.table_clearance_m = snapshot["table_clearance_m"]
    ik.set_contact_phase(snapshot["collision_phase"])
    plant.right_arm = np.asarray(snapshot["held_right_arm"])
    plant.right_hand = np.asarray(snapshot["held_right_hand"])
    plant.left_hand = np.asarray(snapshot["held_left_hand"])
    kind = snapshot["kind"]
    request = snapshot["request"]
    if kind == "ik":
        fn = lambda: ik.solve(
            np.asarray(request["position"]), np.asarray(request["rotation"])
        )
    else:
        names = {
            "preflight": "_preflight",
            "move_poses": "_move_poses",
            "continuous_ik": "_continuous_lift_ik",
            "joint_poses": "_joint_poses",
        }
        fn = lambda: getattr(plant, names[kind])(
            *[np.asarray(a) if isinstance(a, list) else a for a in request["args"]],
            **request["kwargs"],
        )
    try:
        result = plain(fn())
        error = None
        if kind == "preflight":
            result = dict(
                staging_start=plain(plant.staging_start),
                staging_waypoints=plain(plant.staging_waypoints),
                path_preflight_ok=plant.path_preflight_ok,
            )
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    expected = snapshot.get("result")
    identical = result == expected and error == snapshot.get("error")
    delta = None
    if error is None and kind in ("ik", "continuous_ik"):
        delta = float(np.max(np.abs(np.asarray(result) - np.asarray(expected))))
    return dict(
        path=str(path),
        kind=kind,
        identical=identical,
        max_abs_error_rad=delta,
        error=error,
        captured_error=snapshot.get("error"),
        hardware_used=False,
    )
