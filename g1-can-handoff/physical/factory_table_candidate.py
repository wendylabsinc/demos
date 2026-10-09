"""Read-only factory-mount/table candidate and shared planner check.

No motor adapters are constructed. Candidate provenance is model-based, never
independently measured calibration. Table bounds cover the whole arm workspace.
"""

import json
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from camera_input import Camera, geometry
from g1_can_handoff.sim import scene
from physical_core import BODY_NAMES, PhysicalPlant, calibrated_scene
from pickup import prepare


def status():
    return json.load(urllib.request.urlopen("http://127.0.0.1:8102/status", timeout=5))


def candidate(frame, measured):
    p = scene(height=0.762)
    for name, q in zip(BODY_NAMES, measured["body"]["q"]):
        p.data.qpos[p.model.jnt_qposadr[p.model.joint(name).id]] = q
    mujoco.mj_forward(p.model, p.data)
    pelvis, torso = [p.model.body(name).id for name in ("pelvis", "torso_link")]
    rp, rt = [p.data.xmat[i].reshape(3, 3) for i in (pelvis, torso)]
    mount = Rotation.from_euler("y", 0.8307767239493009).as_matrix()
    optical_axes = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    camera_rotation = rp.T @ rt @ mount @ optical_axes
    camera_position = rp.T @ (
        p.data.xpos[torso]
        - p.data.xpos[pelvis]
        + rt @ np.array([0.0576235, 0.01753, 0.41987])
    )
    # World yaw zero follows the robot's forward direction, not magnetic north.
    roll, pitch = measured["body"]["imu_rpy"][:2]
    base_rotation = Rotation.from_euler("xyz", [roll, pitch, 0]).as_matrix()
    r, g, b = np.moveaxis(frame["color_bgr"][..., ::-1].astype(float), -1, 0)
    depth = frame["depth_m"]
    points = frame["rays"] * depth[..., None]
    yellow = binary_erosion(
        (r > 100)
        & (g > 100)
        & (b < 0.65 * np.minimum(r, g))
        & np.isfinite(depth)
        & (depth > 0.1)
        & (depth < 2),
        iterations=2,
    )
    cloud = points[yellow][::3]
    if len(cloud) < 100:
        raise ValueError("Too few yellow tape depth points")
    from table_plane import fit_table_plane

    up, center, fitted, plane_quality = fit_table_plane(
        cloud, points[::4, ::4].reshape(-1, 3), points[2::4, 2::4].reshape(-1, 3)
    )
    predicted = (base_rotation @ camera_rotation).T @ np.array([0.0, 0.0, 1.0])
    axis = np.cross(up, predicted)
    angle = np.arctan2(np.linalg.norm(axis), up @ predicted)
    if angle > np.deg2rad(12):
        raise ValueError(
            "Table tilt differs by more than 12 degrees from factory estimate"
        )
    correction = (
        np.eye(3)
        if np.linalg.norm(axis) < 1e-9
        else Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
    )
    camera_rotation = base_rotation @ camera_rotation @ correction
    camera_world = np.eye(4)
    camera_world[:3, :3] = camera_rotation
    camera_world[:3, 3] = base_rotation @ camera_position
    camera_world[2, 3] = 0.762 - float(up @ center)
    base_world = np.eye(4)
    base_world[:3, :3] = base_rotation
    base_world[2, 3] = camera_world[2, 3] - (base_rotation @ camera_position)[2]
    table_points = fitted @ camera_rotation.T + camera_world[:3, 3]
    # Extend conservatively toward the robot by 20 mm from the nearest tape.
    # A 10 m wide/deep box covers every reachable arm pose, including staging.
    near_x = float(np.quantile(table_points[:, 0], 0.01)) - 0.02
    arm_limits = p.model.jnt_range[[p.model.joint(n).id for n in BODY_NAMES[15:]]]
    return dict(
        commissioned=False,
        calibration_method="factory_mount_and_table_plane_candidate",
        camera_serial=frame["meta"]["camera_serial"],
        device="unitree-g1-nx-2.local",
        camera_to_world=camera_world.tolist(),
        pelvis_to_world=base_world.tolist(),
        table_height_m=0.762,
        table_center_xy_m=[near_x + 5, 0],
        table_half_size_xy_m=[5, 5],
        table_near_edge_conservative_x_m=near_x,
        table_extent_source="conservative extension beyond reachable workspace",
        can_radius_m=0.033,
        can_height_m=0.122,
        arm_limits_rad=arm_limits.tolist(),
        source_frame_id=frame["meta"]["frame_id"],
        tilt_correction_deg=float(np.degrees(angle)),
        table_plane_rmse_m=plane_quality["rmse_m"],
        table_plane_quality=plane_quality,
        independent_validation=False,
        assumptions=[
            "operator confirms factory mount position and stationary base",
            "camera tilt adjusted; pivot translation not modeled",
            "horizontal 0.762 m table; nearest yellow tape marks near edge",
            "factory d435 frame approximates aligned RGB optical origin",
            "model arm limits; body IMU describes pelvis orientation",
        ],
    )


def main():
    before = status()
    camera = Camera()
    camera.acquire()
    try:
        frame = camera.frame()
        config = candidate(frame, status()["measured"])
        events = []
        result = dict(config=config, passive_plan_passed=False)
        p = calibrated_scene(config)

        def event(phase, **fields):
            events.append(dict(phase=phase, **fields))

        def read():
            return status()["measured"]

        plant = PhysicalPlant(p, None, None, read, config, event)

        def observe(phase, require_table, expected_center_xy=None):
            observation, vision = geometry(
                camera.frame(), config, require_table, expected_center_xy
            )
            result["vision"] = vision
            result["can_center_world_m"] = observation.center.tolist()
            return observation, vision

        try:
            prepare(plant, observe, event)
            result["passive_plan_passed"] = True
        except Exception as exc:
            result["planning_error"] = str(exc)
        result["events"] = events
        result["new_motion_commands"] = (
            status()["motion_commands_sent"] - before["motion_commands_sent"]
        )
        Path("/data/factory-table-passive-plan.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))
    finally:
        camera.close()


if __name__ == "__main__":
    main()
