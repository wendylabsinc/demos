"""Read-only mount hypothesis; never writes commissioning calibration."""

import json, io, urllib.request
import numpy as np
from scipy.optimize import least_squares
from camera_input import Camera, target_coke_mask
from g1_can_handoff.sim import scene
from physical_core import BODY_NAMES
from scipy.ndimage import binary_erosion, label


def diagnose():
    status = json.load(urllib.request.urlopen("http://127.0.0.1:8102/status"))
    p = scene(height=0.732)
    for n, q in zip(BODY_NAMES, status["measured"]["body"]["q"]):
        p.data.qpos[p.model.jnt_qposadr[p.model.joint(n).id]] = q
    import mujoco

    mujoco.mj_forward(p.model, p.data)
    camera = Camera()
    camera.acquire()
    try:
        frame = camera.frame()
    finally:
        camera.close()
    depth = frame["depth_m"]
    points = frame["rays"] * depth[..., None]
    valid = (depth > 0.1) & (depth < 2)
    target_mask, detection = target_coke_mask(
        frame, {"can_radius_m": 0.033}, points, require_table=False
    )
    target = points[target_mask]
    # Fit only desk points below target in image. Report conditioning and inliers.
    rows, cols = np.indices(depth.shape)
    uv = np.argwhere(target_mask)
    roi = (
        valid & (rows > uv[:, 0].max() + 15) & (rows < 450) & (cols > 70) & (cols < 570)
    )
    cloud = points[roi][::5]
    best = None
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, c, d = cloud[rng.choice(len(cloud), 3, replace=False)]
        n = np.cross(c - a, d - a)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n /= norm
        if n[2] > 0:
            n = -n
        inliers = np.abs((cloud - a) @ n) < 0.006
        if best is None or inliers.sum() > best[0].sum():
            best = (inliers, n)
    fitted = cloud[best[0]]
    center = fitted.mean(0)
    _, _, vt = np.linalg.svd(fitted - center, full_matrices=False)
    up = vt[-1]
    if up[2] > 0:
        up = -up
    right = np.array([1.0, 0, 0])
    right -= up * (right @ up)
    right /= np.linalg.norm(right)
    y = -right
    x = np.cross(y, up)
    rotation = np.stack([x, y, up])
    translation = p.data.cam_xpos[p.model.camera("head").id].copy()
    world = target @ rotation.T + translation
    fit = least_squares(
        lambda c: np.linalg.norm(world[:, :2] - c, axis=1) - 0.033,
        world[:, :2].mean(0),
        loss="soft_l1",
        f_scale=0.002,
    )
    table = float(up @ center + translation[2])
    out = {
        "estimated_only": True,
        "detection": detection,
        "camera_mount": "operator confirms rigid head mount",
        "assumptions": [
            "model head camera origin",
            "camera image right follows robot right",
            "horizontal table",
        ],
        "camera_to_world_hypothesis": np.block(
            [[rotation, translation[:, None]], [np.array([[0, 0, 0, 1]])]]
        ).tolist(),
        "table_height_hypothesis_m": table,
        "target_xy_hypothesis_m": fit.x.tolist(),
        "target_cylinder_rmse_m": float(np.sqrt(np.mean(fit.fun**2))),
        "table_plane_rmse_m": float(np.sqrt(np.mean(((fitted - center) @ up) ** 2))),
        "table_inlier_fraction": float(best[0].mean()),
        "up_camera": up.tolist(),
        "left_workspace_match": bool(
            0.27 <= fit.x[0] <= 0.45 and 0.09 <= fit.x[1] <= 0.29
        ),
        "frame_id": frame["meta"]["frame_id"],
        "motion_commands_sent": status["motion_commands_sent"],
        "physical_pickup_verified": False,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    diagnose()
