"""Real MuJoCo perspective RGB/depth cameras and geometry ray measurements.

Create, capture and close each SensorRig on one sensory thread. Pass a copied
MjData plus scene_state(model) copied under the physics lock. Rendering then
uses its own model and never blocks or changes the flight's physical state.
Linux deployments choose EGL or OSMesa with MUJOCO_GL before importing MuJoCo;
macOS uses MuJoCo's offscreen CGL backend.
"""

from __future__ import annotations

import copy
import math
import threading

import mujoco
import numpy as np


def scene_state(model: mujoco.MjModel) -> dict:
    return {name: getattr(model, name).copy() for name in ("geom_pos", "geom_size", "geom_rgba")}


def ranges(model: mujoco.MjModel, data: mujoco.MjData, *, azimuths: int = 24,
           max_distance: float = 25.0) -> list[dict]:
    """Return unlabeled range rays from physical geometry, including occlusion.

    Horizontal and downward-tilted beams measure obstacles beneath cruise
    altitude. Non-colliding smoke/fire/oil display cells are excluded; obstacles,
    the floor, fence and physical worker remain measurable. No object names or
    scenario flags are exposed to the navigation stack.
    """
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    rotation = data.xmat[body].reshape(3, 3)
    origin = data.xpos[body] + rotation @ np.array([0., 0., -.03])
    geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    measurements = []
    for elevation in (0., -20., -45., -70.):
        elevation = math.radians(elevation)
        for azimuth in np.linspace(0, 2 * math.pi, azimuths, endpoint=False):
            direction = rotation @ np.array([math.cos(elevation) * math.cos(azimuth),
                                             math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            distance = mujoco.mj_ray(model, data, origin, direction, geom_group, 1, body, geom_id)
            measurements.append({"origin": origin.tolist(), "direction": direction.tolist(),
                "distance": float(distance) if 0 <= distance <= max_distance else None,
                "maxDistance": max_distance})
    direction = rotation @ np.array([0., 0., -1.])
    distance = mujoco.mj_ray(model, data, origin, direction, geom_group, 1, body, geom_id)
    measurements.append({"origin": origin.tolist(), "direction": direction.tolist(),
        "distance": float(distance) if 0 <= distance <= max_distance else None, "maxDistance": max_distance})
    return measurements


class SensorRig:
    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240, *,
                 tracking_resolution: tuple[int, int] | None = (640, 480)):
        self._owner = threading.get_ident()
        self._source_model = model
        self.model = copy.copy(model)
        self.width, self.height = int(width), int(height)
        self._renderer = mujoco.Renderer(self.model, width=self.width, height=self.height,
                                         max_geom=max(10000, self.model.ngeom + 200))
        self._tracking_size = tuple(map(int, tracking_resolution or (self.width, self.height)))
        # Small distant worker badges need actual extra camera samples. Keep
        # survey imagery at its original resolution for mission comparison.
        # The second context is created lazily on this same sensory thread.
        self._tracking_renderer = self._renderer if self._tracking_size == (self.width, self.height) else None
        self._option = mujoco.MjvOption()
        self._option.geomgroup[:] = 1
        # Camera data measures visible surfaces. Disable contact/joint debug
        # overlays; these are not scene objects a real camera would see.
        self._option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        self._option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        self.closed = False

    def _check_thread(self) -> None:
        if threading.get_ident() != self._owner:
            raise RuntimeError("SensorRig rendering must stay on its creating thread")
        if self.closed:
            raise RuntimeError("SensorRig is closed")

    def capture(self, data: mujoco.MjData, scene_state: dict | None = None,
                camera: str = "survey") -> dict:
        self._check_thread()
        fields = scene_state if scene_state is not None else globals()["scene_state"](self._source_model)
        for name, values in fields.items():
            if name in ("geom_pos", "geom_size", "geom_rgba"):
                getattr(self.model, name)[:] = values
        # MjData is a sensor-thread-owned snapshot. Forward kinematics places
        # the perspective camera at the copied airframe's actual pose.
        mujoco.mj_forward(self.model, data)
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            raise ValueError(f"Camera {camera!r} is unavailable")
        if camera == "tracking":
            width, height = self._tracking_size
            if self._tracking_renderer is None:
                self._tracking_renderer = mujoco.Renderer(self.model, width=width, height=height,
                    max_geom=max(10000, self.model.ngeom + 200))
            renderer = self._tracking_renderer
        else:
            renderer = self._renderer
            width, height = self.width, self.height
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=camera_id, scene_option=self._option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera_id, scene_option=self._option)
        depth = renderer.render().astype(np.float32, copy=True)
        renderer.disable_depth_rendering()
        fovy = float(self.model.cam_fovy[camera_id])
        focal = height / (2 * math.tan(math.radians(fovy) / 2))
        position = data.cam_xpos[camera_id].copy()
        rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
        return {"image": rgb, "depth": depth, "timestamp": float(data.time), "camera": camera,
            "cameraPosition": position.tolist(), "cameraRotation": rotation.tolist(),
            "cameraPose": {"position": position.tolist(), "rotation": rotation.tolist()},
            "intrinsics": {"fx": focal, "fy": focal, "cx": (width - 1) / 2,
                           "cy": (height - 1) / 2, "width": width, "height": height, "fovy": fovy},
            "model": "MuJoCo perspective RGB and metric depth at actual airframe pose"}

    def ranges(self, data: mujoco.MjData) -> list[dict]:
        self._check_thread()
        return ranges(self.model, data)

    def close(self) -> None:
        self._check_thread()
        if self._tracking_renderer is not None and self._tracking_renderer is not self._renderer:
            self._tracking_renderer.close()
        self._renderer.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
