"""RGB-D red-cylinder detection. Inputs are calibrated images, not scene poses."""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import binary_erosion, label

@dataclass(frozen=True)
class Observation:
    center: np.ndarray
    table_height: float
    timestamp: float
    fit_error: float
    pixels: int

def detect(rgb, depth, intrinsics, camera_to_world, timestamp=None):
    rgb,depth = np.asarray(rgb),np.asarray(depth)
    transform = np.asarray(camera_to_world, dtype=float)
    k = np.asarray(intrinsics, dtype=float)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or depth.shape!=rgb.shape[:2]:
        raise ValueError("Aligned RGB and depth images are required")
    if transform.shape!=(4,4) or not np.all(np.isfinite(transform)) or k.shape!=(3,3) or not np.all(np.isfinite(k)) or k[0,0]<=0 or k[1,1]<=0:
        raise ValueError("Invalid camera calibration")
    if not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(transform[:3,:3]),1,atol=1e-4):
        raise ValueError("Camera calibration must contain a proper rigid rotation")
    age = time.monotonic() - timestamp if timestamp is not None else 0
    if age < -.05 or age > .5: raise ValueError("Camera observation is stale")
    v,u = np.indices(depth.shape)
    valid = np.isfinite(depth)&(depth>.1)&(depth<2.5)
    pts_camera = np.stack([(u-k[0,2])*depth/k[0,0],(v-k[1,2])*depth/k[1,1],depth],axis=-1)
    world = pts_camera @ transform[:3,:3].T + transform[:3,3]
    r,g,b = np.moveaxis(rgb.astype(float),-1,0)
    red = binary_erosion(valid&(r>75)&(r>g*1.7)&(r>b*1.7),iterations=2)
    components,count=label(red)
    sizes=np.bincount(components.ravel());sizes[0]=0
    if count>1 and np.count_nonzero(sizes>40)>1: raise ValueError("Multiple red objects: select a can before moving")
    red=components==np.argmax(sizes) if count else red
    points = world[red]
    if len(points)<40: raise ValueError("No sufficiently visible red can")
    # Fit the known 66 mm can diameter to observed side-wall points.
    # Robust loss rejects label pixels and small depth outliers.
    xy = points[:, :2]
    if len(xy)>1500: xy=xy[::max(1,len(xy)//1500)]
    fit = least_squares(lambda c: np.linalg.norm(xy-c,axis=1)-.033, np.mean(xy,axis=0), loss="soft_l1", f_scale=.002)
    residual = float(np.sqrt(np.mean(fit.fun**2)))
    if residual>.006: raise ValueError("Red object does not fit the expected can diameter")
    # Find the dominant horizontal surface near the detected can, excluding it.
    near = valid&~red&(np.linalg.norm(world[:,:,:2]-fit.x,axis=2)<.22)&(world[:,:,2]>.45)&(world[:,:,2]<1.15)
    z = world[near,2]
    if len(z)<100: raise ValueError("Table is not sufficiently visible")
    histogram,edges = np.histogram(z,bins=np.arange(.45,1.155,.005))
    peak = edges[np.argmax(histogram)]+.0025
    plane = z[np.abs(z-peak)<.006]
    if len(plane)<100: raise ValueError("No reliable horizontal table surface")
    table = float(np.median(plane))
    if np.quantile(points[:,2],.95)<table+.07 or np.quantile(points[:,2],.05)>table+.06:
        raise ValueError("Can is not upright on the detected table")
    return Observation(np.r_[fit.x,table+.061],table,time.monotonic() if timestamp is None else timestamp,residual,len(points))

class SimCamera:
    def __init__(self,plant,depth_noise=0.,seed=7):
        import mujoco
        self.plant=plant
        self.renderer=mujoco.Renderer(plant.model,height=480,width=640)
        self.depth_noise=depth_noise
        self.rng=np.random.default_rng(seed)

    def observe(self):
        import time
        p=self.plant
        self.renderer.update_scene(p.data,camera="observe")
        rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth=self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if self.depth_noise:depth+=self.rng.normal(0,self.depth_noise,depth.shape)
        cam=p.model.camera("observe").id
        focal=240/np.tan(np.deg2rad(p.model.cam_fovy[cam])/2)
        k=np.array([[focal,0,319.5],[0,focal,239.5],[0,0,1]])
        transform=np.eye(4)
        transform[:3,:3]=p.data.cam_xmat[cam].reshape(3,3)@np.diag([1,-1,-1])
        transform[:3,3]=p.data.cam_xpos[cam]
        return detect(rgb,depth,k,transform,time.monotonic()),rgb

    def close(self):
        self.renderer.close()
