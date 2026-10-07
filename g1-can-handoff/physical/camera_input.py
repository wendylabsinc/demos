"""Exact-frame camera client, calibrated ray geometry, and real visual lift check."""
import io
import json
import time
import threading
import urllib.request
import zipfile
from PIL import Image
import numpy as np
from scipy.ndimage import binary_erosion,binary_closing,label
from scipy.optimize import least_squares
from g1_can_handoff.perception import Observation


class Camera:
    def __init__(self):
        self.base='http://127.0.0.1:8001';self.token=None;self.rays=None;self.intrinsics=None;self.serial=None
        self.lease_stop=threading.Event();self.lease_thread=None;self.lease_error=None
    def request(self,path,data=None):
        headers={'Content-Type':'application/json'}
        if self.token: headers['X-Camera-Lease']=self.token
        r=urllib.request.Request(self.base+path,data=None if data is None else json.dumps(data).encode(),headers=headers)
        return urllib.request.urlopen(r,timeout=5).read()
    def acquire(self):
        if self.token:raise RuntimeError('Camera lease already acquired')
        self.token=json.loads(self.request('/lease/acquire',{'owner':'DEV-g1-handoff-physical'}))['token']
        self.lease_error=None;self.lease_stop.clear()
        self.lease_thread=threading.Thread(target=self._renew,daemon=True)
        self.lease_thread.start()
    def _renew(self):
        # Staging search and measured arm motion can take longer than the
        # camera service lease. Keep the existing lease, never reacquire it
        # silently if another owner takes control.
        while not self.lease_stop.wait(.5):
            try:
                self.request('/lease/renew',{})
                self.lease_error=None
            except Exception as exc:
                self.lease_error=str(exc)
    def close(self):
        self.lease_stop.set()
        if self.lease_thread:
            self.lease_thread.join(timeout=6)
            self.lease_thread=None
        if self.token:
            try:self.request('/lease/release',{})
            finally:self.token=None
    def frame(self):
        self.request('/lease/renew',{})
        health=json.loads(self.request('/health'))
        if self.rays is None or health['serial'] != self.serial:
            # Rays are calibration data: cache once, never use this slow full
            # bundle's old image as a new observation.
            raw=self.request('/frame.rgbd?frame_id='+str(health['frame_id']))
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:
                self.rays=f['rays'].copy();meta=json.loads(str(f['metadata_json']))
            self.intrinsics=meta['intrinsics'];self.serial=meta['camera_serial']
            health=json.loads(self.request('/health'))
        fid=health['frame_id']
        timing=self.request('/frame.policy-rgbd?frame_id='+str(fid))
        with np.load(io.BytesIO(timing),allow_pickle=False) as f:
            meta=json.loads(str(f['metadata_json']))
        compact=self.request('/frame.recorder-rgbd?frame_id='+str(fid))
        with zipfile.ZipFile(io.BytesIO(compact)) as z:
            cm=json.loads(z.read('metadata.json'))
            depth=np.asarray(Image.open(io.BytesIO(z.read('depth.png'))),dtype=np.float32)*cm['calibration']['depth_scale_m']
            color=np.asarray(Image.open(io.BytesIO(z.read('color.jpg'))).convert('RGB'))[...,::-1].copy()
        if cm['source_frame_id'] != fid or meta['frame_id'] != fid or cm['captured_at_unix_ns'] != meta['captured_at_unix_ns']:
            raise RuntimeError('Camera color/depth exact-frame identity mismatch')
        if meta['intrinsics'] != self.intrinsics or meta['camera_serial'] != self.serial:
            self.rays=None;raise RuntimeError('Camera calibration changed; refresh ray cache')
        if depth.shape!=color.shape[:2] or self.rays.shape!=(*depth.shape,3):
            raise RuntimeError('Camera color/depth/ray geometry dimensions disagree')
        if cm['calibration']['intrinsics']!=self.intrinsics:
            raise RuntimeError('Camera compact-frame intrinsics disagree with cached rays')
        timestamp=meta['depth_time']
        if not meta.get('timestamp_valid') or not 0<=time.monotonic()-timestamp<=.3:
            raise RuntimeError('Camera exposure timestamp invalid/stale')
        if abs(meta['color_time']-timestamp)>.012:raise RuntimeError('Camera RGB/depth exposure skew exceeds 12 ms')
        frame={'color_bgr':color,'depth_m':depth,'rays':self.rays,'meta':meta,'timestamp':timestamp,
               'color_codec':'jpeg_exact_source_frame','source_rgbd_zip':compact,'source_timing_npz':timing}
        bundle=io.BytesIO()
        np.savez(bundle,color_bgr=color,depth_m=depth,rays=self.rays,metadata_json=json.dumps({**meta,'color_codec':frame['color_codec']}))
        frame['raw']=bundle.getvalue()
        return frame



def target_red_mask(rgb,depth,roi=None):
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    valid=np.isfinite(depth)&(depth>.1)&(depth<2.5)
    source_red=valid&(r>75)&(r>g*1.7)&(r>b*1.7)
    original=source_red.copy()
    if roi is not None:original &= roi
    # Printed white stripes may divide one label. Closing groups adjacent
    # label islands; retain only originally red measured pixels for fitting.
    grouped=binary_closing(original,structure=np.ones((11,11),dtype=bool))
    cc,count=label(grouped);sizes=np.bincount(cc.ravel());sizes[0]=0
    candidates=np.flatnonzero(sizes>100)
    if not len(candidates) or (roi is None and len(candidates)!=1):
        raise RuntimeError('Target red component absent/ambiguous')
    component = np.isin(cc,candidates)
    # Closing erodes the image boundary. Check the un-eroded source label
    # near the boundary too, so a partially off-screen can cannot look whole.
    if (source_red[:6].any() or source_red[-6:].any() or
            source_red[:,:6].any() or source_red[:,-6:].any()):
        raise RuntimeError('Red target is clipped by the camera image boundary')
    selected=binary_erosion(original,iterations=2)&component
    if selected.sum()<40:raise RuntimeError('Target has too few valid red depth points')
    return selected

def geometry(frame,config,require_table=True,expected_center_xy=None):
    if frame['meta']['camera_serial']!=config['camera_serial']:raise RuntimeError('Camera serial changed')
    rgb=frame['color_bgr'][...,::-1].astype(float);depth=frame['depth_m']
    # Exact distortion-corrected RealSense rays, not simulated pinhole geometry.
    pts=frame['rays']*depth[...,None]
    t=np.asarray(config['camera_to_world']);world=pts@t[:3,:3].T+t[:3,3]
    r,g,b=np.moveaxis(rgb,-1,0)
    valid=np.isfinite(depth)&(depth>.1)&(depth<2.5)
    roi=None
    if expected_center_xy is not None:
        if require_table:raise ValueError('Tracked component merging is only allowed after lifting')
        expected=np.asarray(expected_center_xy,dtype=float)
        if expected.shape!=(2,) or not np.isfinite(expected).all():
            raise ValueError('Expected finite tracked target XY position')
        # Candidate pixels include the existing 40 mm center tolerance plus
        # the cylinder radius and 10 mm depth/pixel margin. Acceptance still
        # requires the fitted cylinder center within 40 mm, below.
        roi=np.linalg.norm(world[:,:,:2]-expected,axis=2)<=config['can_radius_m']+.04+.01
    red=target_red_mask(rgb,depth,roi=roi)
    points=world[red]
    if len(points)<40:raise RuntimeError('Target has too few valid depth points')
    points=points[::max(1,len(points)//1500)]
    fit=least_squares(lambda c:np.linalg.norm(points[:,:2]-c,axis=1)-config['can_radius_m'],points[:,:2].mean(axis=0),loss='soft_l1',f_scale=.002)
    residual=float(np.sqrt(np.mean(fit.fun**2)))
    if residual>.006:raise RuntimeError('Physical target cylinder fit exceeds 6 mm')
    if expected_center_xy is not None and np.linalg.norm(fit.x-expected)>.04:
        raise RuntimeError('Tracked target center differs by over 40 mm')
    table=config['table_height_m']
    if require_table:
        near=valid&~red&(np.linalg.norm(world[:,:,:2]-fit.x,axis=2)<.22)
        z=world[near,2];plane=z[np.abs(z-table)<.008]
        if len(plane)<100:raise RuntimeError('Independent measured table height disagrees with camera')
        if abs(np.median(plane)-table)>.005:raise RuntimeError('Table plane differs by over 5 mm')
        if np.quantile(points[:,2],.05)>table+.06 or np.quantile(points[:,2],.95)<table+.07:
            raise RuntimeError('Can is not upright on the measured table')
    obs=Observation(np.r_[fit.x,table+config['can_height_m']/2],table,frame['timestamp'],residual,len(points))
    return obs,{'top_quantile_z_m':float(np.quantile(points[:,2],.95)), 'center_xy_m':fit.x.tolist(),
                'fit_error_m':residual,'pixels':len(points),'frame_id':frame['meta']['frame_id']}
