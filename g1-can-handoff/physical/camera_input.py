"""Exact-frame camera client, calibrated ray geometry, and real visual lift check."""
import io
import json
import time
import threading
import urllib.request
import zipfile
from PIL import Image
import numpy as np
from scipy.ndimage import binary_erosion,maximum_filter,minimum_filter,label
from scipy.optimize import least_squares
from g1_can_handoff.perception import Observation
from yolo_client import YoloClient, fresh


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



def close_red_label(mask):
    # Rectangular max/min filters give the same binary closing with separable
    # passes, avoiding a 121-pixel footprint scan for every camera pixel.
    padded=np.pad(mask,5)
    dilated=maximum_filter(padded,size=11,mode='constant',cval=0)
    return minimum_filter(dilated,size=11,mode='constant',cval=0)[5:-5,5:-5]


def target_red_mask(rgb,depth,roi=None):
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    valid=np.isfinite(depth)&(depth>.1)&(depth<2.5)
    source_red=valid&(r>75)&(r>g*1.7)&(r>b*1.7)
    original=source_red.copy()
    if roi is not None:original &= roi
    # Printed white stripes may divide one label. Closing groups adjacent
    # label islands; retain only originally red measured pixels for fitting.
    grouped=close_red_label(original)
    cc,count=label(grouped);sizes=np.bincount(cc.ravel());sizes[0]=0
    candidates=np.flatnonzero(sizes>100)
    if not len(candidates) or (roi is None and len(candidates)!=1):
        raise RuntimeError('Target red component absent/ambiguous')
    component = np.isin(cc,candidates)
    # Only the selected target component can establish target clipping.
    # Preserve edge pixels while closing, then follow source connectivity even
    # outside a tracking ROI so ROI cropping cannot hide an offscreen label.
    if roi is None:
        target_source=component
    else:
        source_grouped=close_red_label(source_red)
        source_cc,_=label(source_grouped)
        ids=np.unique(source_cc[component & original]);ids=ids[ids!=0]
        target_source=np.isin(source_cc,ids)
    if (target_source[:6].any() or target_source[-6:].any() or
            target_source[:,:6].any() or target_source[:,-6:].any()):
        raise RuntimeError('Red target is clipped by the camera image boundary')
    selected=binary_erosion(original,iterations=2)&component
    if selected.sum()<40:raise RuntimeError('Target has too few valid red depth points')
    return selected

def target_coke_mask(frame, config, world, roi=None, detector=None, require_table=True):
    """Semantic target selection followed by measured depth, with no red gate."""
    import os
    threshold=float(os.getenv('HANDOFF_YOLO_MIN_CONFIDENCE', '.30'))
    if not np.isfinite(threshold) or not .30<=threshold<=.99:
        raise ValueError('HANDOFF_YOLO_MIN_CONFIDENCE must be in [0.30, 0.99]')
    result=(detector or YoloClient()).detect(frame)
    depth=frame['depth_m'];height,width=depth.shape
    valid=np.isfinite(depth)&(depth>.1)&(depth<2.5)&np.isfinite(world).all(axis=2)
    boxes=[b for b in result['detections'] if b['confidence']>threshold]
    if not boxes or (roi is None and len(boxes)!=1):
        raise RuntimeError('Coke YOLO target absent/ambiguous')
    masks=[];selected_boxes=[]
    for box in boxes:
        x1,y1,x2,y2=box['xyxy']
        # Inset the detection to exclude background at the box boundary.
        dx=(x2-x1)*.1;dy=(y2-y1)*.1
        inside=np.zeros(depth.shape,dtype=bool)
        inside[int(np.ceil(y1+dy)):int(np.floor(y2-dy)),
               int(np.ceil(x1+dx)):int(np.floor(x2-dx))]=True
        selected=inside&valid
        if roi is not None:selected &= roi
        if require_table:
            selected &= (world[:,:,2]>config['table_height_m']+.012)&(world[:,:,2]<config['table_height_m']+config['can_height_m']+.02)
        if selected.sum()<40:
            continue
        # Reject the complete selected box before cropping to the tracking ROI.
        if x1<6 or y1<6 or x2>width-6 or y2>height-6:
            raise RuntimeError('Coke YOLO target is clipped by the camera image boundary')
        foreground_depth=float(np.median(depth[selected]))
        selected &= np.abs(depth-foreground_depth)<=config['can_radius_m']+.01
        selected=binary_erosion(selected,iterations=2)
        if selected.sum()>=40:
            masks.append(selected);selected_boxes.append(box)
    if len(masks)!=1:
        raise RuntimeError('Coke YOLO depth target absent/ambiguous')
    return masks[0],{'detector':'coke_yolo','model_sha256':result['model_sha256'],
                     'backend':result['backend'],'box':selected_boxes[0],
                     'image_sha256':result['source']['image_sha256']}


def geometry(frame,config,require_table=True,expected_center_xy=None,*,detector=None,synthetic_red=False):
    if frame['meta']['camera_serial']!=config['camera_serial']:raise RuntimeError('Camera serial changed')
    depth=frame['depth_m']
    # Exact distortion-corrected RealSense rays, not simulated pinhole geometry.
    pts=frame['rays']*depth[...,None]
    t=np.asarray(config['camera_to_world']);world=pts@t[:3,:3].T+t[:3,3]
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
    if synthetic_red:
        # Explicitly reserved for untextured MuJoCo regression fixtures. The
        # physical runtime never passes this flag and has no color fallback.
        if frame['meta'].get('synthetic_rgbd') is not True:
            raise ValueError('Synthetic red detection requires a MuJoCo camera')
        target=target_red_mask(frame['color_bgr'][...,::-1],depth,roi=roi)
        detection={'detector':'synthetic_red'}
    else:
        target,detection=target_coke_mask(frame,config,world,roi,detector,require_table)
    points=world[target]
    if len(points)<40:raise RuntimeError('Target has too few valid depth points')
    points=points[::max(1,len(points)//1500)]
    fit=least_squares(lambda c:np.linalg.norm(points[:,:2]-c,axis=1)-config['can_radius_m'],points[:,:2].mean(axis=0),loss='soft_l1',f_scale=.002)
    residual=float(np.sqrt(np.mean(fit.fun**2)))
    if residual>.006:raise RuntimeError('Physical target cylinder fit exceeds 6 mm')
    if expected_center_xy is not None and np.linalg.norm(fit.x-expected)>.04:
        raise RuntimeError('Tracked target center differs by over 40 mm')
    table=config['table_height_m']
    if require_table:
        near=valid&~target&(np.linalg.norm(world[:,:,:2]-fit.x,axis=2)<.22)
        z=world[near,2];plane=z[np.abs(z-table)<.008]
        if len(plane)<100:raise RuntimeError('Independent measured table height disagrees with camera')
        if abs(np.median(plane)-table)>.005:raise RuntimeError('Table plane differs by over 5 mm')
        if np.quantile(points[:,2],.05)>table+.06 or np.quantile(points[:,2],.95)<table+.07:
            raise RuntimeError('Can is not upright on the measured table')
    if not synthetic_red:fresh(frame)
    obs=Observation(np.r_[fit.x,table+config['can_height_m']/2],table,frame['timestamp'],residual,len(points))
    return obs,{'top_quantile_z_m':float(np.quantile(points[:,2],.95)), 'center_xy_m':fit.x.tolist(),
                'fit_error_m':residual,'pixels':len(points),'frame_id':frame['meta']['frame_id'],**detection}
