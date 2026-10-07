"""Exact-frame camera client, calibrated ray geometry, and real visual lift check."""
import io
import json
import time
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from scipy.ndimage import binary_erosion,maximum_filter,minimum_filter,label
from scipy.optimize import least_squares
from g1_can_handoff.perception import Observation
from yolo_client import YoloClient, fresh, frame_age_limit_s

# A bounded inference worker avoids per-observation Python thread joins.
# Each caller joins its exact-frame result; the client checks age when queued
# work starts and after it returns, so queuing cannot admit an old frame.
_geometry_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='exact-frame-yolo')


class Camera:
    def __init__(self):
        self.base='http://127.0.0.1:8001';self.token=None;self.rays=None;self.intrinsics=None;self.serial=None
        self.lease_stop=threading.Event();self.lease_thread=None;self.lease_error=None
        self.frame_executor=None
        # Pillow otherwise loads its decoder plugins during the first bounded
        # observation. Warm them before acquiring a camera lease or moving.
        Image.init()
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
        if self.frame_executor is not None:
            self.frame_executor.shutdown(wait=True,cancel_futures=True)
            self.frame_executor=None
    def frame(self, *, materialize=True):
        start=time.perf_counter()
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
        # Both bundles address the same immutable frame ID. Fetch independent
        # payloads concurrently, then retain all identity/skew/freshness checks.
        if self.frame_executor is None:
            self.frame_executor=ThreadPoolExecutor(max_workers=2,thread_name_prefix='camera-frame')
        timing_request=self.frame_executor.submit(self.request,'/frame.policy-rgbd?frame_id='+str(fid))
        compact_request=self.frame_executor.submit(self.request,'/frame.recorder-rgbd?frame_id='+str(fid))
        timing=timing_request.result()
        with np.load(io.BytesIO(timing),allow_pickle=False) as f:
            meta=json.loads(str(f['metadata_json']))
        compact=compact_request.result()
        fetched=time.perf_counter()
        with zipfile.ZipFile(io.BytesIO(compact)) as z:
            cm=json.loads(z.read('metadata.json'))
            png=z.read('depth.png');jpg=z.read('color.jpg')
        # Both C decoders release the GIL. Decode the immutable paired payloads
        # concurrently, retaining exactly the same Pillow pixels and scale.
        depth_request=self.frame_executor.submit(decode_depth,png,cm['calibration']['depth_scale_m'])
        color_request=self.frame_executor.submit(decode_color,jpg)
        depth=depth_request.result();color=color_request.result()
        if cm['source_frame_id'] != fid or meta['frame_id'] != fid or cm['captured_at_unix_ns'] != meta['captured_at_unix_ns']:
            raise RuntimeError('Camera color/depth exact-frame identity mismatch')
        if meta['intrinsics'] != self.intrinsics or meta['camera_serial'] != self.serial:
            self.rays=None;raise RuntimeError('Camera calibration changed; refresh ray cache')
        if depth.shape!=color.shape[:2] or self.rays.shape!=(*depth.shape,3):
            raise RuntimeError('Camera color/depth/ray geometry dimensions disagree')
        if cm['calibration']['intrinsics']!=self.intrinsics:
            raise RuntimeError('Camera compact-frame intrinsics disagree with cached rays')
        timestamp=meta['depth_time']
        if not meta.get('timestamp_valid') or not 0<=time.monotonic()-timestamp<=frame_age_limit_s():
            raise RuntimeError('Camera exposure timestamp invalid/stale')
        if abs(meta['color_time']-timestamp)>.012:raise RuntimeError('Camera RGB/depth exposure skew exceeds 12 ms')
        frame={'color_bgr':color,'depth_m':depth,'rays':self.rays,'meta':meta,'timestamp':timestamp,
               'color_codec':'jpeg_exact_source_frame','source_rgbd_zip':compact,'source_timing_npz':timing,
               'timings_ms':{'fetch':(fetched-start)*1000,'decode':(time.perf_counter()-fetched)*1000}}
        if materialize:
            frame['raw']=materialize_rgbd(frame)
        return frame


def decode_depth(png,scale):
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image,dtype=np.float32)*scale


def decode_color(jpg):
    with Image.open(io.BytesIO(jpg)) as image:
        return np.asarray(image.convert('RGB'))[...,::-1].copy()


def materialize_rgbd(frame):
    bundle=io.BytesIO()
    np.savez(bundle,color_bgr=frame['color_bgr'],depth_m=frame['depth_m'],rays=frame['rays'],
             metadata_json=json.dumps({**frame['meta'],'color_codec':frame['color_codec']}))
    return bundle.getvalue()



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

def target_coke_mask(frame, config, world, roi=None, detector=None, require_table=True, result=None):
    """Semantic target selection followed by measured depth, with no red gate."""
    from yolo.confidence import confidence_threshold
    threshold=confidence_threshold()
    if result is None:
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
        ys=slice(int(np.ceil(y1+dy)),int(np.floor(y2-dy)))
        xs=slice(int(np.ceil(x1+dx)),int(np.floor(x2-dx)))
        local_world=world[ys,xs];local_depth=depth[ys,xs]
        selected=valid[ys,xs].copy()
        if roi is not None:selected &= roi[ys,xs]
        if require_table:
            selected &= (local_world[:,:,2]>config['table_height_m']+.012)&(local_world[:,:,2]<config['table_height_m']+config['can_height_m']+.02)
        if selected.sum()<40:
            continue
        # Reject the complete selected box before cropping to the tracking ROI.
        if x1<6 or y1<6 or x2>width-6 or y2>height-6:
            raise RuntimeError('Coke YOLO target is clipped by the camera image boundary')
        # YOLO includes the silver lid. Its interior is a disk, not a cylinder
        # sidewall, and must not enter the radial fit. Remove the top 10 mm
        # before estimating foreground depth, so the lid cannot dominate it.
        z=local_world[selected,2]
        z95=float(np.quantile(z,.95));span=z95-np.quantile(z,.05)
        if require_table:
            top=config['table_height_m']+config['can_height_m']
            if span>.03:
                top=min(top,z95)
        elif span>.03:
            top=z95
        else:
            top=None
        if top is not None:
            selected &= local_world[:,:,2]<top-.01
        if selected.sum()<40:
            continue
        foreground_depth=float(np.median(local_depth[selected]))
        selected &= np.abs(local_depth-foreground_depth)<=config['can_radius_m']+.01
        selected=binary_erosion(selected,iterations=2)
        if selected.sum()>=40:
            # Pixels outside this exact inset were false in the full-image
            # mask. The crop's false border preserves the original erosion.
            target=np.zeros(depth.shape,dtype=bool);target[ys,xs]=selected
            masks.append(target);selected_boxes.append(box)
    if len(masks)!=1:
        raise RuntimeError('Coke YOLO depth target absent/ambiguous')
    return masks[0],{'detector':'coke_yolo','model_sha256':result['model_sha256'],
                     'backend':result['backend'],'box':selected_boxes[0],
                     'image_sha256':result['source']['image_sha256']}


def geometry(frame,config,require_table=True,expected_center_xy=None,*,detector=None,synthetic_red=False):
    if frame['meta']['camera_serial']!=config['camera_serial']:raise RuntimeError('Camera serial changed')
    depth=frame['depth_m']
    # Exact distortion-corrected RealSense rays, not simulated pinhole geometry.
    if synthetic_red:
        result=None
        pts=frame['rays']*depth[...,None]
        t=np.asarray(config['camera_to_world']);world=pts@t[:3,:3].T+t[:3,3]
    else:
        # HTTP inference and the independent ray transform can overlap. The
        # detector still sees the same frame and validates its exact identity.
        detection=_geometry_executor.submit((detector or YoloClient()).detect,frame)
        pts=frame['rays']*depth[...,None]
        t=np.asarray(config['camera_to_world']);world=pts@t[:3,:3].T+t[:3,3]
        result=detection.result()
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
        target,detection=target_coke_mask(frame,config,world,roi,detector,require_table,result)
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
        # Avoid a full HxWx2 temporary and square root for the table-neighbor
        # test. Squared distance retains the identical 220 mm boundary.
        distance2=(world[:,:,0]-fit.x[0])**2+(world[:,:,1]-fit.x[1])**2
        near=valid&~target&(distance2<.22**2)
        z=world[near,2];plane=z[np.abs(z-table)<.008]
        if len(plane)<100:raise RuntimeError('Independent measured table height disagrees with camera')
        if abs(np.median(plane)-table)>.005:raise RuntimeError('Table plane differs by over 5 mm')
        if np.quantile(points[:,2],.05)>table+.06 or np.quantile(points[:,2],.95)<table+.07:
            raise RuntimeError('Can is not upright on the measured table')
    if not synthetic_red:fresh(frame)
    obs=Observation(np.r_[fit.x,table+config['can_height_m']/2],table,frame['timestamp'],residual,len(points))
    return obs,{'top_quantile_z_m':float(np.quantile(points[:,2],.95)), 'center_xy_m':fit.x.tolist(),
                'fit_error_m':residual,'pixels':len(points),'frame_id':frame['meta']['frame_id'],**detection}
