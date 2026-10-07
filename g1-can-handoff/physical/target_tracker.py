"""Initial semantic identity followed by conservative exact-frame RGB-D tracking.

select() only proposes pixels. accept() commits state after the caller's full
geometry/freshness checks. Every uncertainty reacquires YOLO on the same frame;
there is no stale-mask, color-only or last-known-target fallback.
"""
import hashlib
import numpy as np
from scipy.ndimage import binary_closing,binary_erosion,label

from yolo_client import YoloClient,fresh,validate_result


def color_bins(image):
    values=image.astype(np.uint16)//32
    return values[:,:,0]*64+values[:,:,1]*8+values[:,:,2]


def histogram(bins,mask):
    values=np.bincount(bins[mask],minlength=512).astype(float)
    return values/max(1,values.sum())


class TargetTracker:
    def __init__(self,detector=None,recheck_interval_s=1.0):
        if not 0<recheck_interval_s<=1.0:raise ValueError('Semantic recheck interval must be within0..1s')
        self.detector=detector or YoloClient();self.recheck_interval_s=recheck_interval_s
        self.state=None;self.pending=None

    def reset(self):
        self.state=None;self.pending=None

    def _semantic(self,frame,config,world,require_table,expected_center_xy,result,reason):
        from camera_input import target_coke_mask
        detection=validate_result(frame,self.detector.detect(frame) if result is None else result)
        mask,diagnostic=target_coke_mask(frame,config,world,require_table=require_table,result=detection)
        diagnostic={**diagnostic,'tracking':{'mode':'yolo','reason':reason,'semantic_frame_id':frame['meta']['frame_id']}}
        self.pending={'frame_id':frame['meta']['frame_id'],'mask':mask,'histogram':histogram(color_bins(frame['color_bgr']),mask),
            'diagnostic':diagnostic,'semantic_timestamp':frame['timestamp'],'semantic_frame_id':frame['meta']['frame_id']}
        return mask,diagnostic

    def select(self,frame,config,world,require_table=True,expected_center_xy=None,result=None):
        fresh(frame);self.pending=None
        if frame['meta']['camera_serial']!=config['camera_serial']:raise RuntimeError('Tracked camera serial changed')
        if self.state is None:
            return self._semantic(frame,config,world,require_table,expected_center_xy,result,'initial_identity')
        state=self.state
        if frame['meta']['frame_id']<=state['frame_id'] or frame['timestamp']<=state['timestamp']:
            raise RuntimeError('Tracker requires a distinct increasing exact camera frame')
        if frame['timestamp']-state['semantic_timestamp']>=self.recheck_interval_s:
            return self._semantic(frame,config,world,require_table,expected_center_xy,result,'periodic_recheck')
        depth=frame['depth_m']
        center=np.asarray(state['center_xy'] if expected_center_xy is None else expected_center_xy)
        if center.shape!=(2,) or not np.isfinite(center).all():raise RuntimeError('Tracker expected center is invalid')
        distance2=(world[:,:,0]-center[0])**2+(world[:,:,1]-center[1])**2
        # Nonfinite X/Y already fail the squared-distance comparison. Retain
        # an explicit Z check without reducing a full HxWx3 temporary.
        valid=np.isfinite(depth)&(depth>.1)&(depth<2.5)&np.isfinite(world[:,:,2])
        roi=valid&(distance2<(config['can_radius_m']+.04)**2)&(np.abs(depth-state['depth_median'])<.12)
        if require_table:
            roi&=(world[:,:,2]>config['table_height_m']+.012)&(world[:,:,2]<config['table_height_m']+config['can_height_m']+.02)
        # Bound work by ALL eligible measured-world pixels, retaining other
        # possible components for ambiguity checks. Four false border pixels
        # preserve the existing two-iteration closing at crop boundaries.
        rows,columns=np.nonzero(roi)
        if not len(rows):
            return self._semantic(frame,config,world,require_table,expected_center_xy,result,'insufficient_visual_depth_support')
        ys=slice(max(0,int(rows.min())-4),min(depth.shape[0],int(rows.max())+5))
        xs=slice(max(0,int(columns.min())-4),min(depth.shape[1],int(columns.max())+5))
        full_world=world
        roi=roi[ys,xs];depth=depth[ys,xs];world=world[ys,xs]
        bins=color_bins(frame['color_bgr'][ys,xs])
        # Keep measured pixels whose color belonged to the semantically
        # confirmed can. Closing groups printing/finger fragments only within
        # this bounded measured-depth/world ROI; it never invents fit points.
        selected=roi&(state['histogram'][bins]>=.002)
        if selected.sum()>=40:
            z=world[selected,2];z95=float(np.quantile(z,.95));span=z95-float(np.quantile(z,.05))
            if require_table:
                top=config['table_height_m']+config['can_height_m']
                if span>.03:top=min(top,z95)
            else:top=z95 if span>.03 else None
            if top is not None:selected&=world[:,:,2]<top-.01
            if selected.sum()>=40:
                selected&=np.abs(depth-float(np.median(depth[selected])))<=config['can_radius_m']+.01
        components,count=label(binary_closing(selected,iterations=2)&roi)
        sizes=np.bincount(components.ravel());sizes[0]=0
        order=np.argsort(sizes)[::-1];largest=int(order[0])
        minimum=max(40,int(state['initial_support']*.2))
        reason=None
        if sizes[largest]<minimum:reason='insufficient_visual_depth_support'
        elif len(order)>1 and sizes[order[1]]>=max(40,sizes[largest]*.2):reason='ambiguous_visual_depth_components'
        else:
            mask=binary_erosion(selected&(components==largest),iterations=2)
            support=int(mask.sum())
            agreement=float(np.sqrt(state['histogram']*histogram(bins,mask)).sum())
            full_mask=np.zeros(frame['depth_m'].shape,dtype=bool);full_mask[ys,xs]=mask
            mask=full_mask
            if support<minimum:reason='insufficient_eroded_support'
            elif agreement<.72:reason='visual_identity_uncertain'
            elif (mask[:6].any() or mask[-6:].any() or mask[:,:6].any() or mask[:,-6:].any()):
                reason='tracked_target_clipped'
        if reason is not None:
            return self._semantic(frame,config,full_world,require_table,expected_center_xy,result,reason)
        fresh(frame)
        diagnostic={**state['diagnostic'],'detector':'coke_visual_depth_tracker',
            'image_sha256':hashlib.sha256(frame['color_bgr'].tobytes()).hexdigest(),
            'tracking':{'mode':'visual_depth','semantic_frame_id':state['semantic_frame_id'],
            'semantic_age_s':frame['timestamp']-state['semantic_timestamp'],'visual_agreement':agreement,'support_pixels':support,
            'roi_shape':list(roi.shape)}}
        self.pending={**state,'frame_id':frame['meta']['frame_id'],'mask':mask,'diagnostic':diagnostic}
        return mask,diagnostic

    def accept(self,observation,vision,frame):
        fresh(frame)
        if self.pending is None or self.pending['frame_id']!=frame['meta']['frame_id']:
            raise RuntimeError('Tracker acceptance does not match pending exact frame')
        # The caller has validated the same fit/table/workspace before this
        # commit; the tracker cannot convert a proposed mask into motion.
        center=np.asarray(observation.center[:2],dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():raise RuntimeError('Accepted tracker center invalid')
        if self.state is not None and np.linalg.norm(center-self.state['center_xy'])>.04:
            raise RuntimeError('Tracked target center differs by over40mm')
        state=self.pending
        state.update(timestamp=frame['timestamp'],center_xy=center,
            depth_median=float(np.median(frame['depth_m'][state['mask']])),
            initial_support=int(state['mask'].sum()) if self.state is None else self.state['initial_support'])
        self.state=state;self.pending=None
