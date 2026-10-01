"""Persistent flight imagery, geometric registration and learned change labels.

Only camera frames enter this module. It has no scenario flags or scene objects.
"""
from __future__ import annotations
import json
import os
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .navigation import unproject


def ground_projection(meta):
    k=meta['intrinsics']
    intrinsic=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]])
    rotation=np.diag([1,-1,-1])@np.asarray(meta['cameraRotation']).T
    translation=-rotation@np.asarray(meta['cameraPosition'])
    return intrinsic@np.column_stack([rotation[:,0],rotation[:,1],translation])


def register(previous, current, previous_meta, current_meta):
    """Accept a visual warp only when it agrees with measured camera geometry.

    Repeated yard edges and elevated surfaces can produce a convincing ORB
    homography for the wrong plane. The calibrated ground plane constrains the
    warp because survey flights revisit the measured baseline camera poses.
    """
    calibrated=ground_projection(current_meta)@np.linalg.inv(ground_projection(previous_meta))
    orb=cv2.ORB_create(nfeatures=1400,edgeThreshold=12,fastThreshold=8)
    p,kp=orb.detectAndCompute(cv2.cvtColor(previous,cv2.COLOR_RGB2GRAY),None)
    c,kc=orb.detectAndCompute(cv2.cvtColor(current,cv2.COLOR_RGB2GRAY),None)
    transform=None; inliers=0; method='calibrated ground-plane reprojection'
    rejection=None;deviation=None
    if kp is not None and kc is not None:
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(kp,kc,k=2)
        good=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
        if len(good)>=8:
            transform,mask=cv2.findHomography(np.float32([p[m.queryIdx].pt for m in good]),
                np.float32([c[m.trainIdx].pt for m in good]),cv2.RANSAC,2.5)
            inliers=int(mask.sum()) if mask is not None else 0
            if transform is None or inliers<8:
                transform=None;rejection='Insufficient feature inliers'
            else:
                # A spread grid checks the whole image rather than only the
                # feature cluster that generated the candidate homography.
                ph,pw=previous.shape[:2]
                x,y=np.meshgrid(np.linspace(.05*pw,.95*pw,3),np.linspace(.05*ph,.95*ph,3))
                points=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
                measured=points@calibrated.T;visual=points@transform.T
                if (not np.isfinite(visual).all() or not np.isfinite(measured).all() or
                        np.any(np.abs(visual[:,2])<1e-9) or np.any(np.abs(measured[:,2])<1e-9)):
                    transform=None;rejection='Non-finite or singular feature projection'
                else:
                    measured=measured[:,:2]/measured[:,2,None]
                    visual=visual[:,:2]/visual[:,2,None]
                    errors=np.linalg.norm(visual-measured,axis=1)
                    deviation={'median':round(float(np.median(errors)),3),'maximum':round(float(errors.max()),3)}
                    if np.median(errors)>2. or errors.max()>5.:
                        transform=None;rejection='Feature warp disagrees with calibrated ground plane'
                    else:method='ORB features and RANSAC homography'
    if transform is None:
        transform=calibrated
    h,w=current.shape[:2]
    aligned=cv2.warpPerspective(previous,transform,(w,h))
    valid=cv2.warpPerspective(np.full(previous.shape[:2],255,np.uint8),transform,(w,h))>250
    valid=cv2.erode(valid.astype(np.uint8),np.ones((11,11),np.uint8)).astype(bool)
    return aligned,valid,{'method':method,'featureInliers':inliers,'validFraction':round(float(valid.mean()),3),
                         'featureRejection':rejection,'calibrationDeviationPixels':deviation}


def compare_images(previous,current,previous_meta,current_meta):
    aligned,valid,registration=register(previous,current,previous_meta,current_meta)
    if not valid.any():
        return aligned,[],{**registration,'comparison':'No valid image overlap'}
    before=cv2.GaussianBlur(aligned,(5,5),0).astype(np.float32)
    after=cv2.GaussianBlur(current,(5,5),0).astype(np.float32)
    # Estimate modest illumination differences using the median unchanged view.
    offset=np.median((after-before)[valid],axis=0) if valid.any() else np.zeros(3)
    difference=np.max(np.abs(after-before-offset),axis=2)
    changed=((difference>35)&valid).astype(np.uint8)
    changed=cv2.morphologyEx(changed,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    changed=cv2.morphologyEx(changed,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    changed&=valid.astype(np.uint8)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(changed)
    regions=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<85: continue
        regions.append({'box':[x,y,w,h],'pixelCentre':centres[i].tolist(),
                        'changedPixels':area,'changePercent':round(area/changed.size*100,2)})
    return aligned,regions,registration


class MissionTracker:
    def __init__(self,data_dir=None,vision=None):
        self.root=Path(data_dir or os.environ.get('YARD_DRONE_DATA','/data'))
        self.root.mkdir(parents=True,exist_ok=True)
        self.evidence=self.root/'evidence';self.evidence.mkdir(exist_ok=True)
        self.vision=vision
        self.baseline={'available':False,'zones':[],'id':None,'version':2,'observations':{}}
        self.images={}
        self.mission={'id':None,'kind':None,'status':'idle','scannedZones':[],'progress':0,'viewCount':0}
        self.incidents=[];self.observations={}
        saved=self.root/'baseline-camera.json'
        if saved.exists():
            try:
                candidate=json.loads(saved.read_text())
                loaded={key:np.array(Image.open(self.evidence/value['image']).convert('RGB'))
                        for key,value in candidate['observations'].items()}
                if candidate['version']==2 and loaded:
                    self.baseline,self.images=candidate,loaded
            except (OSError,ValueError,KeyError): pass

    def begin(self,kind):
        if kind=='survey':kind='current'
        if kind not in ['baseline','current']:raise ValueError('Choose baseline or current mission.')
        if kind=='current' and not self.baseline['available']:raise ValueError('Record a baseline flight first.')
        self.mission={'id':uuid.uuid4().hex[:12],'kind':kind,'status':'running','scannedZones':[],
                      'progress':0,'viewCount':0,'startedAt':time.time()}
        self.incidents=[];self.observations={}

    def prepare(self,key,frame,position,zone,mission_id):
        image=frame['image']
        metadata={k:frame[k] for k in ['cameraPosition','cameraRotation','intrinsics','timestamp']}
        name=f'{mission_id}-{key}.png'
        Image.fromarray(image).save(self.evidence/name)
        observation={**metadata,'pose':np.asarray(position).tolist(),'image':name,'zone':zone,
                     'simulationTime':float(frame['timestamp'])}
        findings=[];registration=None
        if self.mission['kind']=='current' and key in self.images:
            aligned,regions,registration=compare_images(self.images[key],image,
                                                       self.baseline['observations'][key],frame)
            for region in regions:
                x,y,w,h=region['box'];margin=18
                crop=image[max(0,y-margin):min(image.shape[0],y+h+margin),
                           max(0,x-margin):min(image.shape[1],x+w+margin)]
                result=self.vision.classify(crop) if self.vision else {'label':'visual change','confidence':0,'candidates':[]}
                label=result.get('label','visual change')
                if label=='oil leak':label='oil'
                if label not in ['fire','oil','person']:label='visual change'
                u,v=region.pop('pixelCentre')
                location=unproject(frame,u,v).tolist()
                findings.append({**region,'type':label,'status':'new','severity':'critical' if label=='fire' else 'review',
                    'confidence':result.get('confidence',0),'classifierSource':result.get('source'),
                    'uncertain':result.get('uncertain',True),'predictedLabel':result.get('predictedLabel'),'candidates':result.get('candidates',[]),
                    'outOfDomain':result.get('outOfDomain',False),'abstentionReason':result.get('reason'),
                    'scoreType':result.get('scoreType'),
                    'position':location,'zone':zone,'time':float(frame['timestamp']),
                    'id':f'{mission_id}-{key}-{len(findings)}','baselineId':self.baseline['id'],
                    'baselineUrl':f"/api/evidence/{self.baseline['observations'][key]['image']}",
                    'currentUrl':f'/api/evidence/{name}','registration':registration})
        return {'key':key,'missionId':mission_id,'observation':observation,'findings':findings,'registration':registration}

    def record(self,result,coverage=0):
        if result['missionId']!=self.mission['id'] or self.mission['status']!='running':return False
        self.observations[result['key']]=result['observation']
        self.incidents.extend(result['findings'])
        zone=result['observation']['zone']
        if zone and zone not in self.mission['scannedZones']:self.mission['scannedZones'].append(zone)
        self.mission['viewCount']=len(self.observations)
        expected=len(self.baseline['observations']) if self.mission['kind']=='current' else 0
        self.mission['progress']=len(self.observations)/expected if expected else min(.99,coverage/100)
        self.mission['coveragePercent']=coverage
        self.mission['registration']=result['registration']
        return True

    def finish(self):
        if self.mission['status']!='running' or not self.observations:raise ValueError('Capture flight imagery before finishing.')
        if self.mission['kind']=='current' and set(self.observations)!=set(self.baseline['observations']):
            raise ValueError('Every baseline view must be observed before completing the comparison.')
        self.mission.update(status='complete',progress=1,completedAt=time.time())
        if self.mission['kind']=='baseline':
            self.baseline={'version':2,'id':self.mission['id'],'available':True,'createdAt':time.time(),
                'zones':self.mission['scannedZones'].copy(),'observations':self.observations.copy(),
                'sensor':'MuJoCo perspective RGB/depth camera at measured airframe pose',
                'coveragePercent':self.mission.get('coveragePercent',0)}
            self.images={k:np.array(Image.open(self.evidence/v['image']).convert('RGB')) for k,v in self.observations.items()}
            temporary=self.root/'baseline-camera.tmp';temporary.write_text(json.dumps(self.baseline))
            os.replace(temporary,self.root/'baseline-camera.json')
        (self.root/f"mission-{self.mission['id']}.json").write_text(json.dumps({**self.snapshot(),'observations':self.observations},indent=2))

    def cancel(self):
        if self.mission['status']=='running':self.mission['status']='interrupted'

    def snapshot(self):
        # Images stay on disk; SSE includes metadata, not a growing image archive.
        benchmark={k:v for k,v in self.baseline.items() if k!='observations'}
        benchmark['viewCount']=len(self.baseline['observations'])
        return {'mission':self.mission.copy(),'benchmark':benchmark,'incidents':[dict(x) for x in self.incidents]}
