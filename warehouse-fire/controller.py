"""Shared thermal-frame controller for simulator and calibrated hardware."""
import numpy as np
from twin import zone_at

class FireController:
    def __init__(self,hits,cameras,emit,actuate):
        self.hits=hits; self.cameras=cameras; self.emit=emit; self.actuate=actuate
        self.incidents=[]; self.candidates={}; self.next_id=1

    def update(self,frames,now):
        seen=set()
        for cid,frame in enumerate(frames):
            mask=(frame>=80)&np.isfinite(self.hits[cid,:,:,0])
            # Connected components keep simultaneous fires in one camera separate.
            pending=set(map(tuple,np.argwhere(mask)))
            while pending:
                first=pending.pop(); stack=[first]; pixels=[first]
                while stack:
                    y,x=stack.pop()
                    for cell in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                        if cell in pending: pending.remove(cell); stack.append(cell); pixels.append(cell)
                if len(pixels)<2: continue
                ys,xs=np.array(pixels).T
                weights=frame[ys,xs]-22
                position=np.average(self.hits[cid,ys,xs],axis=0,weights=weights)
                # Stable confirmation is per camera, so repeated views cannot speed it up.
                key=next((k for k in self.candidates if k[0]==cid and np.linalg.norm(position[:2]-np.array(k[1:]))<1.2), (cid,float(position[0]),float(position[1])))
                already_seen=key in seen
                seen.add(key)
                count,last=self.candidates.get(key,(0,-10))
                count=count if already_seen else count+1 if now-last<.6 else 1
                self.candidates[key]=(count,now)
                existing=next((i for i in self.incidents if i['state']!='resolved' and np.linalg.norm(position-np.array(i['position']))<1.7),None)
                if existing:
                    existing['last_hot']=now; existing['peak_c']=round(max(existing['peak_c'],float(frame[ys,xs].max())),1)
                    if self.cameras[cid]['id'] not in existing['cameras']: existing['cameras'].append(self.cameras[cid]['id'])
                    continue
                if count<3: continue
                zone=zone_at(position)
                incident=dict(id=self.next_id,position=position.round(3).tolist(),zone=zone,cameras=[self.cameras[cid]['id']],peak_c=round(float(frame[ys,xs].max()),1),state='detected',detected_at=now,last_hot=now,pixel_count=len(pixels))
                self.next_id+=1; self.incidents.append(incident)
                self.emit('fire_detected',**incident)
                try:
                    self.actuate(zone,True)
                    incident['state']='suppressing'
                    self.emit('sprinkler_on',incident_id=incident['id'],zone=zone)
                except Exception as exc:
                    incident['state']='actuator_fault'; self.emit('actuator_fault',incident_id=incident['id'],error=str(exc))
        self.candidates={k:v for k,v in self.candidates.items() if k in seen}
        for incident in self.incidents:
            if incident['state']=='suppressing' and now-incident['last_hot']>5:
                # Close only after a five-second cool hold and no other hot incident in the zone.
                other=any(i is not incident and i['zone']==incident['zone'] and i['state']!='resolved' and now-i['last_hot']<=5 for i in self.incidents)
                if not other:
                    try:
                        self.actuate(incident['zone'],False)
                    except Exception as exc:
                        incident['state']='actuator_fault'; self.emit('actuator_fault',incident_id=incident['id'],error=str(exc)); continue
                incident['state']='resolved'; self.emit('fire_resolved',incident_id=incident['id'],zone=incident['zone'])
