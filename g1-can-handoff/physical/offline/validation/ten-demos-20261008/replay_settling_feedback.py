import gzip,json,hashlib,pathlib,numpy as np
from export_cdr import decode
root=pathlib.Path('/records/20261008T162358Z-ab00e585')
windows=[(2,86,1791481826783523549,1791481831809654113),(3,88,1791481957009910278,1791481962035613145)]
out=[]
for trial,seg,start,end in windows:
 p=root/f'samples-{seg:06d}.jsonl.gz'; rows=[]
 with gzip.open(p,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r.get('topic')!='rt/lowstate' or not start<=r['unix_ns']<=end:continue
   m=decode(r)['message']; rows.append([r['unix_ns'],m['imu_state']['rpy'],[s['q'] for s in m['motor_state'][:29]],[s['dq'] for s in m['motor_state'][:29]]])
 def assess(sampled):
  t=np.array([r[0] for r in sampled],dtype=np.int64); poses=np.array([r[1]+r[2] for r in sampled]);poses[:,:3]=np.unwrap(poses[:,:3],axis=0); dq=np.abs([r[3] for r in sampled]); scores=[]
  for j in range(len(t)):
   i=int(np.searchsorted(t,t[j]-550000000))
   if t[j]-t[i]<500000000:continue
   span=np.ptp(poses[i:j+1],axis=0); speeds=np.max(dq[i:j+1],axis=0)
   values=[float(np.max(span[:3])),float(np.max(span[3:18])),float(np.max(span[18:])),float(np.max(speeds))]
   ratios=np.array(values)/[.002,.005,.015,.03]
   scores.append(dict(offset_s=float((t[j]-start)/1e9),imu_span_rad=values[0],base_joint_span_rad=values[1],arm_span_rad=values[2],max_speed_rad_s=values[3],speed_joint_index=int(np.argmax(speeds)),ratios=ratios.tolist()))
  best=min(scores,key=lambda s:max(s['ratios']))
  return dict(samples=len(t),windows=len(scores),passing_windows=sum(all(r<=1 for r in s['ratios']) for s in scores),best_window=best,min_window_speed_rad_s=min(s['max_speed_rad_s'] for s in scores),criterion_failure_fraction=[sum(s['ratios'][k]>1 for s in scores)/len(scores) for k in range(4)])
 sampled=[];previous=-1
 for row in rows:
  if row[0]-previous>=20000000:sampled.append(row);previous=row[0]
 out.append(dict(trial=trial,segment=p.name,segment_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),start_unix_ns=start,end_unix_ns=end,all_recorded=assess(rows),approximate_50hz=assess(sampled)))
print(json.dumps(dict(method='Read-only CDR feedback replay; recorder receipt times and approximate 20ms sampling, not exact controller loop sample selection',criterion_order=['imu_span','base_joint_span','arm_span','speed_all_joints'],trials=out)))
