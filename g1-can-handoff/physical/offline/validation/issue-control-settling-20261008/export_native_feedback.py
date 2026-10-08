import gzip,json,pathlib,hashlib,base64,copy
from export_cdr import decode
root=pathlib.Path('/records/20261008T162358Z-ab00e585')
windows=[(2,86,1791481826783523549,1791481831809654113,'0c2416217683eb965b376c731adcf773afb977aad31b4c5a7995426c6e45b7a5'),(3,88,1791481957009910278,1791481962035613145,'79bb4623af49cffaea852b8e359b32713ef473740afa89c9417ed3788d8b6045')]
out=[]
for trial,seg,start,end,expected in windows:
 p=root/f'samples-{seg:06d}.jsonl.gz'
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest==expected,(p,digest)
 rows=[];previous={}
 with gzip.open(p,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r.get('topic') not in ['rt/lowstate','rt/dex3/left/state','rt/dex3/right/state'] or not start-200000000<=r.get('unix_ns',0)<=end:continue
   if r['unix_ns']-previous.get(r['topic'],0)<20000000:continue
   previous[r['topic']]=r['unix_ns']
   d=decode(copy.deepcopy(r));m=d['message'];count=29 if r['topic']=='rt/lowstate' else 7;motors=m['motor_state'][:count]
   sample={k:[s[k] for s in motors] for k in ['q','dq','tau_est']}
   sample.update(temperature_max=max(max(s['temperature']) for s in motors),faults=[s['motorstate'] for s in motors])
   if count==29:sample['imu_rpy']=m['imu_state']['rpy']
   rows.append({'topic':r['topic'],'unix_ns':r['unix_ns'],'monotonic_ns':r['monotonic_ns'],'receive_sequence':r['receive_sequence'],'sample':sample,'cdr_base64':r.get('cdr_base64')})
 out.append({'trial':trial,'source_segment':str(p),'source_segment_sha256':digest,'start_unix_ns':start,'end_unix_ns':end,'rows':rows})
payload=json.dumps({'decoder_sha256':hashlib.sha256(pathlib.Path('/app/export_cdr.py').read_bytes()).hexdigest(),'trials':out},separators=(',',':'),allow_nan=False).encode()
print(base64.b64encode(gzip.compress(payload,mtime=0)).decode())

