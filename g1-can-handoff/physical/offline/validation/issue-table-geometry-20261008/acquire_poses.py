import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from operator_panel import Wendy
from trial_evidence import result_payload
out=Path('offline/validation/issue-table-geometry-20261008');m=json.loads((out/'input-manifest.json').read_text());targets=[dict(frame_id=f['native']['metadata']['frame_id'],timestamp=f['native']['metadata']['depth_time']) for c in m['cases'] for f in c['frames']]
script='''import gzip,hashlib,json,sys
from pathlib import Path
from export_cdr import decode
p=Path('/records/20261007T235101Z-76f038c2');manifest=json.loads((p/'manifest.json').read_text());targets=json.loads(sys.argv[1]);best={};sources=[]
for name in ['samples-000030.jsonl.gz','samples-000037.jsonl.gz']:
 file=p/name;h=hashlib.sha256(file.read_bytes()).hexdigest();expected=next(x['sha256'] for x in manifest['closed_segments'] if x['file']==name);assert h==expected;sources.append(dict(path=str(file),sha256=h))
 with gzip.open(file,'rt') as stream:
  for line in stream:
   row=json.loads(line)
   if row.get('topic')!='rt/lowstate':continue
   ts=row['monotonic_ns']/1e9
   for target in targets:
    diff=abs(ts-target['timestamp']);key=str(target['frame_id'])
    if diff<=.02 and (key not in best or diff<best[key]['skew_s']):best[key]=dict(skew_s=diff,raw=row.copy())
for key,v in best.items():
 decoded=decode(v['raw'].copy());msg=decoded['message'];v['pose']=dict(timestamp=decoded['monotonic_ns']/1e9,rpy=msg['imu_state']['rpy'],waist_q=[x['q'] for x in msg['motor_state'][12:15]]);v['decoded']=decoded
print(json.dumps(dict(recorder_session=manifest['session'],segments=sources,poses=best)))
'''
c=Wendy()
try:
 r=result_payload(c.tool('container_exec',dict(app_name='DEV-g1-handoff-recorder',command=['python3','-c',script,json.dumps(targets)],timeout_seconds=60,max_bytes=250000)))
 data=json.loads(r['stdout']);(out/'native-poses.json').write_text(json.dumps(data,indent=2)+'\n');print([(k,v['skew_s'],v['pose']) for k,v in data['poses'].items()])
finally:c.process.terminate()
