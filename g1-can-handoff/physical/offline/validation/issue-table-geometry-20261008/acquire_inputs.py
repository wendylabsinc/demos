import base64,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from operator_panel import Wendy
from trial_evidence import result_payload
out=Path('offline/validation/issue-table-geometry-20261008');out.mkdir(parents=True,exist_ok=True)
t=json.loads(gzip.decompress(Path('offline/trial-events.json.gz').read_bytes()))
c=Wendy();manifest={'device':'10.10.20.238:50052','read_only':True,'files':{},'cases':[]}
def run(app,script,*args,max_bytes=170000,timeout=30):
 return result_payload(c.tool('container_exec',dict(app_name=app,command=['python3','-c',script,*map(str,args)],timeout_seconds=timeout,max_bytes=max_bytes)))['stdout']
try:
 for a,fid,sem in [('20261008T002131Z',27901,27892),('20261008T002809Z',34006,34004)]:
  es=t[a]['events'];cfg=next(e['calibration'] for e in es if e['phase']=='plan_context');case=dict(attempt=a,fault_frame_id=fid,semantic_frame_id=sem,calibration=cfg,frames=[])
  for id_ in [cfg['source_frame_id']]+[e['frame_id'] for e in es if e['phase']=='perception_result' and sem<=e['frame_id']<=fid]:
   event=next(e for e in es if e['phase']=='phase_camera_evidence' and e['metadata']['frame_id']==id_)
   stem=Path(event['path']).stem
   for suffix in ['.source.zip','.timing.npz']:
    remote='/data/'+stem+suffix
    row=json.loads(run('DEV-g1-handoff-physical',"from pathlib import Path\nimport hashlib,json,sys\np=Path(sys.argv[1]);print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))",remote))
    p=out/(stem+suffix)
    with p.open('wb') as f:
     for offset in range(0,row['bytes'],120000):
      raw=base64.b64decode(run('DEV-g1-handoff-physical',"import base64,sys\nwith open(sys.argv[1],'rb') as f:\n f.seek(int(sys.argv[2]));print(base64.b64encode(f.read(120000)).decode())",remote,offset).strip(),validate=True);f.write(raw)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'];manifest['files'][p.name]=dict(**row,remote=remote)
   localray=np.load('offline/evidence/rgbd/rays.npz')['rays'];rayhash=hashlib.sha256(localray.tobytes()).hexdigest()
   native=json.loads(run('DEV-g1-handoff-physical',"import json,hashlib,numpy as np,sys\nwith np.load(sys.argv[1]) as d:\n print(json.dumps(dict(rays_sha256=hashlib.sha256(d['rays'].tobytes()).hexdigest(),color_sha256=hashlib.sha256(d['color_bgr'].tobytes()).hexdigest(),depth_sha256=hashlib.sha256(d['depth_m'].tobytes()).hexdigest(),npz_sha256=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(),metadata=json.loads(str(d['metadata_json'])))))",event['path']))
   assert rayhash==native['rays_sha256'];assert native['metadata']['captured_at_unix_ns']==event['metadata']['captured_at_unix_ns'];assert native['metadata']['depth_time']==event['metadata']['depth_time'];assert native['metadata']['camera_serial']==cfg['camera_serial']
   result=next((e for e in es if e['phase']=='perception_result' and e['frame_id']==id_),None)
   if result and result.get('source'):assert result['source']['image_sha256']==native['color_sha256']
   case['frames'].append(dict(stem=stem,event=event,native=native,perception_result=result))
   print('Verified archive',stem,flush=True)
  manifest['cases'].append(case)
 (out/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
finally:c.process.terminate()
