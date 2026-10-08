"""Run one explicitly supported, unobstructed physical reach comparison.

Each output filename is reserved before sending exactly one start request.
An uncertain acknowledgement is inspected, never retried. Full command/sample
events are exported through native Wendy. No camera/pickup policy is invoked.
"""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import time

from operator_panel import Wendy,DEPLOYING


def exec_json(client,script,args=(),max_bytes=30000):
    result=client.tool('container_exec',dict(app_name='DEV-g1-handoff-physical',
        command=['python3','-c',script,*args],timeout_seconds=15,max_bytes=max_bytes))
    payload=result.get('structuredContent')
    if payload is None:payload=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
    if payload['exit_code']!=0:raise RuntimeError(payload.get('stderr') or str(payload))
    return json.loads(payload['stdout'])


STATUS="""import json,time,urllib.request
s=json.load(urllib.request.urlopen('http://127.0.0.1:8102/status',timeout=5))
r=json.load(urllib.request.urlopen('http://127.0.0.1:8101/status',timeout=5))
print(json.dumps({'unix_ns':time.time_ns(),'status':s,'recorder':r}))
"""


def run(output,fixture_sha,kp,*,harness_supported,workspace_clear,readiness):
    if not harness_supported or not workspace_clear:raise ValueError('Explicit harness support and clear workspace are required')
    if DEPLOYING.exists():raise RuntimeError('Deployment is in progress')
    expected=json.loads(Path(readiness).read_text())['source_sha256']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump({'state':'reserved','kp':kp},f)
    data={'state':'reserved','kp':kp,'fixture_sha256':fixture_sha,'harness_supported':True,'workspace_clear':True}
    def save():output.write_text(json.dumps(data,indent=2)+'\n')
    client=Wendy();started=None
    try:
        hashes=exec_json(client,"import json,pathlib,hashlib,sys;e=json.loads(sys.argv[1]);print(json.dumps({n:hashlib.sha256((pathlib.Path('/app')/n).read_bytes()).hexdigest() for n in e}))",[json.dumps(expected)])
        if hashes!=expected:raise RuntimeError('Physical source differs from verified release')
        before=exec_json(client,STATUS);state=before['status']
        if state['demo_running'] or not state['telemetry_healthy'] or state.get('grasp_started'):
            raise RuntimeError('Physical controller is not ready for a harness replay')
        if state['adapter_initialized'] and state['phase']!='FROZEN':raise RuntimeError('Another arm operation owns control')
        data['before']=before;started=before['unix_ns'];data['started_unix_ns']=started
        request=dict(fixture='/data/joint-replays/reach-20261008.json',fixture_sha256=fixture_sha,
                     kp=kp,harness_confirmed=True,workspace_clear_confirmed=True)
        data['request']=request;data['state']='start_requested';save()
        script="""import json,sys,urllib.request
r=urllib.request.Request('http://127.0.0.1:8102/joint-replay/start',data=sys.argv[1].encode(),headers={'Content-Type':'application/json'},method='POST')
print(json.dumps(json.load(urllib.request.urlopen(r,timeout=10))))
"""
        # Exactly one request, even if transport/acknowledgement is uncertain.
        try:data['acknowledgement']=exec_json(client,script,[json.dumps(request)])
        except Exception as exc:data['acknowledgement_error']=str(exc)
        save();deadline=time.monotonic()+125;last_phase=None
        while True:
            after=exec_json(client,STATUS);state=after['status'];data['after']=after;save()
            if state['phase']!=last_phase:
                last_phase=state['phase'];print(json.dumps({'kp':kp,'phase':last_phase}),flush=True)
            if not state['demo_running']:break
            if time.monotonic()>deadline:
                client.request(action='stop');data['stop_requested']=True;save()
                raise RuntimeError('Harness replay exceeded its bounded deadline')
            time.sleep(.5)
        data['state']='terminal';data['metrics']=(state.get('demo_result') or {}).get('joint_replay')
        data['error']=state.get('demo_error') or state.get('error');save()
        return data
    except Exception as exc:
        data['driver_error']=str(exc);save();raise
    finally:
        if started is not None:
            script="""import base64,gzip,hashlib,json,pathlib,sys
rows=[]
for p in pathlib.Path('/data').glob('events-*.jsonl'):
 for line in p.read_text().splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('unix_ns',0)>=int(sys.argv[1]):rows.append(r)
raw=gzip.compress(('\\n'.join(json.dumps(r) for r in rows)+'\\n').encode(),compresslevel=1)
p=pathlib.Path('/data/harness-replay-export.gz');p.write_bytes(raw)
print(json.dumps({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'events':len(rows)}))
"""
            try:
                export=exec_json(client,script,[str(started)])
                raw=bytearray()
                for offset in range(0,export['bytes'],120000):
                    piece=exec_json(client,"import base64,json,pathlib,sys;b=pathlib.Path('/data/harness-replay-export.gz').read_bytes();o=int(sys.argv[1]);print(json.dumps({'b64':base64.b64encode(b[o:o+120000]).decode()}))",[str(offset)],max_bytes=170000)
                    raw.extend(base64.b64decode(piece['b64']))
                if hashlib.sha256(raw).hexdigest()!=export['sha256']:raise RuntimeError('Harness evidence checksum mismatch')
                output.with_suffix('.events.jsonl.gz').write_bytes(raw)
                data['export']=export;data['export']['local_path']=str(output.with_suffix('.events.jsonl.gz'));save()
            except Exception as exc:data['export_error']=str(exc);save()
        client.process.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--fixture-sha256',required=True);p.add_argument('--kp',required=True,type=float)
    p.add_argument('--harness-supported',action='store_true',required=True);p.add_argument('--workspace-clear',action='store_true',required=True)
    p.add_argument('--readiness',required=True)
    a=p.parse_args();r=run(a.output,a.fixture_sha256,a.kp,harness_supported=a.harness_supported,
                         workspace_clear=a.workspace_clear,readiness=a.readiness)
    print(json.dumps({k:r.get(k) for k in ('state','metrics','error','export','export_error')},indent=2))
