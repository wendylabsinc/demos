"""Run exactly one authorized harness experiment and verify/export its evidence.

No motion by default. --actuate requires all three physical setup attestations.
An uncertain start acknowledgement is never retried. Each output is reserved
before starting and cannot be reused. The existing recorder remains independent.
"""
import argparse
import base64
import hashlib
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from operator_panel import Wendy, DEPLOYING
from deployment_version import source_hashes
from offline.run_harness_replay import exec_json, STATUS
from harness_calibration import validate_request


def export_artifact(client, artifact, output):
    """Transfer sealed evidence in bounded chunks through native Wendy."""
    manifest=exec_json(client,"import hashlib,json,pathlib,sys,tarfile; p=pathlib.Path(sys.argv[1]).resolve(); assert p.parent==pathlib.Path('/data/calibration-trials'); a=p.with_suffix('.tar.gz'); t=tarfile.open(a,mode='w:gz'); [t.add(x,arcname=x.name) for x in sorted(p.iterdir()) if x.is_file() and x.name!='model.mjb']; t.close(); raw=a.read_bytes(); m=p/'model.mjb'; r=json.loads((p/'result.json').read_text()); print(json.dumps({'path':str(a),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'model_sha256':r['planning']['model_sha256'],'model_bytes':m.stat().st_size,'trial_id':p.name}))",[artifact])
    part=Path(output).with_suffix('.tar.gz.part')
    with part.open('wb') as destination:
        for offset in range(0,manifest['bytes'],48000):
            chunk=exec_json(client,"import base64,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.parent==pathlib.Path('/data/calibration-trials') and p.suffixes[-2:]==['.tar','.gz']; f=p.open('rb'); f.seek(int(sys.argv[2])); print(json.dumps({'base64':base64.b64encode(f.read(48000)).decode()}))",[manifest['path'],str(offset)],max_bytes=100000)
            destination.write(base64.b64decode(chunk['base64'],validate=True))
    raw=part.read_bytes()
    if len(raw)!=manifest['bytes'] or hashlib.sha256(raw).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Calibration artifact checksum mismatch')
    archive=Path(output).with_suffix('.tar.gz');part.replace(archive)
    directory=Path(output).with_suffix('.artifacts');directory.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
        if any(Path(m.name).name!=m.name or not m.isfile() for m in t.getmembers()):
            raise RuntimeError('Unexpected artifact member')
        t.extractall(directory,filter='data')
    model=Path(output).parent/'models'/f"{manifest['model_sha256']}.mjb"
    model.parent.mkdir(exist_ok=True)
    if not model.is_file():
        partial=model.with_suffix('.part')
        deadline=time.monotonic()+120
        with urllib.request.urlopen(f"http://10.10.20.238:8102/calibration-artifacts/{manifest['trial_id']}/model.mjb",timeout=10) as response, partial.open('wb') as destination:
            while chunk:=response.read(131072):
                if time.monotonic()>deadline:raise TimeoutError('Model transfer deadline exceeded')
                destination.write(chunk)
        if partial.stat().st_size!=manifest['model_bytes'] or hashlib.sha256(partial.read_bytes()).hexdigest()!=manifest['model_sha256']:
            raise RuntimeError('Compiled model checksum mismatch')
        partial.replace(model)
    elif hashlib.sha256(model.read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise RuntimeError('Cached model checksum mismatch')
    return {**manifest,'archive':str(archive),'directory':str(directory),'model_local':str(model)}


def recover_export(output):
    """Recover evidence from a finished attempt; never starts or stops motion."""
    output=Path(output)
    result=json.loads(output.read_text())
    client=Wendy()
    try:
        result['export']=export_artifact(client,result['calibration']['artifact'],output)
        result.pop('export_error',None)
        output.write_text(json.dumps(result,indent=2)+'\n')
        return result['export']
    finally:
        client.process.terminate()


def upload_fixture(client, local, request):
    raw=Path(local).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=request['fixture_sha256']:
        raise RuntimeError('Local replay fixture changed')
    encoded=base64.b64encode(gzip.compress(raw)).decode()
    for offset in range(0,len(encoded),12000):
        exec_json(client,"import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); assert p.parent==pathlib.Path('/data/joint-replays'); p.parent.mkdir(exist_ok=True); f=p.with_suffix('.upload').open('w' if sys.argv[3]=='0' else 'a'); f.write(sys.argv[2]); f.close(); print('{}')",[request['fixture'],encoded[offset:offset+12000],str(offset)])
    result=exec_json(client,"import base64,gzip,hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); u=p.with_suffix('.upload'); raw=gzip.decompress(base64.b64decode(u.read_text(),validate=True)); digest=hashlib.sha256(raw).hexdigest(); assert digest==sys.argv[2]; t=p.with_suffix('.part'); t.write_bytes(raw); t.replace(p); u.unlink(); print(json.dumps({'sha256':digest,'bytes':len(raw)}))",[request['fixture'],request['fixture_sha256']])
    return result


def run(output, request, *, actuate=False, replay_file=None):
    validate_request(request)
    if not actuate:
        return {"motion_commands_sent":0,"request":request,"next_step":"Use --actuate with the confirmed harness setup to run one trial"}
    if DEPLOYING.exists():
        raise RuntimeError("Deployment in progress")
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    result={"state":"reserved","request":request,"physical_pickup_verified":False}
    with output.open("x") as f:json.dump(result,f)
    def save():output.write_text(json.dumps(result,indent=2)+"\n")
    client=Wendy()
    started=False
    terminal=False
    try:
        expected=source_hashes(ROOT)
        expected={k:v for k,v in expected.items() if k.endswith('.py')}
        actual=exec_json(client,"import hashlib,json,pathlib,sys; e=json.loads(sys.argv[1]); print(json.dumps({n:hashlib.sha256((pathlib.Path('/app')/n).read_bytes()).hexdigest() for n in e}))",[json.dumps(expected)])
        if actual!=expected:raise RuntimeError("Deployed Python differs from local calibration source")
        before=exec_json(client,STATUS)
        result["before"]=before
        state=before["status"]
        if state["adapter_initialized"] or state["demo_running"] or not state["telemetry_healthy"]:
            raise RuntimeError("Trial requires released idle controller and healthy telemetry")
        if replay_file:
            result['uploaded_fixture']=upload_fixture(client,replay_file,request)
        result["state"]="start_requested";save();started=True
        try:
            result["acknowledgement"]=exec_json(client,"import json,sys,urllib.request; r=urllib.request.Request('http://127.0.0.1:8102/joint-replay/start',data=sys.argv[1].encode(),headers={'Content-Type':'application/json'},method='POST'); print(json.dumps(json.load(urllib.request.urlopen(r,timeout=10))))",[json.dumps(request)])
        except Exception as e:
            result["acknowledgement_error"]=str(e)
        save()
        deadline=time.monotonic()+(130 if request.get("kind")=="demo_replay" else 50)
        last=None
        while True:
            result["after"]=exec_json(client,STATUS)
            state=result["after"]["status"]
            save()
            if state['phase']!=last:
                last=state['phase'];print(last,flush=True)
            if not state['demo_running']:break
            if time.monotonic()>deadline:
                client.request(action='stop')
                raise RuntimeError('Calibration exceeded its bounded deadline; Stop requested')
            time.sleep(.5)
        terminal=True
        result["state"]="terminal"
        result["error"]=state.get('demo_error') or state.get('error')
        calibration=(state.get('demo_result') or {}).get('calibration')
        result["calibration"]=calibration
        if state['adapter_initialized'] or state['demo_running']:
            raise RuntimeError('Trial did not release control')
        result['control_released_verified']=True
        save()
        if calibration:
            try:
                result['export']=export_artifact(client,calibration['artifact'],output)
            except Exception as exc:
                # The terminal physical outcome and release proof survive a
                # transfer failure. Evidence remains on the recorder/device.
                result['export_error']=str(exc)
        save()
        print(json.dumps({'error':result['error'],'calibration':calibration,'export':result.get('export')},indent=2),flush=True)
        return result
    except Exception as exc:
        result['driver_error']=str(exc);save()
        # No blind activation retry. If the worker is still running request
        # cancellation; its calibration finally block owns supported release.
        if started and not terminal:
            try:client.request(action='stop')
            except Exception as stop_error:result['stop_error']=str(stop_error);save()
        raise
    finally:
        client.process.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--actuate',action='store_true')
    p.add_argument('--recover-export',action='store_true')
    p.add_argument('--harness-confirmed',action='store_true')
    p.add_argument('--workspace-clear',action='store_true')
    p.add_argument('--hands-empty',action='store_true')
    p.add_argument('--kp',type=float,default=50)
    p.add_argument('--kd',type=float,default=3)
    p.add_argument('--kind',choices=['joint','ik','demo_replay'],default='joint')
    p.add_argument('--replay-file')
    p.add_argument('--max-temperature-rise',type=float,default=8)
    p.add_argument('--joint',type=int,default=1)
    p.add_argument('--amplitude',type=float,default=.06)
    p.add_argument('--solver',default='baseline')
    p.add_argument('--offset',type=float,nargs=3,default=[.015,0,.015])
    a=p.parse_args()
    if a.recover_export:
        if a.actuate:p.error('Evidence recovery does not actuate')
        print(json.dumps(recover_export(a.output),indent=2))
        sys.exit(0)
    request=dict(experiment='calibration',kind=a.kind,kp=a.kp,kd=a.kd,joint=a.joint,
                 max_temperature_rise_c=a.max_temperature_rise,
                 amplitude_rad=a.amplitude,solver=a.solver,offset_m=a.offset,
                 harness_confirmed=a.harness_confirmed,workspace_clear_confirmed=a.workspace_clear,
                 hands_empty_confirmed=a.hands_empty)
    if a.kind=='demo_replay':
        if not a.replay_file:p.error('demo_replay requires --replay-file')
        digest=hashlib.sha256(Path(a.replay_file).read_bytes()).hexdigest()
        request.update(fixture=f'/data/joint-replays/{digest}.json',fixture_sha256=digest)
    print(json.dumps(run(a.output,request,actuate=a.actuate,replay_file=a.replay_file),indent=2))
