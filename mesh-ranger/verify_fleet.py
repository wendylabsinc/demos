"""Verify real inference and optionally exercise DNS-SD withdrawal/reannouncement."""
import argparse
import base64
import hashlib
from collections import Counter
import json
import time
import uuid
from urllib.request import ProxyHandler, Request, build_opener

HTTP = build_opener(ProxyHandler({}))


def request(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    with HTTP.open(Request(base + path, data=data, headers={'Content-Type':'application/json'}), timeout=25) as response:
        return json.load(response)


def until(read, accept, timeout=60):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=read()
        if accept(value): return value
        time.sleep(1)
    raise RuntimeError('Verification timed out')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a',default='http://127.0.0.1:19090')
    parser.add_argument('--b',default='http://127.0.0.1:19091')
    parser.add_argument('--exercise-discovery',action='store_true',help='Temporarily withdraw B, then restore it')
    parser.add_argument('--workers',type=int,default=2,help='Expected total workers, including A')
    parser.add_argument('--frame-event',help='Reuse a retained webcam frame from A for real vision checks')
    args=parser.parse_args()
    if args.workers < 2:
        parser.error('--workers must be at least two')
    a=request(args.a,'/api/state'); b=request(args.b,'/api/state')
    assert a['mode']==b['mode']=='inference', 'Simulation is not hardware evidence'
    assert any(p['node']==b['node'] and p['mesh_address'] for p in a['discovered']), 'A must discover B through Wendy Net'
    assert any(p['node']==a['node'] and p['mesh_address'] for p in b['discovered']), 'B must discover A through Wendy Net'
    expected={a['node'], *(p['node'] for p in a['discovered'] if p['mesh_address'])}
    assert len(expected)==args.workers, f'Expected {args.workers} workers, discovered {len(expected)}'
    frame=None
    frame_hash=None
    if args.frame_event:
        original=next((j for j in a['events'] if j['event']['event_id']==args.frame_event and j['has_image']),None)
        assert original, 'Frame event must exist in A history'
        frame=request(args.a,'/api/frames/'+args.frame_event)['image_b64']
        frame_hash=hashlib.sha256(base64.b64decode(frame,validate=True)).hexdigest()
    prefix='verify-'+uuid.uuid4().hex[:8]
    completed=[]
    # Complete one job at a time so busy-worker fallback cannot hide a fleet member.
    for i in range(args.workers*2):
        event_id=f'{prefix}-{i}'
        event={'event_id':event_id,'source':'fleet-check','zone':'north gate',
               'motion':True,'door_open':False,'temperature_c':22.0}
        if frame:
            event={'event_id':event_id,'source':'webcam-frame-recheck',
                   'zone':'Previously captured webcam frame','image_b64':frame}
        request(args.a,'/api/events',event)
        def job():
            return next((j for j in request(args.a,'/api/state')['events'] if j['event']['event_id']==event_id),None)
        result=until(job,lambda j:j and j['status'] in ('done','failed'),180)
        assert result['status']=='done', result
        completed.append(result)
    counts=Counter(j['result']['served_by'] for j in completed)
    assert expected <= set(counts), f'Missing worker results: {expected-set(counts)}'
    assert all(j['result']['mode']=='inference' and j['result']['gpu_bytes']>0 for j in completed)
    if frame:
        assert all(j['result']['model']=='moondream:1.8b' and j['result']['image_sha256']==frame_hash for j in completed)
    proof={'jobs':len(completed),'served_by':dict(counts),'frame_sha256':frame_hash,
           'results':[{k:j['result'].get(k) for k in ('event_id','served_by','endpoint','model','elapsed_ms','gpu_bytes','image_sha256')} for j in completed],
           'inference_ms':[j['result']['elapsed_ms'] for j in completed],
           'gpu_bytes':sorted(set(j['result']['gpu_bytes'] for j in completed)),
           'a_observes':a['discovered'],'b_observes':b['discovered']}
    if args.exercise_discovery:
        assert b['advertisement']['enabled'], 'B must start announced'
        try:
            request(args.b,'/api/advertisement',{'enabled':False})
            until(lambda:request(args.a,'/api/state'),lambda s:not any(p['node']==b['node'] for p in s['discovered']))
            proof['withdrawal']='B disappeared from A'
        finally:
            request(args.b,'/api/advertisement',{'enabled':True})
        until(lambda:request(args.a,'/api/state'),lambda s:any(p['node']==b['node'] for p in s['discovered']))
        proof['reannouncement']='B rediscovered by A'
    print(json.dumps(proof,indent=2))


if __name__=='__main__':
    main()
