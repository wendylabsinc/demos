"""Run one authorized demo attempt and retain its native controller evidence.

Run locally with --start to issue one pickup request. Loading the module does
not connect or start motion. Stops on timeout; never releases or restarts a
controller. Every retry requires another invocation after reviewing evidence.
"""
import argparse
import datetime
import json
from pathlib import Path
import time

from operator_panel import Wendy


def payload(result):
    row=result.get('structuredContent')
    if row is None:row=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
    if row.get('exit_code')!=0 or row.get('truncated'):
        raise RuntimeError(row.get('stderr') or 'Incomplete native controller evidence')
    return json.loads(row['stdout'])


def snapshot(client,since_ns):
    script="""import collections,json,pathlib,urllib.request
s=json.load(urllib.request.urlopen('http://127.0.0.1:8102/status',timeout=5))
events=[json.loads(x) for x in pathlib.Path(s['events']).read_text().splitlines()]
events=[e for e in events if e['unix_ns']>=SINCE]
frequent={'target','hand_target','phase_camera_evidence','perception_result'}
print(json.dumps({'status':s,'native_event_file':s['events'],
 'event_counts':dict(collections.Counter(e['phase'] for e in events)),
 'events':[e for e in events if e['phase'] not in frequent],
 'perception_tail':[e for e in events if e['phase'] in {'phase_camera_evidence','perception_result'}][-20:]}))
""".replace('SINCE',str(since_ns))
    return payload(client.tool('container_exec',{'app_name':'DEV-g1-handoff-physical',
        'command':['python3','-c',script],'timeout_seconds':15,'max_bytes':1000000}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true',help='Explicitly issue one authorized pickup request')
    parser.add_argument('--resume-pregrasp',action='store_true',help='Allow an explicitly reviewed FROZEN hold; controller still rejects any begun grasp')
    parser.add_argument('--output',type=Path,default=Path(__file__).parents[2]/'.runtime/demo-attempts')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    started_ns=time.time_ns();name=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    evidence={'started_unix_ns':started_ns,'device':'10.10.20.238:50052','start_requested':args.start}
    client=Wendy();accepted=False;finished=False
    try:
        evidence['before']=snapshot(client,started_ns)
        before=evidence['before']['status']
        if args.start:
            resume=args.resume_pregrasp and before.get('phase')=='FROZEN'
            if before['demo_running'] or (before['adapter_initialized'] and not resume):
                raise RuntimeError('Existing demo or arm ownership requires operator review before another attempt')
            # A failed response does not prove the command was unaccepted.
            accepted=True
            evidence['start_response']=client.request(action='start')
            deadline=time.monotonic()+125
            while time.monotonic()<deadline:
                state=client.request()
                if not state['demo_running']:finished=True;break
                time.sleep(1)
            if not finished:
                evidence['timeout_stop']=client.request(action='stop')
                raise TimeoutError('Demo exceeded 125 seconds; Stop and hold requested')
    except Exception as exc:
        evidence['runner_error']=f'{type(exc).__name__}: {exc}'
        if accepted and not finished:
            # Request cancellation after a transport error, then inspect. Do
            # not replay Start when its outcome is uncertain.
            try:evidence['error_stop']=client.request(action='stop')
            except Exception as stop_error:evidence['stop_error']=str(stop_error)
    finally:
        try:evidence['after']=snapshot(client,started_ns)
        except Exception as exc:evidence['snapshot_error']=str(exc)
        client.process.terminate()
        evidence['finished_unix_ns']=time.time_ns()
        path=args.output/(name+'.json');path.write_text(json.dumps(evidence,indent=2)+'\n')
        state=evidence.get('after',{}).get('status',{})
        summary={'evidence':str(path),'phase':state.get('phase'),'error':(state.get('first_failure') or {}).get('error') or state.get('demo_error') or state.get('error'),
                 'physical_lift_verified':(state.get('demo_result') or {}).get('physical_lift_verified',False),
                 'motion_commands_sent':state.get('motion_commands_sent'),'adapter_initialized':state.get('adapter_initialized'),
                 'runner_error':evidence.get('runner_error')}
        with (args.output/'attempts.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        print(json.dumps(summary,indent=2))
    if evidence.get('runner_error') or (args.start and not summary['physical_lift_verified']):raise SystemExit(1)


if __name__=='__main__':main()
