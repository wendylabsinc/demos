"""Retrieve a terminal trial's sealed raw DDS recording. Never starts motion."""
import argparse
import json
from pathlib import Path

from operator_panel import Wendy
from trial_evidence import fetch_artifact,recorder_status,result_payload


def archive(client,evidence,destination):
    status=recorder_status(client)
    if status['session']!=evidence['recorder_after']['session']:
        raise RuntimeError('Recorder session changed; select the original session explicitly before exporting')
    source=(Path(__file__).parent/'native_recorder_export.py').read_text()
    script=source+"\nimport sys,urllib.request\ns=json.load(urllib.request.urlopen('http://127.0.0.1:8101/status',timeout=5))\nif s['session']!=sys.argv[1]:raise RuntimeError('Recorder session changed')\nprint(json.dumps(create_recording_export(s,int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])))\n"
    remote='/records/trial-exports/'+Path(destination).name
    metadata=json.loads(result_payload(client.tool('container_exec',dict(app_name='DEV-g1-handoff-recorder',
        command=['python3','-c',script,status['session'],str(evidence['started_unix_ns']),str(evidence['finished_unix_ns']),remote],
        timeout_seconds=60,max_bytes=10000)))['stdout'])
    return fetch_artifact(client,'DEV-g1-handoff-recorder',remote,metadata,destination,max_bytes=65000000)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('trial',type=Path);args=parser.parse_args()
    evidence=json.loads(args.trial.read_text());client=None
    try:
        client=Wendy()
        evidence['dds_artifact']=archive(client,evidence,args.trial.with_suffix('.dds.tar.gz'))
        evidence.pop('dds_artifact_error',None)
    except Exception as exc:
        evidence['dds_artifact_error']=str(exc);raise
    finally:
        if client is not None:client.process.terminate()
        args.trial.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence['dds_artifact'],indent=2))


if __name__=='__main__':main()
