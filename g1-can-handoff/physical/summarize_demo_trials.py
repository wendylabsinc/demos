"""Read-only local trial summary; does not connect to the robot."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import tarfile


def summarize(directory):
    trials=[]
    for path in sorted(directory.glob('trial-*.json')):
        row=json.loads(path.read_text())
        result=dict(index=row['index'],state=row['state'],error=row.get('error'),
                    primary_failure=row.get('primary_failure'),
                    physical_lift_verified=row.get('physical_lift_verified',False),
                    reset_confirmed_after=row.get('reset_confirmed_after',False),
                    artifact_verified=row.get('artifact',{}).get('verified',False),
                    recorder_session=row.get('recorder_before',{}).get('session'))
        artifact=path.with_suffix('.tar.gz')
        if row['state']=='terminal' and result['artifact_verified'] and artifact.is_file():
            with tarfile.open(artifact) as tar:
                events=[json.loads(line) for line in gzip.decompress(tar.extractfile('events.jsonl.gz').read()).splitlines()]
            ages=[e['exposure_age_ms'] for e in events if e['phase']=='perception_result'
                  and e.get('geometry') and e.get('exposure_age_ms') is not None]
            result['events']=len(events)
            result['perception']=dict(accepted=len(ages),exposure_age_ms=dict(
                min=min(ages),median=statistics.median(ages),max=max(ages))) if ages else dict(accepted=0)
        trials.append(result)
    finished=[t for t in trials if t['state']=='terminal']
    return dict(requested=10,completed=len(finished),remaining=10-len(finished),
                successful_lifts=sum(t['physical_lift_verified'] for t in finished),
                errors=dict(Counter(t['error'] for t in finished if t['error'])),trials=trials)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();report=summarize(args.directory)
    (args.directory/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
