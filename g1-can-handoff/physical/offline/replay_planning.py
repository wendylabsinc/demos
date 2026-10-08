"""Replay captured planner decisions with strict model/source/version checks."""
import argparse
import gc
import json
from pathlib import Path
from planning_snapshot import replay


def run(directory,kinds=('ik','continuous_ik'),output=None):
    rows=[]
    for path in sorted(Path(directory).glob('*.json')):
        snapshot=json.loads(path.read_text())
        if snapshot.get('kind') not in kinds:continue
        rows.append(replay(path));gc.collect()
    report=dict(hardware_used=False,decisions=len(rows),all_identical=bool(rows) and all(r['identical'] for r in rows),results=rows)
    if output:Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--kind',action='append')
    parser.add_argument('--output');args=parser.parse_args()
    report=run(args.directory,args.kind or ('ik','continuous_ik'),args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))
    raise SystemExit(0 if report['all_identical'] else 1)
