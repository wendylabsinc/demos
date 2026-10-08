#!/usr/bin/env python3
"""Inspect recorded exposure latency and first failures without a live service."""
import argparse
import gzip
import json
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--attempt',help='E.g. 20261008T004101Z; omit for all archived attempts')
parser.add_argument('--events',action='store_true',help='Print full ordered events instead of summary')
args=parser.parse_args()
trials=json.loads(gzip.decompress((Path(__file__).parent/'trial-events.json.gz').read_bytes()))
for path in (Path(__file__).parent/'evidence').glob('*/events.jsonl.gz'):
    trials[path.parent.name]={'events':[json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]}
if args.attempt:trials={args.attempt:trials[args.attempt]}
rows=[]
for name,trial in trials.items():
    events=trial['events']
    if args.events:rows.append(dict(attempt=name,events=events));continue
    observations=[e for e in events if e['phase']=='perception_result']
    ages=sorted(e['exposure_age_ms'] for e in observations if e.get('exposure_age_ms') is not None)
    failure=next((e for e in events if e['phase']=='primary_failure'),None)
    rows.append(dict(attempt=name,first_failure=failure,observations=len(observations),
        median_exposure_age_ms=statistics.median(ages) if ages else None,
        p95_exposure_age_ms=ages[min(len(ages)-1,int(len(ages)*.95))] if ages else None,
        maximum_exposure_age_ms=max(ages) if ages else None,
        exposure_limit_ms=400,hardware_used=False))
print(json.dumps(rows,indent=2))
