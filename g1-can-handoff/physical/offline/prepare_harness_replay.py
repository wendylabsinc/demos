"""Extract the recorded reach and preflight it with a passive robot snapshot."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from physical_core import PhysicalPlant,calibrated_scene
from harness_replay import harness_config,validate_fixture,preflight
from replay_arm_tracking import load


def prepare(trial,seed,output):
    _,commands,_=load(trial)
    fixture=dict(schema='wendy.g1.harness-reach-replay.v1',period_s=.02,
        source_trial_sha256=hashlib.sha256(Path(trial).read_bytes()).hexdigest(),
        left_q=[row['arm'][:7] for row in commands],
        left_tau=[row['gravity_feedforward_nm'][:7] for row in commands])
    seed=json.loads(Path(seed).read_text());snapshot=copy.deepcopy(seed['measured'])
    # These are recorded samples, not advancing live feedback. Synthetic
    # freshness is only for this explicitly offline collision calculation.
    def read():
        for row in snapshot.values():row['time']=time.monotonic()
        return snapshot
    config=harness_config(seed['calibration'],snapshot)
    targets,_=validate_fixture(fixture,config['arm_limits_rad'])
    events=[]
    plant=PhysicalPlant(calibrated_scene(config,x=10.,y=-10.),None,None,read,config,
                        lambda event,**fields:events.append(dict(event=event,**fields)))
    preflight(plant,targets)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(fixture,separators=(',',':'))+'\n').encode();output.write_bytes(raw)
    report=dict(offline_only=True,hardware_commands_sent=0,commands=len(targets),
        fixture_sha256=hashlib.sha256(raw).hexdigest(),collision_preflight_passed=True,
        workspace_absence_assumption=True,operator_confirmation_required_for_motion=True,
        seed_captured_unix_s=seed['captured_unix_s'],events=events)
    output.with_suffix('.preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='events'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',required=True);p.add_argument('--seed',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.trial,a.seed,a.output)
