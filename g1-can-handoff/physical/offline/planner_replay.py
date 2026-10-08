"""Recorded G1 geometry replay. No network, DDS, command publishers or motion."""
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from physical_core import PhysicalPlant, calibrated_scene

ROOT=Path(__file__).parent/'evidence/ten-demos-20261008'


def fixture(index=1):
    directory=ROOT/f'trial-{index:02d}'
    trial=json.loads((directory/'trial.json').read_text())
    events=[json.loads(line) for line in gzip.decompress((directory/'events.jsonl.gz').read_bytes()).decode().splitlines()]
    config=next(e['calibration'] for e in reversed(events) if e['phase']=='scene_refreshed')
    center=next(e['center_m'] for e in reversed(events) if e['phase']=='observed_can_collision_proxy')
    snapshot=copy.deepcopy(trial['controller_after']['measured'])
    snapshot['body']['q'][:15]=config['reference_base_q']
    snapshot['body']['imu_rpy']=config['reference_imu_rpy']
    snapshot['body']['q'][15:22]=next(e['start_q'] for e in reversed(events) if e['phase']=='staging_planned')
    owner=next(e for e in reversed(events) if e['phase']=='ownership_acquired')
    snapshot['body']['q'][22:29]=owner['right_arm_hold']
    def read():
        result=copy.deepcopy(snapshot)
        for sample in result.values():sample['time']=100.
        return result
    logged=[]
    def event(phase,**data):logged.append(dict(phase=phase,**data))
    clock=SimpleNamespace(monotonic=lambda:100.)
    scene=calibrated_scene(config,x=center[0],y=center[1])
    plant=PhysicalPlant(scene,None,None,read,config,event,clock=clock)
    observation=SimpleNamespace(center=np.array(center),table_height=config['table_height_m'],timestamp=100.)
    plant.observe_can(observation)
    return plant,observation,events,logged


def clearance_auditor(plant):
    """Independently query geometry distances on returned command samples."""
    model=plant.ik.model;data=mujoco.MjData(model)
    table=model.geom('table').id
    geoms=[i for i in range(model.ngeom) if model.geom_contype[i] and
           model.body(model.geom_bodyid[i]).name.startswith('left_')]
    report=dict(samples=0,min_table_distance_m=1.,required_clearance_m=plant.ik.table_clearance_m)
    original=plant._joint_poses
    def checked(goal,duration=2.):
        before=plant.ik.data.qpos.copy()
        result=original(goal,duration)
        for q in result[1]:
            data.qpos[:]=before;data.qpos[plant.ik.qadr[plant.ik.arm]]=q
            mujoco.mj_forward(model,data)
            minimum=min(mujoco.mj_geomDistance(model,data,table,g,.1,None) for g in geoms)
            if minimum+1e-8<report['required_clearance_m']:
                raise AssertionError(f'Returned trajectory lacks table clearance: {minimum}')
            report['min_table_distance_m']=min(report['min_table_distance_m'],float(minimum))
            report['samples']+=1
        return result
    plant._joint_poses=checked
    original_event=plant.event
    def event(phase,**fields):
        if phase=='grasp_candidate_rejected':
            report.update(samples=0,min_table_distance_m=1.)
        original_event(phase,**fields)
    plant.event=event
    return report


if __name__=='__main__':
    import argparse
    import time
    from pickup import prepare
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trial',type=int,default=1,choices=(1,2,3))
    args=parser.parse_args()
    plant,observation,events,logged=fixture(args.trial);observation.timestamp=time.monotonic()
    clearance=clearance_auditor(plant)
    started=time.monotonic()
    try:
        _,_,points,rotation=prepare(plant,lambda *args:(observation,dict(source='saved_scene_replay')),plant.event)
        result=dict(passed=True,waypoints=[p.tolist() for p in points],rotation=rotation.tolist())
    except (ValueError,RuntimeError) as exc:result=dict(passed=False,error=str(exc))
    result.update(elapsed_s=time.monotonic()-started,
                  rejected=[e['reason'] for e in logged if e['phase']=='grasp_candidate_rejected'],
                  events=logged,clearance=clearance,trial=args.trial,actuation=False,physical_lift_verified=False,
                  provenance='Recorded accepted calibration and terminal feedback; base/arm replaced by recorded calibration/hold/staging values, receipt times rebased offline')
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
