"""Compare captured command targets and reconstructed IK without hardware.

Command replay uses the actual current interpolation function, isolated from
collision admission. IK retains collision checks but uses explicitly reconstructed
input state. Neither comparison measures DDS packet timing or physical tracking.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from physical_core import PhysicalPlant, calibrated_scene

ROOT=Path(__file__).resolve().parents[1]


def subject(name, rows, inputs):
    config=next(e['calibration'] for e in rows if e['phase']=='scene_refreshed')
    center=next(e['center_m'] for e in rows if e['phase']=='observed_can_collision_proxy')
    path=ROOT/'offline/attempts'/f'{name}.json'
    inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot=copy.deepcopy(json.loads(path.read_text())['before']['status']['measured'])
    snapshot['body']['q'][:15]=config['reference_base_q']
    snapshot['body']['imu_rpy']=config['reference_imu_rpy']
    for value in snapshot.values():value['time']=100.
    plant=PhysicalPlant(calibrated_scene(config,x=center[0],y=center[1]),None,None,
                        lambda:copy.deepcopy(snapshot),config,lambda *a,**k:None,
                        clock=SimpleNamespace(monotonic=lambda:100.))
    plant.observe_can(SimpleNamespace(center=center))
    return plant,snapshot


def compare():
    source=ROOT/'offline/trial-events.json.gz'
    captures=json.loads(gzip.decompress(source.read_bytes()))
    inputs={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()}
    segments=[];ik_results=[]
    for name,blob in captures.items():
        rows=blob['events']
        if not any(e['phase']=='target' for e in rows):continue
        plant,snapshot=subject(name,rows,inputs)
        for index,event in enumerate(rows):
            if event['phase']!='staging_segment':continue
            tail=rows[index+1:]
            end=next((i for i,e in enumerate(tail) if e['phase'] in
                     ('staging_segment','approach','align','reach','lift','primary_failure')),len(tail))
            samples=np.asarray([e['arm'][:7] for e in tail[:end] if e['phase']=='target'])
            if not len(samples):continue
            plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]]=samples[0]
            # This isolated comparison asks whether the commanded curve is
            # the same. It does not re-admit a historical trajectory as safe.
            with patch.object(plant.ik,'collision_details',return_value=None):
                _,poses=plant._joint_poses(event['goal_q'])
            generated=np.asarray(poses)
            count=min(len(samples),len(generated))
            delta=samples[:count]-generated[:count]
            segments.append(dict(trial=name,segment=event['index'],recorded_samples=len(samples),
                generated_samples=len(generated),compared_samples=count,
                all_recorded_samples_covered=len(samples)<=len(generated),
                max_abs_error_rad=float(np.max(np.abs(delta))),rms_error_rad=float(np.sqrt(np.mean(delta**2))),
                collision_admission_tested=False,first_sample_used_as_execution_start=True))
        if name not in ('20261008T001246Z','20261008T003513Z'):continue
        plan=next(e for e in rows if e['phase']=='plan')
        staging=next(e for e in rows if e['phase']=='staging_planned')
        snapshot['body']['q'][15:22]=staging['start_q']
        plant.sync(snapshot);plant.set_contact_phase('approach')
        solved=[plant.ik.solve(np.asarray(plan['waypoints'][0]),np.asarray(plan['rotation'])) for _ in range(2)]
        recorded=np.asarray(staging['joint_waypoints'][-1])
        ik_results.append(dict(trial=name,target_m=plan['waypoints'][0],current_joint_goal_rad=solved[0].tolist(),
            recorded_joint_goal_rad=recorded.tolist(),repeat_max_abs_error_rad=float(np.max(np.abs(solved[0]-solved[1]))),
            vs_recorded_max_abs_error_rad=float(np.max(np.abs(solved[0]-recorded))),
            full_planning_snapshot_available=False,
            reconstruction='recorded staging start and calibration; other joints from before-trial snapshot'))
    return dict(hardware_used=False,physical_trajectory_verified=False,scope='Logged left-arm command targets and two reconstructed approach IK endpoints',
        all_logged_targets_match=all(s['all_recorded_samples_covered'] and s['max_abs_error_rad']<1e-12 for s in segments),
        compared_segments=len(segments),compared_samples=sum(s['compared_samples'] for s in segments),
        segments=segments,ik=ik_results,input_sha256=inputs,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                      ('physical_core.py','g1_can_handoff/sim.py','offline/trajectory_parity.py')},
        limitations=['Recorded IK goals are inputs to command-curve replay, not regenerated by that test.',
                     'Complete model and feedback state at planning time were not captured for these two IK comparisons.',
                     'Current solver/model may differ from the historical version.',
                     'No exact DDS packet cadence, feedforward torque, adapter stream or measured physical path equivalence claim.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=compare()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('segments','input_sha256','source_sha256')},indent=2))
    raise SystemExit(0 if result['all_logged_targets_match'] else 1)
