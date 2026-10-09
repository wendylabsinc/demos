"""Replay captured pre-motion stops without hardware or invented feedback.

Numerical planning replay remains a separate strict model/source check. This
script replays the planner's deadline using recorded candidate runtimes and the
waist guard using the saved terminal status (not its unrecorded failing read).
"""
import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pose_alignment import corrected_config
from g1_can_handoff.perception import Observation
from g1_can_handoff.task import plan


def extract_archives(directory,destination=None):
    directory=Path(directory);destination=Path(destination or directory)
    destination.mkdir(parents=True,exist_ok=True)
    for name in ('trial-01','trial-02'):
        metadata=json.loads((directory/(name+'-transfer.json')).read_text())
        archive=directory/(name+'.tar.gz')
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=metadata['sha256']:
            raise ValueError('Changed native trial archive: '+name)
        with tarfile.open(archive) as tar:tar.extractall(destination/name,filter='data')
    (destination/'trial-01-status.json').write_bytes((directory/'trial-01-status.json').read_bytes())
    return destination


def events(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Changed trial evidence: '+name)
    root=Path(__file__).resolve().parents[1]
    for name in ('pose_alignment.py','g1_can_handoff/task.py'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=manifest['source_hashes'][name]:
            raise ValueError('Replay source differs from trial: '+name)
    return [json.loads(row) for row in gzip.decompress((directory/'events.jsonl.gz').read_bytes()).splitlines()]


def error_from(fn):
    try:fn()
    except (RuntimeError,ValueError) as exc:return f'{type(exc).__name__}: {exc}'
    return None


def run(directory):
    directory=Path(directory)
    first=events(directory/'trial-01');second=events(directory/'trial-02')
    config=next(row['calibration'] for row in first if row['phase']=='scene_refreshed')
    body=json.loads((directory/'trial-01-status.json').read_text())['end']['status']['measured']['body']
    pose=dict(timestamp=body['time'],rpy=body['imu_rpy'],waist_q=body['q'][12:15])
    waist_error=error_from(lambda:corrected_config(config,pose))
    expected_waist=next(row['error'] for row in first if row['phase']=='primary_failure')
    starts={row['path']:row for row in second if row['phase']=='planning_snapshot' and row['kind']=='preflight'}
    complete=[row for row in second if row['phase']=='planning_snapshot_complete' and row['kind']=='preflight']
    durations=[(row['monotonic_ns']-starts[row['path']]['monotonic_ns'])/1e9 for row in complete]
    snapshots=[json.loads((directory/'trial-02/planning'/Path(row['path']).name).read_text()) for row in complete]
    if len(snapshots)!=3 or any('error' in snapshot for snapshot in snapshots):
        raise ValueError('Expected three successful captured candidate validations')
    config=snapshots[0]['config'];clock={'time':100.};calls=[]
    plant=SimpleNamespace(grasp_candidate_budget=config['grasp_candidate_budget'],
        grasp_orientation_step_rad=config['grasp_orientation_step_rad'],
        set_contact_phase=lambda phase:None,grasp_noise_score=lambda:0.)
    center=np.asarray(snapshots[0]['request']['args'][0][2])
    observation=Observation(center,config['table_height_m'],100.,0.,100)
    def validate(points,rotation):
        index=len(calls)
        if index>=len(snapshots):raise AssertionError('Unexpected additional candidate')
        expected_points,expected_rotation=snapshots[index]['request']['args']
        np.testing.assert_array_equal(points,expected_points)
        np.testing.assert_array_equal(rotation,expected_rotation)
        calls.append(index);clock['time']+=durations[index]
        return True
    with patch('g1_can_handoff.task.time.monotonic',side_effect=lambda:clock['time']):
        deadline_error=error_from(lambda:plan(plant,observation,validate=validate,include_offer=False))
    expected_deadline=next(row['error'] for row in second if row['phase']=='primary_failure')
    return dict(hardware_used=False,all_errors_reproduced=waist_error==expected_waist and deadline_error==expected_deadline,
        waist=dict(error=waist_error,captured_error=expected_waist,
            max_delta_rad=float(np.max(np.abs(np.asarray(pose['waist_q'])-config_waist(first)))),
            scope='Saved terminal status reproduces guard; exact failing constructor read was not recorded.'),
        deadline=dict(error=deadline_error,captured_error=expected_deadline,
            candidate_seconds=durations,total_validation_seconds=sum(durations),budget_seconds=30.,
            candidate_inputs_identical=True,scope='Recorded validation outcomes and durations; numerical replay is separate.'),
        physical_pickup_verified=False)


def config_waist(rows):
    return np.asarray(next(row['calibration'] for row in rows if row['phase']=='scene_refreshed')['reference_base_q'][12:15])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory');parser.add_argument('--output')
    parser.add_argument('--extract',action='store_true',help='Verify and unpack preserved native archives first')
    args=parser.parse_args()
    if args.extract:extract_archives(args.directory)
    report=run(args.directory)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['all_errors_reproduced'] else 1)
