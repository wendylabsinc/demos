"""Cold-ownership Runtime.run replay with recorded observations and inert sinks.

Real prepare, IK, collision preflight, Runtime.run and publication guards run.
No DDS modules, camera endpoints, publishers, deployment or physical claims.
The trace ended during handover: settled feedback and later exposures do not
exist. The owned branch holds the terminal recorded scene constant and reports
that counterfactual explicitly; receipt times alone are rebased offline.
"""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from physical_core import PhysicalPlant, calibrated_scene
os.environ.setdefault('HANDOFF_DATA',tempfile.mkdtemp(prefix='tracking-replay-'))
from runtime import Runtime
from offline.planner_replay import clearance_auditor

BASE=Path(__file__).resolve().parent.parent
ATTEMPT='20261008T004012Z'
HISTORICAL=BASE/'offline/evidence/20261008T010351Z-85369a28/source/runtime.py'

class CheckedSinkReached(RuntimeError):pass

def run(historical=False, displacement_m=0.):
    paths=[BASE/'offline/trial-events.json.gz',BASE/f'offline/attempts/{ATTEMPT}.json']
    captured=json.loads(gzip.decompress(paths[0].read_bytes()))[ATTEMPT]['events']
    trial=json.loads(paths[1].read_text())
    observations=[e for e in captured if e['phase']=='perception_result' and e.get('geometry')]
    before=copy.deepcopy(trial['before']['status']['measured'])
    after=copy.deepcopy(trial['after']['status']['measured'])
    config=copy.deepcopy(next(e['calibration'] for e in captured if e['phase']=='scene_refreshed'))
    owner=next(e for e in captured if e['phase']=='ownership_acquired')
    staging=next(e for e in captured if e['phase']=='staging_planned')
    # Feedback at exact plan/replan instants was not captured. Use the actual
    # measured staging start and recorded held right arm rather than invented q.
    for snap in (before,after):
        snap['body']['q'][15:22]=staging['start_q']
        snap['body']['q'][22:29]=owner['right_arm_hold']
    before['body']['q'][:15]=config['reference_base_q']
    before['body']['imu_rpy']=config['reference_imu_rpy']
    import runtime as runtime_module
    runtime_class=Runtime
    if historical:
        spec=importlib.util.spec_from_file_location('historical_tracking_runtime',HISTORICAL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        runtime_class=module.Runtime;runtime_module=module
    runtime=runtime_class.__new__(runtime_class)
    runtime.active=False;runtime.phase='PASSIVE';runtime.arm=None;runtime.hand=None;runtime.plant=None
    runtime.cancel=threading.Event();runtime.lock=threading.RLock();runtime.observation_lock=threading.RLock()
    runtime.tracking_stop=threading.Event();runtime.tracking_thread=None;runtime.tracking_baseline=None
    runtime.tracking_observation=None;runtime.tracking_error=None;runtime.perception=None
    runtime.frame_prefetch=None;runtime.pending_frame=None;runtime.grasp_started=False
    runtime.first_failure=None;runtime.motion_commands_sent=0;runtime.hub=None
    runtime.camera=Mock(token='offline-no-camera')
    logged=[];plants=[];clearances=[];baselines=[];sinks=[]
    state=dict(owned=False,settled=False,replanned=False)
    def read():
        result=copy.deepcopy(after if state['owned'] else before)
        for sample in result.values():sample['time']=time.monotonic()
        return result
    runtime.read=read
    runtime.archive_guard=lambda:dict(age_s={'rt/arm_sdk':None},session=owner['recorder_session'])
    def event(phase,**fields):
        logged.append(dict(phase=phase,**fields))
        if phase=='ownership_acquired':
            state['owned']=True
            # Historical worker remained live during ownership; the captured
            # terminal observation reproduces its exact >10 mm failure.
            if runtime.tracking_baseline is not None:
                sample,_=observe({},'handover-terminal',True)
                runtime.publish_tracking_observation(sample,observations[-1]['geometry'],True)
    runtime.event=event
    def refresh(**kwargs):
        result=copy.deepcopy(config)
        result['accepted_at_unix_s']=time.time()
        result['scene_refresh_automatic']=False
        if state['owned']:
            result['reference_imu_rpy']=after['body']['imu_rpy']
            result['reference_base_q']=after['body']['q'][:15]
        return result
    runtime.config=refresh
    def observe(config,phase,require_table,expected_center_xy=None):
        row=observations[-1] if state['owned'] else observations[0]
        center=np.r_[row['geometry']['center_xy_m'],config.get('table_height_m',.762)+.061]
        # Perturb only after the replacement path has been checked, so this
        # proves a genuine new displacement blocks the first pickup sink.
        if state['replanned']:center[0]+=displacement_m
        sample=SimpleNamespace(center=center,table_height=.762,timestamp=time.monotonic(),fit_error=row['geometry']['fit_error_m'])
        vision=copy.deepcopy(row['geometry'])
        runtime.publish_tracking_observation(sample,vision,require_table)
        return sample,vision
    runtime.observe_frame=observe
    def start(config,baseline):
        runtime.tracking_baseline=baseline;runtime.tracking_error=None;runtime.tracking_observation=None
        if state['owned']:state['replanned']=True
        baselines.append(baseline.center.tolist())
        observe(config,'motion-tracking',True);runtime.tracking_guard()
    runtime.start_tracking=start
    def make(config):
        plant=PhysicalPlant(calibrated_scene(config),runtime.arm,runtime.hand,read,config,event,cancel=runtime.cancel)
        clearances.append(clearance_auditor(plant));plants.append(plant)
        def hand(goal,duration):
            plant.validate_start();plant.check()
            sinks.append(dict(kind='first_pickup_hand_command',goal=goal,accepted=True))
            raise CheckedSinkReached('checked offline sink reached; no actuation')
        plant.hand=hand
        # Exception cleanup is inert; no emergency hold publisher exists.
        plant.freeze=lambda:None
        return plant
    runtime.make_plant=make
    def settled():
        state['settled']=True
        event('offline_settle_boundary',measured_settling_verified=False)
    runtime.wait_for_control_settle=settled
    arm=Mock();hand=Mock()
    modules={'g1_arm':SimpleNamespace(G1_29_ArmController=lambda **kw:arm),
             'g1_dex3':SimpleNamespace(Dex3Sender=lambda **kw:hand)}
    started=time.monotonic();error=None;reached=False
    with tempfile.TemporaryDirectory() as directory,patch.dict('sys.modules',modules),\
         patch.dict('os.environ',HANDOFF_PERCEPTION_POLICY='continuous'),\
         patch('perception_worker.PerceptionWorker',return_value=Mock()),\
         patch.object(runtime_module,'http_json',side_effect=ConnectionError),\
         patch.object(runtime_module,'DATA',Path(directory)):
        try:runtime.run(True)
        except CheckedSinkReached:reached=True
        except (RuntimeError,ValueError) as exc:error=str(exc)
    return dict(historical_sequence=historical,checked_pickup_sink_reached=reached,error=error,
        displacement_m=displacement_m,baselines=baselines,clearance=clearances,events=logged,
        recorded_command_sinks=sinks,elapsed_s=time.monotonic()-started,
        native_shift_m=float(np.linalg.norm(np.asarray(observations[-1]['geometry']['center_xy_m'])-
                                           observations[0]['geometry']['center_xy_m'])),
        actuation=False,physical_lift_verified=False,measured_settling_verified=False,
        missing_fields=['feedback at exact planning/replanning instant','post-settle RGB-D exposures',
                        'post-settle measured feedback'],
        feedback_source='captured before/terminal snapshots with recorded staging start and held right arm',
        receipt_times_rebased_offline=True,
        counterfactual='terminal recorded scene held stationary after synthetic settle boundary',
        input_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        historical_source_sha256=hashlib.sha256(HISTORICAL.read_bytes()).hexdigest())

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeat',type=int,default=3);args=parser.parse_args()
    results=[run(old,shift) for _ in range(args.repeat) for old,shift in ((True,0.),(False,0.),(False,.025))]
    passed=all((r['error']=='Tracked can moved outside the checked pickup path' and not r['recorded_command_sinks'])
               if r['historical_sequence'] or r['displacement_m'] else r['checked_pickup_sink_reached'] for r in results)
    print(json.dumps(dict(acceptance_passed=passed,results=results),indent=2))
    raise SystemExit(0 if passed else 1)
