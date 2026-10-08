"""Replay one saved physical reach's exact commands in supported-base dynamics.

Both gains receive the same targets, feedforward and measured initial state.
No camera, DDS, physical controller or hardware action is constructed here.
This isolates arm following; it is not a perception or pickup validation.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import mujoco
import numpy as np
from physical_core import PhysicalPlant
from sim_runtime import setup_scene,SimulationIO,apply_measured_state


def load(trial,phase='reach'):
    with tarfile.open(trial) as tar:
        rows=[json.loads(line) for line in gzip.decompress(tar.extractfile('events.jsonl.gz').read()).splitlines()]
        decisions=[]
        for member in tar.getmembers():
            if member.name.startswith('planning/') and member.name.endswith('-move_poses.json'):
                snapshot=json.load(tar.extractfile(member))
                if snapshot['collision_phase']==phase and 'result' in snapshot:decisions.append(snapshot)
    if len(decisions)!=1:raise ValueError('Expected one completed movement decision for the requested phase')
    commands=[r for r in rows if r['phase']=='target' and r.get('phase_name')==phase]
    if not commands:raise ValueError('Recorded phase contains no arm commands')
    return decisions[0],commands,next((r for r in rows if r['phase']=='arm_settle_failed'),None)


def replay(snapshot,commands,kp):
    config=copy.deepcopy(snapshot['config'])
    # The recorded model proxy is camera-measured; no can truth is supplied.
    can_adr=None
    # Compile the same calibrated robot geometry, then restore the full FK state.
    from physical_core import calibrated_scene
    reference=calibrated_scene(config)
    can_adr=reference.model.jnt_qposadr[reference.model.joint('can_free').id]
    center=snapshot['qpos'][can_adr:can_adr+3]
    setup=dict(config=config,measured=snapshot['measured'],can_center_world_m=center,
               stationary_lower_body_support=True,
               camera_intrinsics=dict(width=640,height=480,fx=603.59765625,fy=603.7045288085938,
                                      ppx=318.05902099609375,ppy=261.0997619628906,coeffs=[0.]*5))
    world=setup_scene(setup);io=SimulationIO(world,arm_kp=kp)
    apply_measured_state(world,io,snapshot['measured'])
    if world.model.nq!=len(snapshot['qpos']):raise ValueError('Recorded joint model no longer matches replay model')
    world.data.qpos[:]=snapshot['qpos']
    world.model.body_pos[:reference.model.nbody]=snapshot['model_body_pos']
    world.model.body_quat[:reference.model.nbody]=snapshot['model_body_quat']
    # Preserve measured velocities, not the FK model's deliberately-zero qvel.
    world.targets[:]=world.data.qpos[world.qadr]
    mujoco.mj_forward(world.model,world.data)
    fk=calibrated_scene(config)
    plant=PhysicalPlant(fk,io,io,io.read,config,lambda *a,**k:None,clock=io)
    plant.observe_can(type('Observation',(),{'center':center})())
    plant.set_contact_phase(snapshot['collision_phase'])
    samples=[];collision=None
    left_hand=np.asarray(snapshot['held_left_hand']);right_hand=np.asarray(snapshot['held_right_hand'])
    for index,command in enumerate(commands):
        q=np.asarray(command['arm']);ff=np.asarray(command['gravity_feedforward_nm'])
        plant.sync(plant.check())
        try:plant.check_live_geometry(left_q=q[:7],hand_q=left_hand)
        except (RuntimeError,ValueError) as exc:collision=str(exc);break
        measured=np.asarray(io.read()['body']['q'][15:29])
        error=measured-q
        samples.append(dict(time_s=float(world.data.time),target=q.tolist(),measured=measured.tolist(),error_rad=error.tolist()))
        if np.max(np.abs(error[:7]))>.15:collision='Tracking error exceeds 0.15 rad';break
        io.ctrl_dual_arm(q,ff);io.ctrl_dual_hand(left_hand,right_hand);io.sleep(.02)
    goal=np.asarray(commands[-1]['arm']);feedforward=np.asarray(commands[-1]['gravity_feedforward_nm'])
    if collision is None:
        for _ in range(100):
            plant.sync(plant.check())
            try:plant.check_live_geometry(left_q=goal[:7],hand_q=left_hand)
            except (RuntimeError,ValueError) as exc:collision=str(exc);break
            io.ctrl_dual_arm(goal,feedforward);io.ctrl_dual_hand(left_hand,right_hand);io.sleep(.02)
    final=np.asarray(io.read()['body']['q'][15:29])-goal
    errors=np.asarray([s['error_rad'][:7] for s in samples])
    return dict(kp=kp,kd=3.,wrist_kp=40.,wrist_kd=1.5,commands_replayed=len(samples),
                recorded_commands=len(commands),error=collision,
                trajectory_max_abs_error_rad=float(np.max(np.abs(errors))) if len(errors) else None,
                trajectory_rms_error_rad=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                shoulder_roll_rms_error_rad=float(np.sqrt(np.mean(errors[:,1]**2))) if len(errors) else None,
                final_error_rad=final.tolist(),final_max_abs_error_rad=float(np.max(np.abs(final[:7]))),
                final_shoulder_roll_error_rad=float(abs(final[1])),final_settle_pass=bool(collision is None and np.max(np.abs(final[:7]))<.04),
                simulation_only=True,physical_tracking_verified=False,samples=samples)


def compare(trial,output,gains=(80.,100.)):
    snapshot,commands,physical=load(trial)
    results=[replay(snapshot,commands,kp) for kp in gains]
    if len(results)<2:raise ValueError('Compare requires at least two gains')
    old,new=results[0],results[-1]
    report=dict(schema='wendy.g1.arm-gain-replay.v1',simulation_only=True,hardware_used=False,
        trial=str(trial),trial_sha256=hashlib.sha256(Path(trial).read_bytes()).hexdigest(),
        same_recorded_targets_feedforward_and_initial_state=True,
        gains=[{k:v for k,v in r.items() if k!='samples'} for r in results],
        physical_baseline_settle_error_rad=None if physical is None else physical['error_rad'],
        final_error_reduction_percent=100*(old['final_max_abs_error_rad']-new['final_max_abs_error_rad'])/old['final_max_abs_error_rad'],
        limitations=['Supported-base dynamics; factory balance and real servo parameters remain approximate.',
                     'Requested 50 Hz targets are recorded; exact 100 Hz DDS cadence is not replayed.',
                     'Only the reach phase is replayed; camera/tracker checks are not exercised.'])
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    trace_path=Path(output).with_suffix('.traces.json.gz')
    trace=gzip.compress(json.dumps([{'kp':r['kp'],'samples':r['samples']} for r in results],separators=(',',':')).encode(),mtime=0)
    trace_path.write_bytes(trace)
    report['traces']={'file':trace_path.name,'sha256':hashlib.sha256(trace).hexdigest(),'bytes':len(trace)}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trial',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--kp',type=float,nargs='+',default=[80.,100.])
    args=parser.parse_args();report=compare(args.trial,args.output,args.kp)
    print(json.dumps(dict(final_error_reduction_percent=report['final_error_reduction_percent'],
        gains=[{k:v for k,v in r.items() if k!='samples'} for r in report['gains']]),indent=2))
