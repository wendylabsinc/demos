"""Run the physical pickup code against a supported MuJoCo G1.

Only device I/O, clock and calibration provenance differ. The physical plant,
planner, visual geometry, obstruction proxy and pickup sequence are unchanged.
Simulation truth is recorded separately and never fed to grasp verification.
"""
import argparse
import json
import hashlib
import importlib.metadata
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from camera_input import geometry
from physical_core import BODY_NAMES, PhysicalPlant, SIM_TO_BRIDGE
from pickup import prepare, execute
from g1_can_handoff.sim import scene, FINGERS
from simulation_position_noise import PositionNoise
from stochastic_position_noise import StochasticPositionNoise


class SimulationIO:
    def __init__(self, world, force_zero_feedforward=False,*,arm_kp=None):
        self.world = world
        self.origin = time.monotonic()
        self.commands = 0
        self.imu_rpy = [0.,0.,0.]
        self.body_ids = np.array([world.model.joint(n).id for n in BODY_NAMES])
        self.hand_ids = {side: np.array([world.model.joint(n.replace('left_',side+'_')).id
                                        for n in FINGERS]) for side in ('left','right')}
        self.actuators = {n: world.model.actuator(n).id for n in BODY_NAMES[15:]}
        self.arm_ids=np.array(list(self.actuators.values()))
        self.all_hand_ids=np.array([i for i in range(world.model.nu) if 'hand' in world.model.actuator(i).name])
        from adapter.arm_gains import arm_gains
        kp,kd=arm_gains(arm_kp)
        self.arm_kp=np.tile(kp,2);self.arm_kd=np.tile(kd,2)
        self.arm_feedforward=np.zeros(14)
        self.force_zero_feedforward=force_zero_feedforward
        self.arm_reference=None
        self.wire_arm_q=None
        self.servo_ticks=0
        self.max_abs_feedforward=np.zeros(14)
        self.arm_limits=world.model.jnt_range[self.body_ids[15:]].copy()

    def monotonic(self):
        return self.origin+self.world.data.time

    def sleep(self, seconds):
        from adapter.arm_reference import ArmReference
        m,d=self.world.model,self.world.data
        for _ in range(round(max(0,seconds)/m.opt.timestep)):
            measured=d.qpos[self.world.qadr[self.arm_ids]]
            if self.arm_reference is None:
                self.arm_reference=ArmReference(measured)
                self.arm_reference.max_velocity=.25
                self.arm_reference.max_acceleration=1.
                self.wire_arm_q=measured.copy()
            # The real SDK writer advances ArmReference at G1_CONTROL_HZ=100.
            if self.servo_ticks%5==0:
                self.wire_arm_q=self.arm_reference.advance(self.world.targets[self.arm_ids],measured,.01,
                                                          limits=self.arm_limits)
            error=self.world.targets-d.qpos[self.world.qadr]
            velocity=d.qvel[self.world.dadr]
            effort=self.world.kp*error-self.world.kd*velocity+d.qfrc_bias[self.world.dadr]
            # Actual arm packets carry kp/kd, zero desired velocity and the
            # supplied second ctrl_dual_arm argument as feedforward torque.
            effort[self.arm_ids]=(self.arm_kp*(self.wire_arm_q-measured)
                                      -self.arm_kd*velocity[self.arm_ids]+self.arm_feedforward)
            # Dex3 packets have no model gravity feedforward either.
            effort[self.all_hand_ids]=(self.world.kp[self.all_hand_ids]*error[self.all_hand_ids]
                                       -self.world.kd[self.all_hand_ids]*velocity[self.all_hand_ids])
            d.ctrl[:]=np.clip(effort,-self.world.torque,self.world.torque)
            mujoco.mj_step(m,d)
            self.servo_ticks+=1
            if self.world.frame_callback:self.world.frame_callback(self.world)

    def read(self):
        m, d = self.world.model, self.world.data
        def sample(ids, order=None):
            q=d.qpos[m.jnt_qposadr[ids]];dq=d.qvel[m.jnt_dofadr[ids]]
            tau=d.qfrc_actuator[m.jnt_dofadr[ids]]
            if order is not None:q,dq,tau=q[order],dq[order],tau[order]
            return dict(q=q.tolist(),dq=dq.tolist(),tau_est=tau.tolist(),
                        time=self.monotonic(),temperature_max=25.,faults=[0]*len(ids),imu_rpy=self.imu_rpy)
        # SDK raw ordering is thumb,middle,index on the left; right state is
        # thumb,index,middle. PhysicalPlant performs the existing conversions.
        return dict(body=sample(self.body_ids), left=sample(self.hand_ids['left'],SIM_TO_BRIDGE),
                    right=sample(self.hand_ids['right']))

    def ctrl_dual_arm(self, q, tau_ff):
        self.commands += 1
        torque=np.asarray(tau_ff,dtype=float)
        if torque.shape!=(14,) or not np.isfinite(torque).all():
            raise ValueError('Expected fourteen finite feedforward torques')
        self.arm_feedforward=np.zeros(14) if self.force_zero_feedforward else torque.copy()
        self.max_abs_feedforward=np.maximum(self.max_abs_feedforward,np.abs(self.arm_feedforward))
        for n,v in zip(BODY_NAMES[15:],q):self.world.targets[self.actuators[n]]=v

    def ctrl_dual_hand(self, left, right):
        self.commands += 1
        for side,values in [('left',left),('right',right)]:
            for name,value in zip(FINGERS,np.asarray(values)[SIM_TO_BRIDGE]):
                self.world.targets[self.world.model.actuator(name.replace('left_',side+'_')).id]=value

    def hold_arm_measured(self):
        measured=np.asarray(self.read()['body']['q'][15:])
        self.ctrl_dual_arm(measured,np.zeros(14))
        if self.arm_reference is not None:self.arm_reference.reset(measured)
        self.wire_arm_q=measured.copy()


class SimulationCamera:
    def __init__(self, world, name='observe', intrinsics=None, serial='mujoco-synthetic'):
        self.world=world
        self.serial=serial
        self.renderer=mujoco.Renderer(world.model, height=480, width=640)
        self.camera=world.model.camera(name).id
        self.counter=0
        f=240/np.tan(np.deg2rad(world.model.cam_fovy[self.camera])/2)
        y,x=np.mgrid[:480,:640]
        if intrinsics is None:
            self.rays=np.stack([(x-319.5)/f,(y-239.5)/f,np.ones_like(x)],axis=-1)
        else:
            self.rays=np.stack([(x-intrinsics['ppx'])/intrinsics['fx'],
                                (y-intrinsics['ppy'])/intrinsics['fy'],np.ones_like(x)],axis=-1)
        self.transform=np.eye(4)
        self.transform[:3,:3]=world.data.cam_xmat[self.camera].reshape(3,3)@np.diag([1,-1,-1])
        self.transform[:3,3]=world.data.cam_xpos[self.camera]

    def frame(self):
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.world.data,camera=self.camera)
        rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth=self.renderer.render().copy()
        self.counter+=1
        return dict(color_bgr=rgb[...,::-1],depth_m=depth,rays=self.rays,
                    timestamp=time.monotonic(),meta=dict(camera_serial=self.serial,frame_id=self.counter,synthetic_rgbd=True))

    def close(self):self.renderer.close()


def setup_scene(setup):
    """Apply supplied setup geometry to a supported-base model, without fitting it."""
    config=setup['config'];intrinsics=setup['camera_intrinsics']
    from physical_core import rigid, calibrated_scene
    optical=rigid(config['camera_to_world'])
    if (intrinsics['width'],intrinsics['height'])!=(640,480):
        raise ValueError('Setup renderer currently requires 640x480 intrinsics')
    if any(intrinsics.get('coeffs',[0]*5)):
        raise ValueError('Setup renderer requires zero optical distortion coefficients')
    center=np.asarray(setup['can_center_world_m'],dtype=float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError('Expected finite three-dimensional initial can center')
    def decorate(root):
        quat=np.empty(4)
        if setup.get('stationary_lower_body_support',False):
            equality=root.find('equality')
            if equality is None:equality=ET.SubElement(root,'equality')
            for name,value in zip(BODY_NAMES[:15],setup['measured']['body']['q'][:15]):
                ET.SubElement(equality,'joint',joint1=name,polycoef=f'{value} 0 0 0 0',
                              solref='.002 1',solimp='.9999 .9999 .001')
        can=root.find(".//body[@name='can']")
        can.set('pos',' '.join(map(str,center)))
        shell=can.find("geom[@name='can_shell']")
        shell.set('size',f"{config['can_radius_m']} {config['can_height_m']/2}")
        top=can.find("geom[@name='can_top']")
        top.set('pos',f"0 0 {config['can_height_m']/2}")
        top.set('size',f"{config['can_radius_m']-.001} .0005")
        rotation=optical[:3,:3]@np.diag([1,-1,-1])
        mujoco.mju_mat2Quat(quat,rotation.ravel())
        # MuJoCo principal offsets have opposite X sign to pixel coordinates.
        # Verified by projecting known optical-axis and off-axis sphere points.
        ET.SubElement(root.find('worldbody'),'camera',name='setup-camera',
            pos=' '.join(map(str,optical[:3,3])),quat=' '.join(map(str,quat)),
            resolution='640 480',sensorsize='640 480',
            focalpixel=f"{intrinsics['fx']} {intrinsics['fy']}",
            principalpixel=f"{319.5-intrinsics['ppx']} {239.5-intrinsics['ppy']}")
    return calibrated_scene(config,x=center[0],y=center[1],decorate=decorate)


def apply_measured_state(world, io, measured):
    body=np.asarray(measured['body']['q'],dtype=float)
    if body.shape!=(29,) or not np.isfinite(body).all():
        raise ValueError('Expected twenty-nine measured body joint angles')
    velocities=np.asarray(measured['body'].get('dq',[0.]*29),dtype=float)
    if velocities.shape!=(29,) or not np.isfinite(velocities).all():raise ValueError('Invalid measured body velocities')
    for name,value in zip(BODY_NAMES,measured['body']['q']):
        jid=world.model.joint(name).id
        world.data.qpos[world.model.jnt_qposadr[jid]]=value
        world.targets[world.model.actuator(name).id]=value
    for name,value in zip(BODY_NAMES,velocities):
        world.data.qvel[world.model.jnt_dofadr[world.model.joint(name).id]]=value
    for side in ('left','right'):
        values=np.asarray(measured[side]['q'])
        if values.shape!=(7,) or not np.isfinite(values).all():
            raise ValueError('Expected seven measured hand joint angles')
        if side=='left':values=values[SIM_TO_BRIDGE]
        velocities=np.asarray(measured[side].get('dq',[0.]*7),dtype=float)
        if side=='left':velocities=velocities[SIM_TO_BRIDGE]
        if velocities.shape!=(7,) or not np.isfinite(velocities).all():raise ValueError('Invalid measured hand velocities')
        for name,value,velocity in zip(FINGERS,values,velocities):
            name=name.replace('left_',side+'_');jid=world.model.joint(name).id
            world.data.qpos[world.model.jnt_qposadr[jid]]=value
            world.targets[world.model.actuator(name).id]=value
            world.data.qvel[world.model.jnt_dofadr[jid]]=velocity
    io.imu_rpy=list(measured['body']['imu_rpy'])
    mujoco.mj_forward(world.model,world.data)


def run(output=None, hand_kp=2., camera='observe', height=.75, initial_arm_q14=None, staging_enabled=True, setup=None, force_zero_feedforward=False,
        *, identity_mode='synthetic_red', fault=None, capture_animation=False,
        position_noise='stochastic', noise_seed=7, noise_scale=1., noise_trace=None, noise_profile=None, planner_profile=None, planner_mode='legacy'):
    if planner_mode not in ('legacy','tuned'):raise ValueError('Planner mode must be legacy or tuned')
    if planner_profile is None and planner_mode=='tuned':
        planner_profile=json.loads((Path(__file__).parent/'g1_can_handoff/noise_aware_planner.json').read_text())
    if position_noise == 'stochastic' and noise_trace is not None:position_noise='recorded'
    noise = (StochasticPositionNoise(seed=noise_seed,scale=noise_scale,profile=noise_profile) if position_noise=='stochastic'
             else PositionNoise(position_noise, seed=noise_seed, scale=noise_scale, trace=noise_trace))
    if identity_mode not in ('synthetic_red','color_depth'):
        raise ValueError('Simulation identity mode must be synthetic_red or color_depth')
    if fault is not None and (fault.get('kind') not in ('stale','missing_target') or not fault.get('phase')):
        raise ValueError('Simulation fault requires a phase and stale/missing_target kind')
    from target_tracker import TargetTracker
    class NoYolo:
        calls=0
        def detect(self,frame):
            self.calls+=1
            raise AssertionError('Color/depth simulation attempted YOLO')
    detector=NoYolo()
    tracker=TargetTracker(detector,identity_mode='color_depth') if identity_mode=='color_depth' else None
    tracking_modes=set();fault_record=None;observation_phase=None
    if setup is not None:height=setup['config']['table_height_m']
    world=scene(height=height) if setup is None else setup_scene(setup)
    # Match the physical hand's configured kp=2, kd=.2; body support and the
    # simulated torque servo are explicit backend assumptions.
    for i in range(world.model.nu):
        if 'hand' in world.model.actuator(i).name:world.kp[i]=hand_kp
    io=SimulationIO(world,force_zero_feedforward=force_zero_feedforward)
    if initial_arm_q14 is not None:
        initial=np.asarray(initial_arm_q14,dtype=float)
        if initial.shape!=(14,) or not np.all(np.isfinite(initial)):
            raise ValueError('Initial arm state must contain fourteen finite joint angles')
        world.data.qpos[world.model.jnt_qposadr[io.body_ids[15:]]]=initial
        for name,value in zip(BODY_NAMES[15:],initial):world.targets[io.actuators[name]]=value
        mujoco.mj_forward(world.model,world.data)
    if setup is not None:
        apply_measured_state(world,io,setup['measured'])
        sensor=SimulationCamera(world,'setup-camera',setup['camera_intrinsics'],setup['config']['camera_serial'])
    else:sensor=SimulationCamera(world,camera)
    rows=[];counts={};worst_tracking=0.;display_phase='planning';trajectory=[]
    animation=[];last_render=-1.
    visualizer=mujoco.Renderer(world.model,height=480,width=640) if capture_animation else None
    view=mujoco.MjvCamera()
    view.lookat[:]=[.1,.1,.9];view.distance=2.25;view.azimuth=135;view.elevation=-20
    def capture(force=False):
        nonlocal last_render
        if visualizer is None or (not force and world.data.time-last_render < .2):return
        from PIL import Image,ImageDraw
        visualizer.update_scene(world.data,camera=view)
        image=Image.fromarray(visualizer.render().copy())
        ImageDraw.Draw(image).rectangle((0,0,640,38),fill='#10151e')
        ImageDraw.Draw(image).text((10,8),f'SIMULATION ONLY | {world.data.time:.1f}s | {display_phase}',fill='white')
        animation.append(image);last_render=float(world.data.time)
    def event(phase,**fields):
        nonlocal worst_tracking,display_phase
        counts[phase]=counts.get(phase,0)+1
        if phase=='target':
            worst_tracking=max(worst_tracking,fields['tracking_error_rad'])
            trajectory.append(dict(simulation_time_s=float(world.data.time),**fields))
            return
        if phase=='hand_target':return
        display_phase=phase
        rows.append(dict(phase=phase,simulation_time_s=float(world.data.time),**fields))
    config=dict(staging_enabled=staging_enabled,camera_serial='mujoco-synthetic',camera_to_world=sensor.transform.tolist(),
                can_radius_m=.033,can_height_m=.122,table_height_m=height,
                arm_limits_rad=world.model.jnt_range[io.body_ids[15:]].tolist())
    # Distinct model state prevents FK/collision queries from overwriting the
    # dynamics state, as on hardware.
    if setup is not None:
        config=dict(setup['config'],staging_enabled=staging_enabled)
    if planner_profile is not None:config.update(planner_profile)
    io.arm_limits=np.asarray(config['arm_limits_rad'],dtype=float)
    model=scene(height=height) if setup is None else setup_scene(setup)
    plant=PhysicalPlant(model,io,io,io.read,config,event,clock=io)
    planning_recorder=None
    if output:
        from planning_snapshot import PlanningRecorder
        planning_recorder=PlanningRecorder(Path(output).with_suffix('.planning'),event)
        plant.planning_recorder=planning_recorder
    initial_support=None
    contact_audit={'pregrasp_contact_steps':0,'minimum_pregrasp_contact_distance_m':None,'bodies':[]}
    def audit_contacts(world):
        if isinstance(noise,StochasticPositionNoise) and world.data.time-noise.last_update >= .02-1e-9:
            speed=float(np.max(np.abs(world.data.qvel[world.model.jnt_dofadr[io.body_ids[15:]]])))
            noise.advance(float(world.data.time),moving=speed>.04)
        capture()
        if plant.ik.contact_phase not in ('pregrasp','staging','approach','align'):return
        for contact in world.data.contact:
            names=[world.model.body(world.model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if 'can' not in names:continue
            other=names[1] if names[0]=='can' else names[0]
            if other=='world':continue
            previous=contact_audit['minimum_pregrasp_contact_distance_m']
            contact_audit['minimum_pregrasp_contact_distance_m']=float(contact.dist) if previous is None else min(previous,float(contact.dist))
            if contact.dist<=0:
                contact_audit['pregrasp_contact_steps']+=1
                if other not in contact_audit['bodies']:contact_audit['bodies'].append(other)
    world.frame_callback=audit_contacts
    def observe(phase,require_table,expected_center_xy=None):
        nonlocal fault_record,observation_phase,display_phase,initial_support
        observation_phase=phase;display_phase=phase
        frame=sensor.frame()
        if fault is not None and phase==fault['phase'] and fault_record is None:
            fault_record={**fault,'command_count_at_injection':io.commands,'frame_id':frame['meta']['frame_id']}
            if fault['kind']=='stale':frame['timestamp']-=float(fault.get('age_s',.6))
            elif fault['kind']=='missing_target':frame['color_bgr']=np.zeros_like(frame['color_bgr'])
            event('simulation_fault',injection=fault_record)
        try:
            observation,vision=geometry(frame,config,require_table=require_table,
                expected_center_xy=expected_center_xy,synthetic_red=identity_mode=='synthetic_red',tracker=tracker)
            if isinstance(noise,StochasticPositionNoise):
                if initial_support is None:initial_support=vision['pixels']
                speed=float(np.max(np.abs(world.data.qvel[world.model.jnt_dofadr[io.body_ids[15:]]])))
                noise.advance(float(world.data.time),moving=speed>.04,occluded=vision['pixels']<.65*initial_support)
            observation,vision=noise.apply(observation,vision,phase,frame['meta']['frame_id'],
                                          expected_center_xy=expected_center_xy)
            if tracker is not None:tracker.accept(observation,vision,frame)
            tracking_modes.add(vision.get('tracking',{}).get('mode','synthetic_red'))
        except Exception:
            if output:
                from PIL import Image
                Image.fromarray(frame['color_bgr'][...,::-1]).save(Path(output).with_suffix('.png'))
                np.savez_compressed(Path(output).with_suffix('.frame.npz'),
                    color_bgr=frame['color_bgr'],depth_m=frame['depth_m'],rays=frame['rays'],
                    metadata_json=json.dumps(dict(meta=frame['meta'],timestamp=frame['timestamp'],
                        config=config,phase=phase,require_table=require_table,
                        expected_center_xy=None if expected_center_xy is None else np.asarray(expected_center_xy).tolist(),
                        synthetic_fixture=True,simulation_position_noise=noise.last_metadata)),
                    truth_can_position=world.can_position(),
                    truth_can_quaternion=world.data.xquat[world.can_id].copy())
            raise
        event('camera',phase_name=phase,vision=vision)
        return observation,vision
    try:
        # As on hardware, reject an invalid pickup before taking SDK ownership.
        prepare(plant,observe,event)
        # Use the same bounded commanded hold as hardware ownership settling.
        # No simulation gravity or object truth is injected into the planner.
        plant.settle_control()
        held_plant=plant
        plant=PhysicalPlant(model,io,io,io.read,config,event,clock=io)
        plant.planning_recorder=planning_recorder
        plant.inherit_control_hold(held_plant)
        prepared=prepare(plant,observe,event)
        result=execute(plant,prepared,observe,event)
        result.update(simulation_only=True,physical_pickup_verified=False)
    except Exception as exc:
        commands_at_failure=io.commands
        if io.commands:plant.freeze()
        event('error',error=f'{type(exc).__name__}: {exc}')
        result=dict(lift_verified=False,error=str(exc),simulation_only=True,physical_pickup_verified=False,
                    commands_at_failure=commands_at_failure,stop_hold_commands=io.commands-commands_at_failure)
    finally:
        sensor.close()
        if visualizer is not None:
            capture(force=True)
            visualizer.close()
            if output and animation:
                animation_path=Path(output).with_suffix('.gif')
                animation[0].save(animation_path,save_all=True,append_images=animation[1:],
                                  duration=200,loop=0,optimize=False)
                result['animation']=animation_path.name
    result.update(perception=dict(identity_mode=identity_mode,yolo_calls=detector.calls,tracking_modes=sorted(tracking_modes)),
                  position_noise=dict(noise.description,generated_jumps=getattr(noise,'jump_events',[])),planner_profile=planner_profile,noise_profile_input=noise_profile,
                  planner_settings={key:config[key] for key in ('ik_reference_weight_m_per_rad','ik_position_tolerance_m','ik_variable_scale','grasp_candidate_budget','grasp_orientation_step_rad','planning_position_cov_m2')},
                  fault=fault_record,failure_phase=observation_phase if result.get('error') else None)
    result.update(simulation_contact_audit=contact_audit,command_count=io.commands,simulated_time_s=float(world.data.time),
                  truth_can_position_m=world.can_position().tolist(),events=rows,event_counts=counts,trajectory=trajectory,
                  max_tracking_error_rad=worst_tracking,
                  max_abs_commanded_feedforward_nm=io.max_abs_feedforward.tolist(),
                  backend_assumptions=dict(supported_base=True,hand_kp=hand_kp,hand_kd=.2,camera=camera,table_height_m=height,can_xy_m=[.35,.18],
                                           initial_arm_q14=None if initial_arm_q14 is None else list(initial_arm_q14),
                                           staging_enabled=staging_enabled,
                                           force_zero_feedforward=force_zero_feedforward,
                                           arm_kp=io.arm_kp.tolist(),arm_kd=io.arm_kd.tolist(),
                                           arm_reference='actual adapter ArmReference at100Hz,velocity.25,acceleration1',
                                           implicit_arm_gravity_compensation=False,implicit_hand_gravity_compensation=False),
                  dependency_versions={name:importlib.metadata.version(name)
                                       for name in ('mujoco','numpy','scipy','pillow')})
    if setup is not None:
        result['setup_input']=setup
        result['backend_assumptions'].update(camera='setup-camera',camera_intrinsics=setup['camera_intrinsics'],
            optical_extrinsic_source='supplied model-based estimate, not independently validated',
            can_xy_m=setup['can_center_world_m'][:2],measured_joint_positions_used=True,
            measured_joint_velocities_used=all('dq' in setup['measured'][side] for side in ('body','left','right')),
            temperature_and_fault_feedback='synthetic healthy',
            stationary_lower_body_equality_support=setup.get('stationary_lower_body_support',False),
            supplied_camera_transform_error=float(np.max(np.abs(sensor.transform-np.asarray(config['camera_to_world'])))))
    root=Path(__file__).parent
    result['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('physical_core.py','pickup.py','staging.py','camera_input.py','target_tracker.py','perception_policy.py',
                     'runtime.py','sim_runtime.py','planner_tuning.py','simulation_position_noise.py','stochastic_position_noise.py',
                     'g1_can_handoff/position_noise_profile.json','g1_can_handoff/stochastic_noise_profile.json','g1_can_handoff/noise_aware_planner.json',
                     'g1_can_handoff/sim.py','g1_can_handoff/task.py',
                     'adapter/arm_reference.py','adapter/g1_arm.py')}
    if output:
        Path(output).write_text(json.dumps(result,indent=2)+'\n')
        if result['lift_verified']:Path(output).with_suffix('.png').unlink(missing_ok=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='sim-physical-pickup.json')
    parser.add_argument('--hand-kp',type=float,default=2.)
    parser.add_argument('--camera',choices=['observe','head'],default='observe')
    parser.add_argument('--height',type=float,default=.75,help='Measured tabletop height in metres')
    parser.add_argument('--initial-arm-q',type=float,nargs=14,help='Measured left then right seven-joint arm angles')
    parser.add_argument('--disable-staging',action='store_true',help='Regression: require the direct approach path')
    parser.add_argument('--setup-json',help='Supplied setup geometry, camera intrinsics and measured joint snapshot')
    parser.add_argument('--force-zero-feedforward',action='store_true',help='Diagnostic regression: discard commanded arm feedforward')
    parser.add_argument('--identity-mode',choices=['color_depth','synthetic_red'],default='color_depth',
                        help='Default runs the deployed color/depth tracker; synthetic_red is the legacy simulation baseline')
    parser.add_argument('--capture-animation',action='store_true',help='Save a sampled 5 fps GIF beside the JSON result')
    parser.add_argument('--position-noise',choices=['stochastic','recorded','none'],default='stochastic',
                        help='Sample fitted jitter/drift/jumps; recorded replays fixed regressions; none retains the ideal baseline')
    parser.add_argument('--noise-seed',type=int,default=7,help='Repeatable fresh stochastic sequence or fixed-recording selection')
    parser.add_argument('--noise-scale',type=float,default=1.,help='1 uses the fitted distribution or original recorded offsets; other values are stress tests')
    parser.add_argument('--planner-mode',choices=['legacy','tuned'],default='tuned',help='Tuned sim controller by default; legacy retains deployed controller settings')
    parser.add_argument('--planner-profile-json',help='Bounded candidate IK/planner settings for offline evaluation')
    parser.add_argument('--noise-profile-json',help='Alternative fitted stochastic profile for offline evaluation')
    parser.add_argument('--noise-trace',choices=['handover','approach-drift'],help='Select one recorded pattern explicitly')
    parser.add_argument('--fault',choices=['stale-align','missing-can-align'],help='Inject a perception fault before alignment')
    args=parser.parse_args()
    fault=None if args.fault is None else dict(phase='align',kind='stale' if args.fault=='stale-align' else 'missing_target',age_s=.6)
    result=run(args.output,args.hand_kp,args.camera,args.height,args.initial_arm_q,not args.disable_staging,
               None if args.setup_json is None else json.loads(Path(args.setup_json).read_text()),args.force_zero_feedforward,
               identity_mode=args.identity_mode,fault=fault,capture_animation=args.capture_animation,
               position_noise=args.position_noise,noise_seed=args.noise_seed,noise_scale=args.noise_scale,noise_trace=args.noise_trace,
               noise_profile=None if args.noise_profile_json is None else json.loads(Path(args.noise_profile_json).read_text()),
               planner_profile=None if args.planner_profile_json is None else json.loads(Path(args.planner_profile_json).read_text()),planner_mode=args.planner_mode)
    print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
    raise SystemExit(0 if result['lift_verified'] else 1)
