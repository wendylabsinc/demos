"""G1 handoff runtime. Starts passive; actuation requires commissioned calibration.
Reuses the existing physical arm/Dex3 adapters. No simulated grasp evidence.
"""
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'adapter'))
from physical_core import calibration,calibration_start_guard,calibrated_scene,PhysicalPlant,snapshot_guard
from camera_input import Camera,geometry,materialize_rgbd
from yolo_client import fresh

DATA=Path(os.getenv('HANDOFF_DATA','/data'));DATA.mkdir(parents=True,exist_ok=True)
EVENTS=DATA/('events-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.jsonl')

def http_json(port,path='/status'):
    return json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=2).read())


class Runtime:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_,HandState_
        from lowstate_hub import LowstateHub
        ChannelFactoryInitialize(0,'enP8p1s0')
        self.operation_lock=threading.Lock();self.cancel=threading.Event();self.lock=threading.RLock();self.phase='PASSIVE';self.error=None;self.plant=None;self.arm=None;self.hand=None
        self.grasp_started=False
        self.camera=Camera();self.latest={};self.motion_commands_sent=0;self.active=False
        self.archive_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='camera-evidence')
        self.body=ChannelSubscriber('rt/lowstate',LowState_);self.body.Init()
        self.hub=LowstateHub(self.body.Read,hz=200).start()
        self.subs={side:ChannelSubscriber(f'rt/dex3/{side}/state',HandState_) for side in ('left','right')}
        for side,sub in self.subs.items():
            sub.Init();threading.Thread(target=self.poll_hand,args=(side,sub),daemon=True).start()
        threading.Thread(target=self.watchdog,daemon=True).start()
        self.event('startup',motion_enabled=False,automatic_home=False)

    def command_written(self,values,result):
        with self.lock:self.motion_commands_sent+=1
        # Exact DDS command content is retained by the independent recorder.
        if result is False:self.event('command_write_failed',result=False)

    def event(self,phase,**fields):
        row={'unix_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'phase':phase,**fields}
        with self.lock:
            with EVENTS.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        if phase not in ('target','hand_target'):print(json.dumps(row),flush=True)

    def poll_hand(self,side,sub):
        while True:
            try:
                msg=sub.Read()
                if msg is not None:
                    with self.lock:self.latest[side]=(msg,time.monotonic())
            except Exception as exc:self.event('hand_reader_error',side=side,error=str(exc))
            time.sleep(.1)

    @staticmethod
    def sample(msg,t,body=False):
        motors=msg.motor_state[:29] if body else msg.motor_state
        return {'imu_rpy':list(msg.imu_state.rpy) if body else None, 'q':[float(m.q) for m in motors], 'dq':[float(m.dq) for m in motors],
                'tau_est':[float(m.tau_est) for m in motors],
                'faults':[int(m.motorstate) for m in motors]+list(getattr(msg,'error',[])),
                'temperatures':[list(m.temperature) for m in motors],
                'temperature_max':max(max(m.temperature) for m in motors),'time':t}

    def read(self):
        _,msg,t=self.hub.latest()
        if msg is None:raise RuntimeError('Body telemetry unavailable')
        with self.lock:
            result={'body':self.sample(msg,t,True)}
            for side in ('left','right'):
                if side not in self.latest:raise RuntimeError(f'{side} hand telemetry unavailable')
                hm,ht=self.latest[side];result[side]=self.sample(hm,ht)
        return result

    def status(self):
        result={'phase':self.phase,'error':self.error,'motion_commands_sent':self.motion_commands_sent,
                'adapter_initialized':self.arm is not None,'automatic_home':False,'events':str(EVENTS),
                'calibration_present':(DATA/'calibration.json').is_file(),'lowstate':self.hub.status()}
        try:
            snap=self.read();result['measured']=snap
            snapshot_guard(snap);result['telemetry_healthy']=True
        except Exception as exc:result.update(telemetry_healthy=False,telemetry_error=str(exc))
        return result

    def config(self, refresh=False):
        path=DATA/'calibration.json'
        if not path.exists():raise RuntimeError('Missing accepted camera/base calibration')
        health=http_json(8001,'/health')
        config=json.loads(path.read_text())
        if refresh and config.get('calibration_method')=='operator_accepted_factory_table':
            from scene_refresh import refresh_scene
            frame=self.camera.frame();self.archive_frame(frame,'scene_refresh')
            config=refresh_scene(config,frame,self.read(),health['serial'])
            self.event('scene_refreshed',calibration=config,independent_validation=False)
        config=calibration(config,health['serial'])
        calibration_start_guard(config,self.read())
        return config

    def archive_guard(self):
        s=http_json(8101)
        if (not s['recording'] or s['error'] or sum(s['dropped'].values()) or
            s['missing_or_stale_required'] or s['queue_depth']>1000 or
            max(s.get('last_written_age_s',{}).get(t,999) for t in ('rt/lowstate','rt/dex3/left/state','rt/dex3/right/state'))>.5):
            raise RuntimeError('Passive recorder health gate failed')
        return s

    def archive_frame(self,frame,phase):
        path=DATA/f"camera-{frame['meta']['frame_id']}-{phase}.npz"
        path.write_bytes(frame['raw'] if 'raw' in frame else materialize_rgbd(frame))
        path.with_suffix('.source.zip').write_bytes(frame['source_rgbd_zip'])
        path.with_suffix('.timing.npz').write_bytes(frame['source_timing_npz'])
        self.event('phase_camera_evidence',phase_name=phase,path=str(path),
                   metadata=frame['meta'],color_codec=frame['color_codec'])

    def observe_frame(self,config,phase,require_table,expected_center_xy=None):
        # Reserve processing time within the unchanged 300 ms final deadline.
        # Discard an old acquisition before it can become a motor observation.
        for attempt in range(3):
            frame=self.camera.frame(materialize=False)
            if 0<=time.monotonic()-frame['timestamp']<=.18:
                break
        else:
            raise RuntimeError('Camera could not supply an exposure within the 180 ms acquisition budget')
        # Persist this exact frame while inference/geometric fitting runs.
        # Join the single archive job before returning any usable observation;
        # IO failure or >300 ms age still blocks the next motor action.
        archived=self.archive_executor.submit(self.archive_frame,frame,phase)
        try:
            result=geometry(frame,config,require_table=require_table,
                            expected_center_xy=expected_center_xy)
        finally:
            archived.result()
        fresh(frame)
        return result

    def capture(self):
        self.camera.acquire()
        try:
            f=self.camera.frame();p=DATA/f"camera-{f['meta']['frame_id']}.npz";p.write_bytes(f['raw'])
            zip_path=p.with_suffix('.source.zip');zip_path.write_bytes(f['source_rgbd_zip'])
            timing_path=p.with_suffix('.timing.npz');timing_path.write_bytes(f['source_timing_npz'])
            self.event('camera_capture',path=str(p),source_zip=str(zip_path),source_timing=str(timing_path),metadata=f['meta'],
                       color_codec=f['color_codec'])
            return {'path':str(p),'metadata':f['meta'],'motion_commands_sent':self.motion_commands_sent}
        finally:self.camera.close()

    def hold_check(self):
        # No camera transform is needed to hold the freshly measured pose.
        # This is an ownership/neutral-reference test, never a pickup claim.
        self.cancel.clear()
        if self.arm:raise RuntimeError('Controller already owns arms')
        snap=snapshot_guard(self.read());archive=self.archive_guard()
        try:http_json(8090)
        except Exception:pass
        else:raise RuntimeError('Existing onboard controller is still running')
        age=archive['age_s'].get('rt/arm_sdk')
        if age is not None and age<2:raise RuntimeError('Another arm SDK publisher is active')
        from g1_arm import G1_29_ArmController
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        observer=ChannelSubscriber('rt/arm_sdk',LowCmd_);observer.Init()
        start=np.array(snap['body']['q'][15:29]);worst=0;zero_seen=False
        self.phase='MEASURED_POSE_HOLD_CHECK'
        self.event('hold_check_start',measured_arm_q=start.tolist(),duration_s=2,
                   recorder_session=archive['session'],hand_commands_enabled=False)
        try:
            self.arm=G1_29_ArmController(motion_mode=True,simulation_mode=False,lowstate_hub=self.hub,
                                        record_lowcmd=self.command_written)
            self.arm.set_arm_motion_limits({'max_velocity_rad_s':.25,'max_acceleration_rad_s2':1.})
            end=time.monotonic()+2
            while time.monotonic()<end:
                if self.cancel.is_set():raise RuntimeError('Operator stopped hold check')
                fresh=snapshot_guard(self.read());q=np.array(fresh['body']['q'][15:29])
                worst=max(worst,float(np.max(np.abs(q-start))))
                if worst>.08:raise RuntimeError('Measured hold drift exceeds 0.08 rad')
                self.archive_guard()
                time.sleep(.05)
            self.event('hold_reference_observed',worst_joint_delta_rad=worst,adapter_publish_hz=self.arm.loop_hz)
        finally:
            if self.arm:
                self.arm.hold_arm_measured()
                self.arm.shutdown(ramp_s=1.)
                # Send an explicit final zero-weight packet after the writer
                # joins. A sleep-based ramp alone does not prove that its final
                # zero reached a publish tick.
                self.arm.msg.motor_cmd[29].q=0.
                self.arm.msg.crc=self.arm.crc.Crc(self.arm.msg)
                wrote=self.arm.lowcmd_publisher.Write(self.arm.msg)
                self.command_written(None,wrote)
                observed=observer.Read(timeout=.5)
                zero_seen=observed is not None and abs(observed.motor_cmd[29].q)<1e-6
                stopped=not self.arm.publish_thread.is_alive()
                self.event('hold_check_release',zero_weight_observed=zero_seen,publisher_thread_stopped=stopped,
                           dds_write_result=wrote,physical_factory_ownership_verified=None,automatic_home=False)
                self.arm=None
            observer.Close();self.phase='PASSIVE'
        if not zero_seen:raise RuntimeError('Final zero arm SDK blend weight was not observed')
        return {'neutral_hold_worst_delta_rad':worst,'zero_weight_observed':zero_seen,
                'publisher_thread_stopped':stopped,'motion_commands_sent':self.motion_commands_sent,
                'physical_pickup_verified':False,'physical_factory_ownership_verified':None}

    def make_plant(self,config):
        p=calibrated_scene(config)
        return PhysicalPlant(p,self.arm,self.hand,self.read,config,self.event,cancel=self.cancel)

    def rest_arm(self,actuate=False):
        """Explicit checked return to the recorded relaxed left-arm pose."""
        from staging import joint_path
        from types import SimpleNamespace
        import mujoco
        if self.active or getattr(self,'grasp_started',False):
            raise RuntimeError('Cannot lower while active or possibly holding the can')
        self.cancel.clear();self.camera.acquire()
        try:
            config=self.config(refresh=True)
            archive=self.archive_guard()
            if self.arm is None:
                age=archive['age_s'].get('rt/arm_sdk')
                if age is not None and age<2:raise RuntimeError('Another arm SDK publisher is active')
                try:http_json(8090)
                except Exception:pass
                else:raise RuntimeError('Existing onboard controller is still running')
            plant=self.make_plant(config)
            observation,_=geometry(self.camera.frame(),config)
            plant.observe_can(observation);plant.set_contact_phase('pregrasp')
            original=plant.ik.data.qpos.copy()
            start=original[plant.ik.qadr[plant.ik.arm]].copy()
            target=np.array([.25,.27,-.08,.82,0.,0.,0.])
            path,metadata=joint_path(plant.ik,target,np.asarray(config['arm_limits_rad'])[:7],check=plant.check)
            try:
                for waypoint in path:
                    plant._joint_poses(waypoint)
                    plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]]=waypoint
                    mujoco.mj_forward(plant.ik.model,plant.ik.data)
            finally:
                plant.ik.data.qpos[:]=original;mujoco.mj_forward(plant.ik.model,plant.ik.data)
            self.event('rest_arm_planned',target_q=target.tolist(),waypoints=path,**metadata)
            if not actuate:return {'planned':True,'hand_commands_sent':0}
            calibration_start_guard(config,self.read())
            if np.max(np.abs(np.asarray(self.read()['body']['q'][15:22])-start))>.04:
                raise RuntimeError('Arm moved during rest planning')
            if self.cancel.is_set():raise RuntimeError('Operator cancelled rest motion')
            if self.arm is None:
                from g1_arm import G1_29_ArmController
                self.arm=G1_29_ArmController(motion_mode=True,simulation_mode=False,lowstate_hub=self.hub,record_lowcmd=self.command_written)
                self.arm.set_arm_motion_limits({'max_velocity_rad_s':.25,'max_acceleration_rad_s2':1.})
            self.arm.set_arm_position_limits(config['arm_limits_rad'])
            plant.adapter=self.arm
            plant.sender=SimpleNamespace(ctrl_dual_hand=lambda *args:None)
            self.plant=plant;self.active=True;self.phase='LOWERING_ARM';self.trial_deadline=time.monotonic()+60
            for waypoint in path:
                plant.sync(plant.check());goal,poses=plant._joint_poses(waypoint)
                plant._execute_poses(goal,poses)
            measured=self.read()['body']['q'][15:22]
            error=float(np.max(np.abs(np.asarray(measured)-target)))
            if error>.04:raise RuntimeError('Resting arm pose not reached')
            self.active=False;self.arm.hold_arm_measured();self.phase='RESTING_ARM'
            self.event('rest_arm_verified',measured_q=measured,max_error_rad=error)
            self.relax_owned_arms('verified_rest_pose')
            return {'arm_lowered':True,'max_error_rad':error,'hand_commands_sent':0,'control_released':True}
        except Exception as exc:
            self.error=str(exc);self.active=False
            if self.plant:self.plant.freeze()
            elif self.arm:self.arm.hold_arm_measured()
            self.phase='FROZEN' if self.arm else 'BLOCKED'
            self.event('rest_arm_error',error=str(exc))
            raise
        finally:self.camera.close()

    def run(self,actuate=False):
        from pickup import prepare, execute
        resuming=self.arm is not None
        if self.active or (resuming and (self.phase!='FROZEN' or getattr(self,'grasp_started',True))):
            raise RuntimeError('Retry requires a pregrasp failure hold; complete handoff before a new pickup')
        self.cancel.clear()
        grasp_started=False
        self.archive_guard();snapshot_guard(self.read())
        self.camera.acquire()
        try:
            config=self.config(refresh=True)
            def observe(phase, require_table, expected_center_xy=None):
                return self.observe_frame(config,phase,require_table,expected_center_xy)
            def phase_event(phase, **fields):
                nonlocal grasp_started
                if phase=='close':
                    grasp_started=True;self.grasp_started=True
                self.phase='HOLDING_CAN' if phase=='lift_verified' else phase
                self.event(phase,**fields)
            plant=self.make_plant(config)
            prepared=prepare(plant,observe,phase_event)
            calibration_start_guard(config,self.read())
            if config.get('scene_refresh_automatic'):
                pending=DATA/'calibration.pending.json'
                pending.write_text(json.dumps(config,indent=2)+'\n')
                pending.replace(DATA/'calibration.json')
            self.event('plan_context',calibration=config,motion_enabled=actuate)
            if not actuate:
                if resuming:self.phase='FROZEN'
                return {'planned':True,'motion_commands_sent':self.motion_commands_sent}
            # Old teleop service must be stopped, not merely STANDBY, before a
            # separate publisher can take ownership. Refuse any reachable service.
            try:http_json(8090)
            except Exception:pass
            else:raise RuntimeError('Existing onboard controller is still running; ownership not exclusive')
            archive=self.archive_guard()
            if not resuming and archive['age_s'].get('rt/arm_sdk') is not None and archive['age_s']['rt/arm_sdk']<2:
                raise RuntimeError('Another arm SDK publisher is active')
            if self.cancel.is_set():raise RuntimeError('Operator cancelled before control acquisition')
            plant.validate_start()
            if not resuming:
                from g1_arm import G1_29_ArmController
                from g1_dex3 import Dex3Sender
                self.arm=G1_29_ArmController(motion_mode=True,simulation_mode=False,lowstate_hub=self.hub,record_lowcmd=self.command_written)
                if self.cancel.is_set():raise RuntimeError('Operator cancelled during control acquisition')
                self.arm.set_arm_motion_limits({'max_velocity_rad_s':.25,'max_acceleration_rad_s2':1.})
                self.arm.set_arm_position_limits(config['arm_limits_rad'])
                self.hand=Dex3Sender(kp=2,kd=.2)
                self.event('ownership_acquired',recorder_session=archive['session'],right_arm_hold=plant.right_arm.tolist(),
                           right_hand_hold=plant.right_hand.tolist(),blend_affects_both_arms=True)
            else:
                self.arm.set_arm_position_limits(config['arm_limits_rad'])
                self.event('ownership_reused',recorder_session=archive['session'])
            self.error=None;self.grasp_started=False
            plant.adapter=self.arm;plant.sender=self.hand;self.plant=plant
            self.active=True;self.trial_deadline=time.monotonic()+90
            result=execute(plant,prepared,observe,phase_event)
            self.event('physical_pickup_verified',physical_success_scope='single calibrated stationary pickup')
            return {'physical_lift_verified':result['lift_verified'],'visual_rise_m':result['visual_rise_m'],
                    'hold_verified_s':result['hold_verified_s'],'awaiting_operator_take':True}
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}';self.phase='BLOCKED' if self.arm is None else 'FROZEN'
            if self.plant:self.plant.freeze()
            elif self.arm:self.arm.hold_arm_measured()
            self.active=False;self.event('error',error=self.error,traceback=traceback.format_exc())
            self.event('failure_hold',grasp_started=grasp_started,
                       control_retained=self.arm is not None,automatic_release=False)
            raise
        finally:self.camera.close()

    def watchdog(self):
        last_archive=0
        while True:
            if self.active and self.plant:
                try:
                    self.plant.check()
                    if time.monotonic()>self.trial_deadline:raise RuntimeError('Bounded trial deadline reached')
                    if time.monotonic()-last_archive>1:
                        self.archive_guard();last_archive=time.monotonic()
                    if self.phase=='HOLDING_CAN':
                        self.plant.maintain_hand()
                except Exception as exc:
                    self.error=str(exc);self.plant.freeze();self.phase='FROZEN';self.active=False
            time.sleep(.02)

    def stop(self):
        self.cancel.set()
        if self.plant:self.plant.freeze()
        self.active=False;self.phase='FROZEN' if self.arm else 'PASSIVE'
        return self.status()

    def relax_owned_arms(self,reason):
        if self.hand:self.hand.relax()
        if self.arm:self.arm.shutdown(ramp_s=1.)
        self.hand=None;self.arm=None;self.plant=None;self.active=False;self.phase='PASSIVE';self.grasp_started=False
        self.event('released',can_supported=True,automatic_home=False,reason=reason)

    def release(self,supported):
        if supported is not True:raise RuntimeError('Can support must be confirmed before release')
        self.stop()
        # A cancelled trial must finish its current call and exception cleanup
        # before release can relax hands or destroy its command adapters.
        if not self.operation_lock.acquire(blocking=False):
            raise RuntimeError('Trial cancellation is pending; wait for the operation to finish before release')
        try:
            self.relax_owned_arms('supported_can_release')
            return self.status()
        finally:self.operation_lock.release()


runtime=None
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(runtime.status())
    def do_POST(self):
        if self.client_address[0] not in ('127.0.0.1','::1'):
            self.respond({'error':'Control endpoints are loopback-only'},403);return
        operation=self.path in ('/plan','/trial','/capture','/hold-check','/rest-arm','/rest-arm-plan')
        acquired=False
        try:
            if operation:
                acquired=runtime.operation_lock.acquire(blocking=False)
                if not acquired:raise RuntimeError('A planning/capture/trial operation is already running')
            size=int(self.headers.get('Content-Length','0'))
            data=json.loads(self.rfile.read(size) or b'{}')
            if self.path=='/rest-arm':result=runtime.rest_arm(True)
            elif self.path=='/rest-arm-plan':result=runtime.rest_arm(False)
            elif self.path=='/hold-check':result=runtime.hold_check()
            elif self.path=='/capture':result=runtime.capture()
            elif self.path=='/plan':result=runtime.run(False)
            elif self.path=='/trial':
                result=runtime.run(True)
            elif self.path=='/stop':result=runtime.stop()
            elif self.path=='/release':result=runtime.release(data.get('can_supported'))
            else:raise ValueError('Unknown endpoint')
            self.respond(result)
        except Exception as exc:
            runtime.event('request_error',path=self.path,error=str(exc),exception_type=type(exc).__name__)
            self.respond({'error':str(exc),'status':runtime.status()},409)
        finally:
            if acquired:runtime.operation_lock.release()
    def respond(self,result,code=200):
        body=json.dumps(result,allow_nan=False).encode();self.send_response(code)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

def main():
    global runtime
    runtime=Runtime()
    server=ThreadingHTTPServer(('0.0.0.0',8102),Handler)
    def shutdown(*args):
        runtime.stop()
        if runtime.arm:
            runtime.arm.shutdown(ramp_s=1.)
            runtime.event('shutdown_ownership_release',automatic_home=False,hand_open_command_sent=False,
                          physical_release_verified=False)
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,shutdown)
    signal.signal(signal.SIGINT,shutdown)
    try:server.serve_forever()
    finally:runtime.hub.stop();server.server_close()

if __name__=='__main__':main()
