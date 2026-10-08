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
from yolo_client import fresh, frame_age_limit_s

DATA=Path(os.getenv('HANDOFF_DATA','/data'));DATA.mkdir(parents=True,exist_ok=True)
EVENTS=DATA/('events-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.jsonl')

def http_json(port,path='/status'):
    return json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}{path}',timeout=2).read())


class Runtime:
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize,ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_,HandState_
        from lowstate_hub import LowstateHub
        from pose_alignment import PoseHistory
        ChannelFactoryInitialize(0,'enP8p1s0')
        self.operation_lock=threading.Lock();self.cancel=threading.Event();self.lock=threading.RLock();self.phase='PASSIVE';self.error=None;self.plant=None;self.arm=None;self.hand=None
        self.grasp_started=False
        self.demo_running=False;self.demo_result=None;self.demo_error=None
        self.observation_lock=threading.RLock();self.tracker=None;self.perception=None
        self.first_failure=None
        self.event_lock=threading.Lock()
        self.tracking_stop=threading.Event();self.tracking_thread=None
        self.frame_prefetch=None;self.pending_frame=None
        self.tracking_error=None;self.tracking_observation=None;self.tracking_vision=None
        self.tracking_baseline=None
        saved=DATA/'perception-config.json'
        if saved.exists():
            os.environ['HANDOFF_CAMERA_MAX_AGE_MS']=str(frame_age_limit_s(json.loads(saved.read_text())['camera_max_age_ms'])*1000)
        self.camera=Camera();self.latest={};self.motion_commands_sent=0;self.active=False
        self.archive_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='camera-evidence')
        self.body=ChannelSubscriber('rt/lowstate',LowState_);self.body.Init()
        self.pose_history=PoseHistory()
        self.hub=LowstateHub(self.body.Read,hz=200)
        self.hub.add_listener(lambda msg:self.pose_history.add(self.hub.latest()[2],
            list(msg.imu_state.rpy),[float(m.q) for m in msg.motor_state[12:15]]))
        self.hub.start()
        self.subs={side:ChannelSubscriber(f'rt/dex3/{side}/state',HandState_) for side in ('left','right')}
        for side,sub in self.subs.items():
            sub.Init();threading.Thread(target=self.poll_hand,args=(side,sub),daemon=True).start()
        threading.Thread(target=self.watchdog,daemon=True).start()
        self.event('startup',motion_enabled=False,automatic_home=False)

    def command_written(self,values,result):
        with self.lock:self.motion_commands_sent+=1
        # Exact DDS command content is retained by the independent recorder.
        if result is False:self.event('command_write_failed',result=False)

    def record_failure(self,exc,source):
        with self.lock:
            if getattr(self,'first_failure',None) is None:
                self.first_failure={'error':f'{type(exc).__name__}: {exc}','source':source}
                self.event('primary_failure',**self.first_failure)
            self.error=self.first_failure['error']
            return self.error

    def stop_perception(self):
        worker=getattr(self,'perception',None)
        if worker is not None:
            worker.close();self.perception=None

    def event(self,phase,**fields):
        row={'unix_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'phase':phase,**fields}
        # File IO must not hold the observation/command state lock. Keep full
        # frame evidence on disk without duplicating it into container stdout.
        with self.event_lock:
            with EVENTS.open('a') as f:
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        if phase not in ('target','hand_target','phase_camera_evidence','perception_result','perception_worker_result','exposure_pose'):
            print(json.dumps(row),flush=True)

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
        from yolo.confidence import confidence_threshold
        result={'phase':self.phase,'error':self.error,'motion_commands_sent':self.motion_commands_sent,
                'adapter_initialized':self.arm is not None,'automatic_home':False,'events':str(EVENTS),
                'calibration_present':(DATA/'calibration.json').is_file(),'lowstate':self.hub.status(),
                'camera_max_age_ms':frame_age_limit_s()*1000,
                'yolo_confidence_strictly_above':confidence_threshold()}
        observation=getattr(self,'tracking_observation',None)
        result.update(perception_mode='yolo_visual_depth',tracking_error=getattr(self,'tracking_error',None),
                      tracking_exposure_age_ms=None if observation is None else (time.monotonic()-observation.timestamp)*1000,
                      tracking_vision=getattr(self,'tracking_vision',None))
        result['first_failure']=getattr(self,'first_failure',None)
        result.update(demo_running=getattr(self,'demo_running',False),demo_result=getattr(self,'demo_result',None),
                      demo_error=getattr(self,'demo_error',None))
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
            config=refresh_scene(config,frame,self.read(),health['serial'],
                                 exposure_pose=self.pose_history.nearest(frame['timestamp']))
            self.event('scene_refreshed',calibration=config,independent_validation=False)
        config=calibration(config,health['serial'])
        calibration_start_guard(config,self.read())
        return config

    def configure_perception(self,max_age_ms):
        if self.active:raise RuntimeError('Cannot change camera limits during an active trial')
        if max_age_ms is None or isinstance(max_age_ms,bool):raise ValueError('camera_max_age_ms must be provided in milliseconds')
        value=frame_age_limit_s(max_age_ms)*1000
        pending=DATA/'perception-config.pending.json'
        pending.write_text(json.dumps({'camera_max_age_ms':value})+'\n')
        pending.replace(DATA/'perception-config.json')
        os.environ['HANDOFF_CAMERA_MAX_AGE_MS']=str(value)
        self.event('perception_configured',camera_max_age_ms=value,automatic_resume=False)
        return self.status()

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
        from contextlib import nullcontext
        with getattr(self,'observation_lock',nullcontext()):
            worker=getattr(self,'perception',None)
            if worker is not None:
                result=worker.observe(config,phase,require_table,expected_center_xy,self.event,
                                      pose_at=self.pose_history.nearest)
            else:result=self._observe_frame(config,phase,require_table,expected_center_xy)
            # Synchronous phase checks share this worker with background
            # tracking. Publish either checked observation before releasing
            # its serialization lock, so the watchdog sees the latest one.
            self.publish_tracking_observation(*result,require_table)
            return result

    def publish_tracking_observation(self,observation,vision,require_table):
        baseline=getattr(self,'tracking_baseline',None)
        if baseline is None:return
        tolerance=.01 if require_table else .04
        if np.linalg.norm(observation.center[:2]-baseline.center[:2])>tolerance:
            raise RuntimeError('Tracked can moved outside the checked pickup path')
        age=time.monotonic()-observation.timestamp
        if not 0<=age<=frame_age_limit_s():
            raise RuntimeError('Cannot publish stale tracking observation')
        with self.lock:
            if self.tracking_error:raise RuntimeError(self.tracking_error)
            previous=self.tracking_observation
            if previous is not None and observation.timestamp<=previous.timestamp:
                raise RuntimeError('Tracking observation exposure did not advance')
            self.tracking_observation=observation;self.tracking_vision=vision

    def exposure_config(self,config,timestamp):
        if config.get('calibration_method')!='operator_accepted_factory_table':return config
        from pose_alignment import corrected_config
        return corrected_config(config,self.pose_history.nearest(timestamp))

    def _observe_frame(self,config,phase,require_table,expected_center_xy=None):
        # Reserve a quarter of the final deadline for tracking/fit/archive.
        # The final freshness check still rejects any completed observation
        # beyond the configured deadline; acquisition is not the decision gate.
        acquisition_budget=frame_age_limit_s()*.75
        attempts=[]
        for attempt in range(3):
            frame=self.next_tracking_frame()
            previous=getattr(getattr(self,'tracker',None),'state',None)
            distinct=previous is None or frame['meta']['frame_id']>previous['frame_id']
            age=time.monotonic()-frame['timestamp']
            attempts.append({'frame_id':frame['meta']['frame_id'],'exposure_age_ms':age*1000,
                             'timings_ms':frame.get('timings_ms'),'distinct':distinct})
            if distinct and 0<=age<=acquisition_budget:
                break
            time.sleep(.02)
        else:
            self.event('camera_acquisition_rejected',attempts=attempts,budget_ms=acquisition_budget*1000)
            raise RuntimeError(f'Camera could not supply an exposure within the {acquisition_budget*1000:.0f} ms acquisition budget')
        config=self.exposure_config(config,frame['timestamp'])
        # Persist this exact frame while inference/geometric fitting runs.
        # Join the single archive job before returning any usable observation;
        # IO failure or excess exposure age still blocks the next motor action.
        archived=self.archive_executor.submit(self.archive_frame,frame,phase)
        detection={};started=time.monotonic();error=None
        from types import SimpleNamespace
        from yolo_client import YoloClient
        def detect(exact_frame):
            result=YoloClient().detect(exact_frame)
            detection.update(result)
            return result
        tracker=getattr(self,'tracker',None)
        if tracker is not None:tracker.detector=SimpleNamespace(detect=detect)
        try:
            result=geometry(frame,config,require_table=require_table,
                            expected_center_xy=expected_center_xy,
                            detector=SimpleNamespace(detect=detect),tracker=tracker)
            archived.result()
            fresh(frame)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            raise
        finally:
            archived.result()
            self.event('perception_result',phase_name=phase,frame_id=frame['meta']['frame_id'],
                       source=detection.get('source'),model_sha256=detection.get('model_sha256'),
                       detections=detection.get('detections'),inference_ms=detection.get('inference_ms'),
                       geometry=None if error else result[1],error=error,
                       processing_ms=(time.monotonic()-started)*1000,
                       exposure_age_ms=(time.monotonic()-frame['timestamp'])*1000)
        fresh(frame)
        if tracker is not None:tracker.accept(result[0],result[1],frame)
        return result

    def next_tracking_frame(self):
        pool=getattr(self,'frame_prefetch',None)
        if pool is None:return self.camera.frame(materialize=False)
        if self.pending_frame is None:
            self.pending_frame=pool.submit(self.camera.frame,materialize=False)
        frame=self.pending_frame.result()
        # One bounded prefetch overlaps camera IO with fitting the current
        # frame. No queue grows, and both frames retain their real exposures.
        self.pending_frame=pool.submit(self.camera.frame,materialize=False)
        return frame

    def tracking_guard(self):
        with self.lock:
            error=self.tracking_error;observation=self.tracking_observation
        if error:raise RuntimeError('Visual/depth tracking uncertain: '+error)
        if observation is None:raise RuntimeError('Visual/depth tracking has no accepted observation')
        age=time.monotonic()-observation.timestamp
        if not 0<=age<=frame_age_limit_s():
            raise RuntimeError(f'Visual/depth tracking exposure stale ({age*1000:.1f} ms)')

    def start_tracking(self,config,baseline):
        self.tracking_error=None;self.tracking_observation=None;self.tracking_vision=None
        self.tracking_baseline=baseline
        if self.cancel.is_set():raise RuntimeError('Operator cancelled before tracking')
        self.tracking_stop.clear()
        self.frame_prefetch=ThreadPoolExecutor(max_workers=1,thread_name_prefix='tracking-frame')
        self.pending_frame=None
        def sample():
            require_table=not self.grasp_started
            observation,vision=self.observe_frame(config,'motion-tracking',require_table,
                 None if require_table else baseline.center[:2])
        sample();self.tracking_guard()
        def work():
            while not self.tracking_stop.wait(.01):
                try:sample()
                except Exception as exc:
                    with self.lock:self.tracking_error=f'{type(exc).__name__}: {exc}'
                    self.event('tracking_failed',error=self.tracking_error)
                    if self.active and self.plant:
                        self.record_failure(exc,'tracking')
                        self.plant.freeze();self.active=False;self.phase='FROZEN'
                    break
        self.tracking_thread=threading.Thread(target=work,daemon=True,name='visual-depth-tracking')
        self.tracking_thread.start()

    def stop_tracking(self):
        stop=getattr(self,'tracking_stop',None)
        if stop is not None:stop.set()
        worker=getattr(self,'tracking_thread',None)
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=6)
            if worker.is_alive():raise RuntimeError('Tracking worker has not stopped; retain controller ownership')
        self.tracking_thread=None
        self.tracking_baseline=None
        pool=getattr(self,'frame_prefetch',None)
        if pool is not None:
            pool.shutdown(wait=True,cancel_futures=True)
            self.frame_prefetch=None;self.pending_frame=None

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

    def start_demo(self):
        # Admit one operator-requested job; its HTTP response never waits for
        # motion, so Stop and Release remain reachable throughout planning.
        if not self.operation_lock.acquire(blocking=False):raise RuntimeError('A demo/planning operation is already running')
        if self.active or (self.arm is not None and (self.phase!='FROZEN' or self.grasp_started)):
            self.operation_lock.release()
            raise RuntimeError('Complete the current pickup/handoff before starting another demo')
        self.cancel.clear();self.demo_running=True;self.demo_result=None;self.demo_error=None
        def work():
            try:self.demo_result=self.run(True,preserve_cancel=True)
            except Exception as exc:self.demo_error=self.record_failure(exc,'demo')
            finally:self.demo_running=False;self.operation_lock.release()
        try:threading.Thread(target=work,daemon=True,name='operator-demo').start()
        except Exception:
            self.demo_running=False;self.operation_lock.release();raise
        return self.status()

    def run(self,actuate=False,*,preserve_cancel=False):
        from pickup import prepare, execute
        resuming=self.arm is not None
        if self.active or (resuming and (self.phase!='FROZEN' or getattr(self,'grasp_started',True))):
            raise RuntimeError('Retry requires a pregrasp failure hold; complete handoff before a new pickup')
        if not preserve_cancel:self.cancel.clear()
        if self.cancel.is_set():raise RuntimeError('Operator cancelled before demo preparation')
        grasp_started=False
        self.first_failure=None
        self.grasp_started=False
        self.archive_guard();snapshot_guard(self.read())
        self.camera.acquire()
        try:
            config=self.config(refresh=True)
            from target_tracker import TargetTracker
            self.tracker=TargetTracker(recheck_interval_s=1.)
            from perception_worker import PerceptionWorker
            self.perception=PerceptionWorker(self.camera.token)
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
            self.start_tracking(config,prepared[0])
            plant.perception_guard=self.tracking_guard
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
            self.record_failure(exc,'execution');self.phase='BLOCKED' if self.arm is None else 'FROZEN'
            if self.plant:self.plant.freeze()
            elif self.arm:self.arm.hold_arm_measured()
            self.active=False;self.event('error',error=self.error,traceback=traceback.format_exc())
            self.event('failure_hold',grasp_started=grasp_started,
                       control_retained=self.arm is not None,automatic_release=False)
            raise
        finally:
            if self.phase!='HOLDING_CAN':
                self.stop_tracking();self.stop_perception();self.camera.close()

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
                    self.record_failure(exc,'watchdog')
                    self.plant.freeze();self.phase='FROZEN';self.active=False
            time.sleep(.02)

    def stop(self):
        self.cancel.set()
        tracking_stop=getattr(self,'tracking_stop',None)
        if tracking_stop is not None:tracking_stop.set()
        if self.plant:self.plant.freeze()
        self.active=False;self.phase='FROZEN' if self.arm else 'PASSIVE'
        return self.status()

    def relax_owned_arms(self,reason):
        self.stop_tracking();self.stop_perception();self.camera.close()
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
        operation=self.path in ('/plan','/trial','/capture','/hold-check','/rest-arm','/rest-arm-plan','/perception-config')
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
            elif self.path=='/demo/start':result=runtime.start_demo()
            elif self.path=='/stop':result=runtime.stop()
            elif self.path=='/perception-config':result=runtime.configure_perception(data.get('camera_max_age_ms'))
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
