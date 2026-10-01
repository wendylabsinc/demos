"""Real force-based flight with asynchronous onboard perception and planning."""
from __future__ import annotations
import copy
import io
import math
import platform
import threading
import time
from collections import deque
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from .control import FlightController, quat_to_matrix
from .model import GRAVITY,MASS,PHYSICS_HZ,ROTOR_MAX_THRUST,SENSOR_WIDTH,TIMESTEP,build_model
from .yard import BOUNDS,HOME,ZONES
from .environment import YardEnvironment
from .sensors import SensorRig,scene_state
from .navigation import Navigator,unproject
from .missions import MissionTracker
from .intent import IntentService
from .voice import VoiceService
CONTROL_HZ=250
CONTROL_DT=1/CONTROL_HZ
CONTROL_DECIMATION=PHYSICS_HZ//CONTROL_HZ
CRUISE_HEIGHT=4.5
MAX_SPEED=2.0
MAX_ACCELERATION=1.5
class Trajectory:
    """A quintic segment preserving the previous reference's derivatives."""

    def __init__(self, start: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
                 goal: np.ndarray, now: float):
        self.started = now
        self.goal = np.asarray(goal, dtype=float).copy()
        distance = float(np.linalg.norm(self.goal - start))
        duration = max(1.0, 1.875 * distance / MAX_SPEED,
                       math.sqrt(5.774 * distance / MAX_ACCELERATION))
        for _ in range(35):
            matrix = np.array([[duration ** 3, duration ** 4, duration ** 5],
                               [3 * duration ** 2, 4 * duration ** 3, 5 * duration ** 4],
                               [6 * duration, 12 * duration ** 2, 20 * duration ** 3]])
            c0, c1, c2 = start, velocity, acceleration / 2
            rhs = np.array([self.goal - c0 - c1 * duration - c2 * duration ** 2,
                            -c1 - 2 * c2 * duration, -2 * c2])
            self.coefficients = np.vstack([c0, c1, c2, np.linalg.solve(matrix, rhs)])
            self.duration = duration
            samples = [self.sample(now + value) for value in np.linspace(0, duration, 150)]
            if (max(np.linalg.norm(s[1]) for s in samples) <= MAX_SPEED + 1e-6 and
                    max(np.linalg.norm(s[2]) for s in samples) <= MAX_ACCELERATION + 1e-6):
                break
            duration *= 1.15

    def sample(self, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = float(np.clip(now - self.started, 0, self.duration))
        c = self.coefficients
        if t >= self.duration - 1e-10:
            return self.goal.copy(), np.zeros(3), np.zeros(3)
        position = np.array([1, t, t * t, t ** 3, t ** 4, t ** 5]) @ c
        velocity = np.array([0, 1, 2 * t, 3 * t * t, 4 * t ** 3, 5 * t ** 4]) @ c
        acceleration = np.array([0, 0, 2, 6 * t, 12 * t * t, 20 * t ** 3]) @ c
        return position, velocity, acceleration

    def complete(self, now: float) -> bool:
        return now >= self.started + self.duration


class Simulation:
    def __init__(self,*,data_dir=None,realtime_speed=1.0):
        self.model=build_model();self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.controller=FlightController();self.environment=YardEnvironment(self.model,self.data)
        self.navigator=Navigator();self._navigation_state=self.navigator.snapshot();self.missions=MissionTracker(data_dir)
        self.intent=IntentService();self.voice=VoiceService();self.vision=None
        self.lock=threading.RLock();self._stop=threading.Event();self._thread=None;self._sensor_thread=None
        self.speed=realtime_speed;self.body_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'drone')
        self.steps=0;self.elapsed=0.;self.started_wall=time.time();self.error=None;self.sensor_error=None
        self.contacts=0;self.obstacle_contacts=0;self.late_resyncs=0;self.rtf_window=deque(maxlen=240)
        self.last_command='Ready for takeoff';self.mode='idle';self.motors='off'
        self.reference=np.array(HOME);self.reference_velocity=np.zeros(3);self.reference_acceleration=np.zeros(3)
        self.goal=self.reference.copy();self.trajectory=None;self.waypoints=[];self.destination=None
        self.mission_zone=None;self._dwell_started=None;self._scan_goal=None;self._scan_key=None
        self._capture_request=None;self._capture_busy=False;self._scan_remaining=[]
        self._follow_active=False;self._tracking_started=False;self._tracking_lost=False;self._person=None;self._search_visited=[];self._last_follow_replan=-10.
        self.error_now=0.;self.max_tracking_error=0.;self._camera_jpeg=None;self._camera_time=None
        self._camera_wall=None;self._sensor_frames=0;self._tracks=[];self._perception_status={}
        self._actions=deque();self._plan=[];self._last_path_check=-1.

    def sensors(self):
        s=self.data.sensordata.reshape(1,SENSOR_WIDTH)
        return s[:,:3].copy(),s[:,3:6].copy(),s[:,6:10].copy(),s[:,10:13].copy()

    def _segment(self,goal):
        self.goal=np.asarray(goal,float).copy()
        self.trajectory=Trajectory(self.reference,self.reference_velocity,self.reference_acceleration,
                                   self.goal,float(self.data.time))

    def _route(self,goal,*,landing=False):
        goal=np.asarray(goal,float);position=self.data.sensordata[:3].copy()
        self.destination=goal.copy();self.waypoints=[]
        if self.motors=='off':
            self.reference=position.copy();self.reference_velocity=np.zeros(3);self.reference_acceleration=np.zeros(3)
        airborne=position.copy()
        if position[2]<CRUISE_HEIGHT-.3 and np.linalg.norm(position[:2]-goal[:2])>.25:
            airborne[2]=CRUISE_HEIGHT;self.waypoints.append(airborne.copy())
        if landing:
            cruise=np.array([HOME[0],HOME[1],CRUISE_HEIGHT])
            self.waypoints.extend(self.navigator.plan(airborne,cruise));self.waypoints.append(goal)
        elif np.linalg.norm(position[:2]-goal[:2])>.25:
            self.waypoints.extend(self.navigator.plan(airborne,goal))
        else:self.waypoints.append(goal.copy())
        self._segment(self.waypoints.pop(0));self.motors='on';self._dwell_started=None

    def _cancel_mission(self):
        self.missions.cancel();self.mission_zone=None;self._scan_goal=None;self._scan_key=None
        self._scan_remaining=[];self._capture_request=None;self._capture_busy=False;self._dwell_started=None
        self._tracks=[];self._person=None

    def _next_scan(self):
        position=self.data.sensordata[:3].copy()
        if self.missions.mission['kind']=='current':
            if not self._scan_remaining:return self._finish_scan()
            key=min(self._scan_remaining,key=lambda k:np.linalg.norm(np.asarray(self.missions.baseline['observations'][k]['pose'])-position))
            self._scan_remaining.remove(key);goal=np.array(self.missions.baseline['observations'][key]['pose'])
        else:
            goal=self.navigator.next_scan(position)
            if goal is None or (len(self.navigator.visited)>=4 and self.navigator.covered.mean()>.95):return self._finish_scan()
            key=f'V{len(self.navigator.visited):02d}'
        self._scan_key=key;self._scan_goal=goal
        self.mission_zone=min(ZONES,key=lambda z:np.linalg.norm(np.asarray(ZONES[z]['position'])[:2]-goal[:2]))
        self._route(goal)

    def _finish_scan(self):
        self.missions.finish();self._scan_goal=None;self._scan_key=None;self.mission_zone=None
        self.mode='hover';self.last_command='Flight imagery captured; mission complete.'

    def _navigation(self,position,velocity):
        now=float(self.data.time)
        if self.trajectory is not None:
            self.reference,self.reference_velocity,self.reference_acceleration=self.trajectory.sample(now)
            arrived=self.trajectory.complete(now) and np.linalg.norm(position-self.goal)<.2 and np.linalg.norm(velocity)<.22
            if arrived:
                if self.waypoints:self._segment(self.waypoints.pop(0))
                else:
                    self.trajectory=None
                    if self.mode=='landing':self.mode,self.motors='idle','off';self.last_command='Landed at home'
                    elif self.mode not in ['follow','baseline','current']:self.mode='hover'
        if self.mode=='follow':
            fresh=self._person is not None and now-self._person['time']<2.5
            if fresh:
                self._tracking_started=True;self._tracking_lost=False
            if not fresh and self._tracking_started:
                if not self._tracking_lost:
                    self.waypoints=[];self.destination=self.reference+.6*self.reference_velocity;self._segment(self.destination)
                    self._tracking_lost=True;self._person=None;self._tracks=[]
                    self.last_command='Worker observation lost; braking and holding position'
            elif fresh and now-self._last_follow_replan>1.0:
                desired=np.array(self._person['position']);desired+=np.array([-2.5,0,CRUISE_HEIGHT-desired[2]])
                desired[:2]=np.clip(desired[:2],[-8.,-7.],[8.,7.])
                if self.destination is None or np.linalg.norm(desired-self.destination)>.35:
                    self._route(desired)
                self._last_follow_replan=now;self.last_command='Following the visually observed worker'
            elif not fresh and self.trajectory is None:
                self._person=None;self._tracks=[]
                # Camera search selects unobserved area, then holds if exhausted.
                goal=self.navigator.next_scan(position)
                if goal is not None:
                    self.navigator.visited.append(goal);self._route(goal)
                    self.last_command='Searching camera coverage for a worker'
                else:self.last_command='No fresh worker observation; holding position'
        if self._scan_goal is not None and self.trajectory is None and not self._capture_busy:
            stable=(np.linalg.norm(position-self._scan_goal)<.03 and np.linalg.norm(velocity)<.035
                    and np.linalg.norm(self.data.sensordata[10:13])<.03)
            if not stable:self._dwell_started=None
            elif self._dwell_started is None:self._dwell_started=now
            elif now-self._dwell_started>.6:
                self._capture_request={'key':self._scan_key,'missionId':self.missions.mission['id'],
                                       'position':position.copy(),'zone':self.mission_zone}
                self._capture_busy=True
        if self._actions and self.trajectory is None and self._scan_goal is None and self.mode!='follow':
            action=self._actions.popleft();self.last_command=self._apply_action(action)

    def _control(self):
        position,velocity,quaternion,rates=self.sensors();self._navigation(position[0],velocity[0])
        target=tuple(v.reshape(1,3) for v in (self.reference,self.reference_velocity,self.reference_acceleration))
        self.data.ctrl[:]=self.controller.update(CONTROL_DT,position,velocity,quat_to_matrix(quaternion),rates,
                                                  target,self.motors,position[0,2]>.3)[0]
        self.data.xfrc_applied[self.body_id,:3]=-.13*velocity[0]*np.linalg.norm(velocity[0])
        self.error_now=float(np.linalg.norm(position[0]-self.reference))
        if self.motors=='on':self.max_tracking_error=max(self.max_tracking_error,self.error_now)

    def step(self):
        if self.error:return
        if self.steps%CONTROL_DECIMATION==0:self._control()
        self.environment.step(TIMESTEP);mujoco.mj_step(self.model,self.data)
        self.steps+=1;self.elapsed+=TIMESTEP
        for contact in self.data.contact[:self.data.ncon]:
            bodies=self.model.geom_bodyid[[contact.geom1,contact.geom2]]
            if self.body_id in bodies:
                self.contacts+=1
                if contact.geom1!=0 and contact.geom2!=0:self.obstacle_contacts+=1
        if not np.isfinite(self.data.qpos).all():self.error='Non-finite physical state';self.data.ctrl[:]=0

    def run_headless(self,seconds):
        for _ in range(round(seconds*PHYSICS_HZ)):
            with self.lock:self.step()

    def start(self):
        if self._thread and self._thread.is_alive():return
        self._stop.clear()
        self._thread=threading.Thread(target=self._loop,name='yard-physics',daemon=True);self._thread.start()
        self._sensor_thread=threading.Thread(target=self._sensor_loop,name='yard-sensors',daemon=True);self._sensor_thread.start()
        threading.Thread(target=self._load_language,name='yard-language',daemon=True).start()

    def _load_language(self):
        self.intent.start()
        try:self.voice.warmup()
        except RuntimeError as error:print(f'Voice startup: {error}',flush=True)

    def _sensor_loop(self):
        try:
            from .perception import VisionService
            self.vision=VisionService();self.missions.vision=self.vision
            self._perception_status=self.vision.status()
            if not self._perception_status['ready']:raise RuntimeError(self._perception_status['error'])
            with SensorRig(self.model) as rig:
                while not self._stop.is_set():
                    wall=time.monotonic()
                    with self.lock:
                        sample=copy.copy(self.data);scene=scene_state(self.model)
                        request=self._capture_request;self._capture_request=None
                        following=self.mode=='follow'
                    survey=rig.capture(sample,scene,'survey')
                    frame=rig.capture(sample,scene,'tracking') if following else survey
                    measurements=rig.ranges(sample)
                    detections=self.vision.detect_people(frame['image']) if following else []
                    tracks=[]
                    for detection in detections:
                        box=detection.get('box',detection.get('bbox'))
                        if not box:continue
                        x1,y1,x2,y2=box
                        u,v=(x1+x2)/2,(y1+y2)/2
                        location=unproject(frame,u,v)
                        if np.isfinite(location).all():
                            tracks.append({'id':'worker','type':'person','position':location.tolist(),
                                'confidence':float(detection.get('confidence',0)), 'box':box,
                                'time':frame['timestamp'],'source':detection.get('source','pretrained visual detector')})
                    encoded=io.BytesIO();Image.fromarray(frame['image']).save(encoded,format='JPEG',quality=85)
                    prepared=self.missions.prepare(request['key'],survey,request['position'],request['zone'],request['missionId']) if request else None
                    with self.lock:
                        self.navigator.observe_ranges(measurements);self.navigator.observe_frame(survey,cover=prepared is not None or self.mode=='follow')
                        if (self.trajectory is not None and self.destination is not None and
                                (self.mode in ['navigate','takeoff','landing','baseline','current'] or
                                 (self.mode=='follow' and not self._tracking_lost))):
                            remaining=[self.goal,*self.waypoints]
                            if not self.navigator.path_safe(self.data.sensordata[:3],remaining):
                                try:
                                    self._route(self.destination,landing=self.mode=='landing')
                                    self.last_command='Replanned around a newly observed obstacle'
                                except ValueError:
                                    self._apply_action({'action':'hover'})
                                    self.last_command='New obstacle blocks the route; holding position'
                        self._navigation_state=self.navigator.snapshot()
                        self._camera_jpeg=encoded.getvalue();self._camera_time=frame['timestamp'];self._camera_wall=wall
                        self._sensor_frames+=1;self.sensor_error=None;self._perception_status=self.vision.status()
                        if tracks and self.mode=='follow':self._tracks=tracks;self._person=max(tracks,key=lambda d:d['confidence'])
                        if prepared:
                            accepted=self.missions.record(prepared,self.navigator.snapshot()['coveragePercent'])
                            if request['missionId']==self.missions.mission['id']:self._capture_busy=False
                            if accepted:
                                self.navigator.visited.append(self._scan_goal.copy());self._next_scan()
                    self._stop.wait(max(.02,.6-(time.monotonic()-wall)))
        except Exception as error:
            self.sensor_error=f'{type(error).__name__}: {error}'
            print(f'Sensor loop stopped: {self.sensor_error}',flush=True)
            with self.lock:
                if self.motors=='on':self._apply_action({'action':'hover'})

    def _loop(self):
        wall0,sim0=time.monotonic(),self.elapsed
        try:
            while not self._stop.is_set():
                wall=time.monotonic();due=sim0+(wall-wall0)*self.speed
                if due-self.elapsed>.25:self.late_resyncs+=1;wall0,sim0=wall,self.elapsed;continue
                if self.elapsed>=due:self._stop.wait(.0015);continue
                with self.lock:
                    for _ in range(25):
                        if self.elapsed>=due:break
                        self.step()
                    self.rtf_window.append((wall,self.elapsed))
        except Exception as error:self.error=f'Physics stopped: {type(error).__name__}: {error}'

    def close(self):
        self._stop.set()
        for thread in [self._thread,self._sensor_thread]:
            if thread:thread.join(timeout=3)
        self.intent.close()

    def realtime_factor(self):
        if len(self.rtf_window)<2:return 0.
        (w0,s0),(w1,s1)=self.rtf_window[0],self.rtf_window[-1]
        return (s1-s0)/max(w1-w0,1e-6)

    def command(self,text):
        try:actions=self.intent.interpret(text)
        except ValueError as error:return {'accepted':False,'message':str(error),'command':text}
        with self.lock:
            try:
                self._actions.clear();self._plan=actions
                message=self._apply_action(actions[0]);self._actions.extend(actions[1:]);self.last_command=message
                return {'accepted':True,'message':message,'command':text,'plan':actions,'interpreter':self.intent.status()}
            except ValueError as error:return {'accepted':False,'message':str(error),'command':text,'plan':actions}

    def action(self,action):
        with self.lock:
            try:
                self._actions.clear();message=self._apply_action(action);self.last_command=message
                return {'accepted':True,'message':message}
            except ValueError as error:return {'accepted':False,'message':str(error)}

    def _apply_action(self,item):
        if self.error:raise ValueError(self.error)
        action=item.get('action')
        if action=='reject':raise ValueError('That request is outside this drone’s capabilities.')
        if action=='inject':return self.environment.inject(item.get('hazard'),item.get('position'))['message']
        if action=='clear':self.environment.clear();return 'Cleared the simulated incidents.'
        if action in ['baseline','survey']:
            if self._camera_time is None or self.sensor_error:raise ValueError('The onboard camera must be ready before a survey.')
            kind='baseline' if action=='baseline' else ('current' if self.missions.baseline['available'] else 'baseline')
            self._cancel_mission()
            self.missions.begin(kind);self.navigator.covered[:]=False;self.navigator.visited=[]
            self._scan_remaining=list(self.missions.baseline['observations']) if kind=='current' else []
            self.mode=kind;self._next_scan();return f'Started {kind} mission with camera coverage planning.'
        self._cancel_mission()
        if action=='goto':
            zone=item.get('zone');goal=HOME if zone=='home' else ZONES.get(zone,{}).get('position')
            if goal is None:raise ValueError('Choose zone A, B, C, D or home.')
            if zone=='home':goal=[HOME[0],HOME[1],CRUISE_HEIGHT]
            self.mode='navigate';self._route(np.array(goal));return f'Flying to {zone} using the measured obstacle map.'
        if action=='takeoff':
            self.mode='takeoff';self._route(np.array([*self.data.sensordata[:2],CRUISE_HEIGHT]));return 'Taking off to 4.5 metres.'
        if action=='follow':
            if self._camera_time is None or self.sensor_error:raise ValueError('The camera must be ready to follow a worker.')
            self.mode='follow';self.navigator.covered[:]=False;self.navigator.visited=[];self._person=None;self._tracks=[]
            self._tracking_started=False;self._tracking_lost=False
            self._route(np.array([*self.data.sensordata[:2],CRUISE_HEIGHT]));return 'Searching the onboard camera for the worker.'
        if action=='hover':
            if self.motors=='off':self.mode='idle';return 'Holding on the ground.'
            self.mode='hover';self.waypoints=[];self.destination=self.reference+.6*self.reference_velocity;self._segment(self.destination);return 'Braking and holding position.'
        if action=='land':
            p=self.data.sensordata[:3]
            if np.linalg.norm(p[:2]-np.array(HOME[:2]))<.2 and p[2]<.2:
                self.mode,self.motors='idle','off';self.trajectory=None;self.waypoints=[];return 'Already on the home pad.'
            self.mode='landing';self._route(np.array(HOME),landing=True);return 'Returning to the home pad and landing.'
        if action=='motors_off':
            self.mode,self.motors='idle','off';self.trajectory=None;self.waypoints=[];return 'Motors off; gravity remains active.'
        raise ValueError('Unsupported flight action.')

    def move_worker(self,dx=0,dy=0,stop=False):
        with self.lock:
            if stop:self.environment.worker.stop();return {'accepted':True,'message':'Worker holding position.'}
            return self.environment.move_worker(dx,dy)

    def snapshot(self):
        p=self.data.xpos[self.body_id];v=self.data.sensordata[3:6];environment=self.environment.snapshot()
        age=time.monotonic()-self._camera_wall if self._camera_wall else None
        return {'ready':self.error is None and self.steps>0,'error':self.error,
            'time':round(float(self.data.time),4),'elapsed':round(self.elapsed,4),'position':np.round(p,4).tolist(),
            'quaternion':np.round(self.data.xquat[self.body_id][[1,2,3,0]],5).tolist(),
            'velocity':np.round(v,4).tolist(),'rotors':np.round(self.data.act,4).tolist(),
            'target':np.round(self.goal,4).tolist(),'reference':np.round(self.reference,4).tolist(),'mode':self.mode,
            **environment,'environment':environment,'activeHazards':[],**self.missions.snapshot(),
            'sensor':{'cameraUrl':'/api/camera.jpg','frameTime':self._camera_time,'ageSeconds':age,
                      'frames':self._sensor_frames,'ready':self._camera_time is not None,'error':self.sensor_error},
            'navigation':{**self._navigation_state,'path':self.navigator.path,'replans':self.navigator.replans},'perception':{**self._perception_status,'tracks':self._tracks},
            'voice':self.voice.status(),'language':self.intent.status(),'commandPlan':self._plan,
            'telemetry':{'altitude':round(float(p[2]),3),'speed':round(float(np.linalg.norm(v)),3),
                'trackingError':round(self.error_now,4),'thrust':round(float(self.data.act.sum()),3),
                'hoverThrust':round(MASS*GRAVITY,3),'motors':self.motors,'physicsHz':PHYSICS_HZ,
                'controlHz':CONTROL_HZ,'realtimeFactor':round(self.realtime_factor(),3),
                'scanZone':self.mission_zone,'lastCommand':self.last_command,'obstacleContacts':self.obstacle_contacts}}

    def status(self):
        with self.lock:
            return {'ready':self.error is None and self.steps>0 and self._camera_time is not None and
                    self.intent.ready and self.voice.status()['ready'] and not self.sensor_error,
                'error':self.error or self.sensor_error,'engine':'MuJoCo','mujocoVersion':mujoco.__version__,
                'physicsHz':PHYSICS_HZ,'controlHz':CONTROL_HZ,'integrator':'RK4','massKg':MASS,
                'rotorMaxThrustN':ROTOR_MAX_THRUST,'time':round(float(self.data.time),3),'mode':self.mode,
                'realtimeFactor':round(self.realtime_factor(),3),'lateResyncs':self.late_resyncs,
                'obstacleContacts':self.obstacle_contacts,'machine':platform.machine(),
                'sensor':{'ready':self._camera_time is not None,'frames':self._sensor_frames,'error':self.sensor_error},
                'language':self.intent.status(),'voice':self.voice.status(),'perception':self._perception_status}
