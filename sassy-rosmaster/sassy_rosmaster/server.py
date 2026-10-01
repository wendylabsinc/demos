"""HTTP controls and the shared simulator/hardware control loop."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import json
import logging
import math
import os
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .audio import RobotAudio
from .navigation import ENCOUNTER_DISTANCE, Navigator, Personality, validate_observation
from .types import Command
from .speech import Speech

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger('sassy-rosmaster')
SIM_CAMERA_MAX_AGE = 1.2  # software-rendered semantics; lidar braking stays independent


class Rover:
    def __init__(self):
        self.mode = os.getenv('ROVER_MODE', 'simulation')
        if self.mode == 'hardware':
            from .hardware import Hardware
            config = json.loads(Path(os.environ['ROVER_CONFIG']).read_text())
            self.backend = Hardware(config)
            geometry = self.backend.geometry
            radius = math.hypot(geometry['length']/2, geometry['width']/2) + .12
            self.navigator_args = dict(max_speed=min(.3, self.backend.max_speed),
                                       wheelbase=geometry['wheelbase'], body_radius=radius,
                                       max_steering=self.backend.max_steering)
        elif self.mode == 'simulation':
            from .simulation import Simulation
            seed = os.getenv('ROVER_SEED')
            self.backend = Simulation(seed=int(seed) if seed is not None else None)
            self.navigator_args = {}
        else:
            raise ValueError('ROVER_MODE must be simulation or hardware')
        self.navigator = Navigator(**self.navigator_args)
        self.personality = Personality()
        self.audio = RobotAudio()
        self.speech = Speech()
        self.explorer = self.speech.explorer()
        self.perception = None
        self.camera_epoch = time.monotonic()
        if self.mode == 'simulation':
            from .perception import CameraPerception, MuJoCoCameraSource
            self.perception = CameraPerception(MuJoCoCameraSource(self.backend))
            self.perception.start()
        self.camera_state = {}
        self.speech_target = None
        self.speech_last_seen = -math.inf
        self._closed = False
        self.intention = 'Observing the room.'
        self.goal_context = {}
        self.goal_requested_at = 0.
        self.last_deliberation_attempt = -math.inf
        self.running, self.fault = False, None
        self.command, self.observation = Command(), None
        self.events, self.event_id = [], 0
        self.distance, self.collisions = 0., 0
        self.last_pose, self.last_collision = None, False
        self.started_at, self.deadline = None, None
        self.last_save = 0.
        self.data_dir = Path(os.getenv('ROVER_DATA', str(ROOT/'media')))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.event('system', 'ROSMASTER online. Waiting for operator start.')

    def event(self, kind, text):
        self.event_id += 1
        self.events.append(dict(id=self.event_id, time=time.time(), kind=kind, text=text))
        self.events = self.events[-60:]
        log.info('%s: %s', kind, text)
        self.audio.emit(kind, text)

    def stop(self, reason='Stopped by operator'):
        self.running, self.command = False, Command()
        self.speech_target = None
        self.speech.cancel_pending()
        self.explorer.cancel_pending()
        if self.mode == 'hardware':
            self.backend.stop()
        else:
            self.backend.step(Command(), .05)
        self.navigator.status = reason

    @staticmethod
    def front_blockers(obs):
        return [person for person in obs.people
                if 0 < person.get('distance', 100) < ENCOUNTER_DISTANCE
                and abs(person.get('bearing', math.pi)) < .7]

    def tick(self, dt=.05):
        now = time.monotonic()
        try:
            obs = self.backend.read()
            validate_observation(obs)
            if self.perception is not None:
                self.camera_state = self.perception.snapshot()
                source_time = self.camera_state.get('captured_at') or 0.
                people = []
                if self.camera_epoch <= source_time <= time.monotonic() and time.monotonic()-source_time < SIM_CAMERA_MAX_AGE:
                    for person in self.camera_state.get('people', []):
                        dx, dy = person['x']-obs.pose.x, person['y']-obs.pose.y
                        people.append(dict(person, distance=math.hypot(dx, dy),
                                           bearing=math.atan2(math.sin(math.atan2(dy, dx)-obs.pose.yaw),
                                                              math.cos(math.atan2(dy, dx)-obs.pose.yaw))))
                obs = replace(obs, people=people)
            self.observation = obs
            if self.last_pose:
                self.distance += math.hypot(obs.pose.x-self.last_pose.x, obs.pose.y-self.last_pose.y)
            self.last_pose = obs.pose
            if obs.collision and not self.last_collision:
                self.collisions += 1
            self.last_collision = obs.collision
            self.navigator.observe(obs, dt)
            blockers = self.front_blockers(obs)
            speech_relevant = self.running and bool(blockers)
            poll_speech = True
            if self.speech_target is not None:
                target_id = self.speech_target.get('id')
                if target_id is not None:
                    # A measured encounter can remain relevant while we back
                    # away. Still require this same person in the current view.
                    speech_relevant = self.running and any(person.get('id') == target_id
                        and 0 < person['distance'] < 3.2 and abs(person['bearing']) < .9
                        for person in obs.people)
                explicitly_left = target_id is not None and any(person.get('id') == target_id for person in obs.people)
                if speech_relevant:
                    self.speech_last_seen = now
                elif self.running and target_id is not None and not explicitly_left and now-self.speech_last_seen < .8:
                    # A missed box can be temporary. Hold an unemitted reply
                    # briefly, and speak only after this person is seen again.
                    poll_speech = False
                else:
                    self.speech.cancel_pending()
                    self.speech_target = None
            speech_results = self.speech.poll(allow_result=speech_relevant) if poll_speech else []
            for result in speech_results:
                if self.running and result.get('text'):
                    self.event('speech', result['text'])
            if speech_results:
                self.speech_target = None
            for result in self.explorer.poll(allow_result=self.running):
                candidate = self.goal_context.get(result.get('goal_id'))
                if self.running and candidate and now-self.goal_requested_at < 8 and self.navigator.accept_goal(candidate, obs):
                    self.intention = result['reason']
                    self.event('decision', self.intention)
            if self.running and self.deadline and now >= self.deadline:
                self.stop('Timed run complete. Brakes on.')
                self.event('system', 'Run time limit reached. Stopped.')
            if self.running:
                if obs.collision:
                    raise RuntimeError('Bumper contact detected. Stopped.')
                hold, events = self.personality.update(obs, now)
                for kind, text in events:
                    self.event(kind, text)
                    if kind == 'horn' and blockers:
                        closest = min(blockers, key=lambda person: person['distance'])
                        if self.speech.request(dict(sass=self.personality.level,
                                                 nearest_person_distance=round(closest['distance'], 2),
                                                 speed=round(self.command.speed, 2),
                                                 coverage=self.navigator.map.export()['coverage'],
                                                 distance=round(self.distance, 1),
                                                 navigation_status=self.navigator.status,
                                                 people=[dict(distance=p['distance'], bearing=p['bearing']) for p in blockers])):
                            self.speech_target = {'id': closest.get('id')}
                            self.speech_last_seen = now
                if now-self.last_deliberation_attempt >= 1. and not self.navigator.yielding and not self.navigator.recovering:
                    self.last_deliberation_attempt = now
                    candidates = [dict(candidate, id=i) for i, candidate in enumerate(self.navigator.goal_candidates(obs)[:8])]
                    if candidates and self.explorer.request(dict(candidates=candidates,
                                      coverage=self.navigator.map.export()['coverage'], recent_goal=self.navigator.goal)):
                        self.goal_context = {item['id']: item for item in candidates}
                        self.goal_requested_at = now
                self.command = Command() if hold else self.navigator.command(obs, dt)
                if hold:
                    self.navigator.status = 'Person ahead. Honking with unreasonable confidence.'
                self.backend.step(self.command, dt)
            elif self.mode == 'simulation':
                self.backend.step(Command(), dt)
            if now-self.last_save > 5:
                self.save_map()
                self.last_save = now
        except Exception as exc:
            message = str(exc)
            changed = self.fault != message
            self.fault = message
            try:
                self.stop('Sensor or control fault. Brakes on.')
            except Exception:
                log.exception('Could not send zero command')
            if changed:
                self.event('system', message)

    def control(self, action, value=None, duration=120):
        if action == 'start':
            if self.fault:
                raise ValueError('Reset the fault before starting: '+self.fault)
            validate_observation(self.backend.read())
            if self.mode == 'hardware':
                self.backend.start()
            self.running, self.started_at = True, time.time()
            self.deadline = time.monotonic()+duration if self.mode == 'hardware' else None
            self.event('system', 'Wandering started. No destination. Plenty of opinions.')
        elif action == 'stop':
            self.stop()
            self.event('system', 'Brakes on. Waiting for an explicit start.')
        elif action == 'estop':
            self.stop('Emergency stop. Brakes on.')
            if self.mode == 'hardware':
                self.backend.emergency_stop()
                self.fault = 'Operator e-stop asserted. Reset before Start.'
            self.event('system', 'Emergency stop applied.')
        elif action == 'reset':
            self.stop()
            if self.mode == 'simulation':
                self.backend.reset()
                self.camera_epoch = time.monotonic()
                self.camera_state = {}
                self.goal_context = {}
                self.intention = 'Observing a new room.'
                self.navigator = Navigator(**self.navigator_args)
                self.distance, self.collisions = 0., 0
                self.last_pose, self.last_collision = None, False
                level = self.personality.level
                self.personality = Personality()
                self.personality.level = level
                self.observation = self.backend.read()
            else:
                self.backend.reset_estop()
                validate_observation(self.backend.read())
            self.fault = None
            self.event('system', 'Reset complete. Ready for Start.')
        elif action in ('person', 'clear_people'):
            if self.mode != 'simulation':
                raise ValueError('Pedestrian controls are available in simulation')
            if action == 'person':
                if not self.backend.add_person(ahead=True):
                    raise ValueError('No safe place ahead for a pedestrian. Try after moving.')
                self.event('system', 'Pedestrian stepped into the route.')
            else:
                self.backend.clear_people()
                self.camera_epoch = time.monotonic()
                self.speech.cancel_pending()
                self.speech_target = None
                if self.observation:
                    self.observation = replace(self.observation, people=[])
                self.event('system', 'Pedestrians cleared.')
        elif action == 'sass':
            self.personality.level = int(value or 2)
        else:
            raise ValueError('Unknown control')

    def state(self):
        obs, scan = self.observation, []
        if obs:
            for i, distance in enumerate(obs.ranges):
                angle = obs.pose.yaw+obs.angle_min+i*obs.angle_increment
                scan.append(dict(x=obs.pose.x+distance*math.cos(angle), y=obs.pose.y+distance*math.sin(angle)))
        scene = self.backend.scene()
        if self.mode == 'hardware':
            scene['car'] = dict(x=obs.pose.x if obs else 0, y=obs.pose.y if obs else 0,
                                yaw=obs.pose.yaw if obs else 0, **self.backend.geometry)
            scene['people'] = [dict(p, id=i) for i, p in enumerate(obs.people)] if obs else []
        return dict(mode=self.mode, running=self.running, fault=self.fault,
                    pose=asdict(obs.pose) if obs else dict(x=0,y=0,yaw=0),
                    speed=self.command.speed, steering=self.command.steering,
                    status=self.navigator.status, distance=round(self.distance,2),
                    map=self.navigator.map.export(), scene=scene, scan=scan,
                    observed_people=obs.people if obs else [],
                    events=self.events, sass=self.personality.level, collisions=self.collisions,
                    started_at=self.started_at, goal=self.navigator.goal,
                    intention=self.intention, sensors=dict(camera=self.camera_status(),
                    speech=self.model_status(self.speech), exploration=self.model_status(self.explorer)))

    @staticmethod
    def model_status(job):
        status = job.status()
        return dict(status, model=status.get('actual_model') or status.get('requested_model'))

    def camera_status(self):
        snapshot = self.camera_state
        fields = ('ready', 'state', 'error', 'model', 'frame_id', 'captured_at', 'inference_ms', 'render_ms', 'detect_ms')
        result = {key: snapshot.get(key) for key in fields}
        result['detected_people'] = len(self.observation.people) if self.observation else 0
        capture = snapshot.get('captured_at') or 0.
        age = time.monotonic()-capture
        result['age_ms'] = round(age*1000) if capture else None
        if result.get('ready') and (capture < self.camera_epoch or not 0 <= age < SIM_CAMERA_MAX_AGE):
            result.update(ready=False, state='stale', error='Camera observation expired. Waiting for a fresh frame.')
        if self.perception is None:
            result.update(state='external ROS camera bridge', ready=False)
        return result

    def save_map(self):
        target = self.data_dir/'map.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.navigator.map.export()))
        temporary.replace(target)

    def close(self):
        if self._closed:
            return
        self._closed = True
        cleanup = [('stop', lambda: self.stop('App shut down'))]
        if self.perception is not None:
            cleanup.append(('camera', self.perception.close))
        cleanup.extend([('exploration model', self.explorer.close),
                        ('speech model', self.speech.close), ('map save', self.save_map)])
        if hasattr(self.backend,'close'):
            cleanup.append(('backend', self.backend.close))
        cleanup.append(('audio', self.audio.close))
        for name, action in cleanup:
            try:
                action()
            except Exception:
                log.exception('Failed to close %s; continuing shutdown cleanup', name)


@asynccontextmanager
async def lifespan(app):
    app.state.rover = Rover()
    async def loop():
        while True:
            started = time.monotonic()
            app.state.rover.tick()
            await asyncio.sleep(max(.001,.05-(time.monotonic()-started)))
    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        app.state.rover.close()

app = FastAPI(title='Sassy ROSMASTER',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

@app.get('/')
async def index():
    return FileResponse(ROOT/'static'/'index.html')

@app.get('/health')
async def health():
    rover=app.state.rover
    return dict(status='ok' if not rover.fault else 'fault',mode=rover.mode,
                running=rover.running,sensor_ready=rover.observation is not None,fault=rover.fault,
                camera=rover.camera_status(), speech=rover.model_status(rover.speech)['state'],
                exploration=rover.model_status(rover.explorer)['state'])

@app.get('/api/state')
async def state():
    return app.state.rover.state()

@app.get('/api/camera')
async def camera():
    rover = app.state.rover
    snapshot = rover.perception.snapshot() if rover.perception else {}
    jpeg = snapshot.get('jpeg')
    if not jpeg:
        raise HTTPException(503, snapshot.get('error') or 'Waiting for a camera frame')
    return Response(jpeg, media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

class Control(BaseModel):
    action: str
    value: int | None = Field(None,ge=1,le=3)
    duration: int = Field(120,ge=1,le=120)

@app.post('/api/control')
async def control(body:Control):
    try:
        app.state.rover.control(body.action,body.value,body.duration)
    except (ValueError,RuntimeError) as exc:
        raise HTTPException(409,str(exc)) from exc
    return app.state.rover.state()

@app.get('/api/map')
async def download_map():
    return Response(json.dumps(app.state.rover.navigator.map.export(),indent=2),
                    media_type='application/json',
                    headers={'Content-Disposition':'attachment; filename="rosmaster-map.json"'})
