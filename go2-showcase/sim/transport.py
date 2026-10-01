"""MuJoCo transport. Robot state changes only via policy torques and mj_step.

Localization is exact simulated pose. Vision is raw egocentric RGB. Sensor
apis expose no object labels or target coordinates. Scene actors move only
when an operator changes the scene.
"""
import math
import time
import mujoco
import numpy as np
from .backend import SimBackend, _angle

class _OwnedSimTransport:
    profile = 'sim'
    def __init__(self):
        self.backend = SimBackend()
        self.command = None
        self.armed = False
        self.name = 'MuJoCo Go2 / pinned ONNX torque policy'
        self.barrier_id = int(self.backend.sim.model.body('demo_barrier').mocapid[0])
        self.last_scan = []
        self.last_reason = 'Waiting for Start'
        self.gesture = None
        self._arm_geometry = {}
        for side in ('left','right'):
            for part in ('sleeve','arm'):
                geom=self.backend.sim.model.geom(f'person_{side}_{part}')
                self._arm_geometry[geom.id]=(geom.pos.copy(),geom.quat.copy(),geom.size.copy())
    def arm(self):
        self.armed = True
    def stop(self):
        self.armed = False
        self.command = None
        self.backend.stop()
    def close(self):
        self.backend.close()
    def send(self, command):
        self.command = command
        self.last_reason = command.reason
    def advance(self, dt=0.05):
        b = self.backend
        if self.armed and self.command:
            c = self.command
            velocity = (float(c.vx), float(c.vy), float(c.yaw_rate))
        else:
            velocity = (0., 0., 0.)
        for _ in range(round(dt / .002)):
            b._step(velocity)
        if b.elapsed - b.last_capture >= .10:
            b._capture()
    def read(self):
        from showcase.models import Observation
        b = self.backend
        if b.elapsed - b.last_capture >= .10:
            b._capture()
        pose = b._pose()
        model, data = b.sim.model, b.sim.data
        origin = data.site_xpos[model.site('lidar').id]
        rotation = data.xmat[b.sim.body_id].reshape(3,3)
        group = np.array([1,0,0,0,0,0], dtype=np.uint8)
        geom = np.empty(1, dtype=np.int32)
        obstacles = []
        sectors = []
        front = 12.
        for angle in np.linspace(-math.pi,math.pi,72,endpoint=False):
            direction = rotation @ np.array([math.cos(angle),math.sin(angle),0.])
            distance = mujoco.mj_ray(model,data,origin,direction,group,True,b.sim.body_id,geom)
            if geom[0]>=0 and model.geom(int(geom[0])).name == "floor": distance=-1
            sectors.append((float(angle),float(distance) if distance>0 else 12.))
            if .10 < distance < 5.:
                point = origin + direction * distance
                obstacles.append((float(point[0]),float(point[1]),.08))
            if abs(angle)<.4 and distance>0:
                front=min(front,float(distance))
        self.last_scan=obstacles
        healthy = b.sim.mode not in {'fallen','fault','damping'} and b.camera_healthy
        return Observation(pose['x'],pose['y'],pose['yaw'],fresh=healthy,obstacles=tuple(obstacles),detections=(),gesture=None,front_clearance=front,scan_fresh=healthy,sectors=tuple(sectors))
    def camera_jpeg(self):
        return self.frame_jpeg('robot')
    def sensor_snapshot(self):
        obs=self.read()
        return {'capture_id': self.backend.capture_id,
                'captured_at_monotonic': self.backend.capture_time,
                'healthy': obs.fresh and obs.scan_fresh,
                'pose': {'x':obs.x,'y':obs.y,'yaw':obs.yaw},
                'front_clearance': obs.front_clearance,
                'scan_sectors':list(obs.sectors),
                'localization_source':'exact simulated localization',
                'camera_source':'egocentric rendered RGB, no object labels or coordinates'}
    def set_object(self,x,y):
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and abs(v)<=3.4 for v in (x,y)):
            raise ValueError('object position requires finite coordinates inside the arena')
        b=self.backend
        b.sim.data.mocap_pos[b.key_id]=[x,y,.60]
        b.sim.data.mocap_pos[b.table_id]=[x,y,.35]
        mujoco.mj_forward(b.sim.model,b.sim.data)
        b.last_capture=-float('inf')
    def randomize_scene(self,seed=None):
        rng=np.random.default_rng(seed)
        xy=rng.uniform([-3.,1.2],[3.,3.1]); self.set_object(float(xy[0]),float(xy[1]))
        xy=rng.uniform([-3.,-3.],[3.,3.]); self.set_person(float(xy[0]),float(xy[1]))
        self.set_barrier(float(rng.uniform(-1,1)),float(rng.uniform(1.,3.)))
        return {'scene':'randomized Go2 arena','semantics':'camera pixels only','localization':'exact simulated pose'}
    def set_barrier(self, x, y):
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and abs(v)<=4 for v in (x,y)):
            raise ValueError('barrier requires finite coordinates inside the arena')
        self.backend.sim.data.mocap_pos[self.barrier_id]=[x,y,.4]
        mujoco.mj_forward(self.backend.sim.model,self.backend.sim.data)
    def set_person(self,x,y):
        self.backend.set_person(x,y)
    def set_person_pose(self,pose):
        if pose not in {'neutral','left','right','both'}:
            raise ValueError('person pose must be neutral, left, right or both')
        model=self.backend.sim.model
        for geom_id,(position,quaternion,size) in self._arm_geometry.items():
            model.geom_pos[geom_id]=position; model.geom_quat[geom_id]=quaternion; model.geom_size[geom_id]=size
        for side,y in (('left',.25),('right',-.25)):
            if pose not in {side,'both'}: continue
            for part,a,b in [('sleeve',np.array([0,y,.32]),np.array([0,y,.62])),('arm',np.array([0,y,.62]),np.array([0,y,.95]))]:
                geom=model.geom(f'person_{side}_{part}')
                geom.pos[:]=(a+b)/2;geom.quat[:]=[1,0,0,0];geom.size[1]=float(np.linalg.norm(b-a)/2)
        mujoco.mj_forward(model,self.backend.sim.data)
        self.backend.last_capture=-float('inf')
    def frame_jpeg(self, view='overview'):
        from io import BytesIO
        out=BytesIO(); self.frame(view).save(out,format='JPEG',quality=85); return out.getvalue()
    def configure_scene(self, payload):
        if payload.get('randomize') or 'seed' in payload:
            self.randomize_scene(payload.get('seed'))
        if 'person' in payload:
            p=payload['person']; self.set_person(p[0],p[1])
        if 'person_pose' in payload: self.set_person_pose(payload['person_pose'])
        if 'barrier' in payload:
            p=payload['barrier']; self.set_barrier(p[0],p[1])
        if 'object' in payload:
            p=payload['object']; self.set_object(p[0],p[1])
        return {'scene':'Go2 arena','perception':'raw egocentric RGB; no semantic observations','localization':'exact simulated pose'}
    def status(self):
        return {**self.backend.status(), 'profile':'sim','armed':self.armed,'reason':self.last_reason,'source':'exact MuJoCo pose; raw egocentric RGB; raycast scan', 'physics_steps': self.backend.sim._steps}
    def frame(self,view='overview'):
        return self.backend.frame(view)


class SimTransport:
    """Thread-safe facade. One worker owns physics and all OpenGL contexts."""
    profile = 'sim'
    def __init__(self):
        import queue, threading
        from concurrent.futures import Future
        self._queue=queue.Queue(); self._ready=Future()
        def owner():
            try: self._owned=_OwnedSimTransport(); self._ready.set_result(None)
            except BaseException as e: self._ready.set_exception(e); return
            while True:
                name,args,kwargs,future=self._queue.get()
                try:
                    if name=='get': result=getattr(self._owned,args[0])
                    elif name=='set': result=setattr(self._owned,args[0],args[1])
                    else: result=getattr(self._owned,name)(*args,**kwargs)
                    future.set_result(result)
                except BaseException as e: future.set_exception(e)
                if name=='close': break
        self._thread=threading.Thread(target=owner,daemon=True,name='go2-physics-render-owner')
        self._thread.start(); self._ready.result(timeout=30)
    def _call(self,name,*args,**kwargs):
        from concurrent.futures import Future
        future=Future(); self._queue.put((name,args,kwargs,future)); return future.result(timeout=30)
    def read(self): return self._call('read')
    def send(self,command): return self._call('send',command)
    def arm(self): return self._call('arm')
    def stop(self): return self._call('stop')
    def close(self): return self._call('close')
    def advance(self,dt=.05): return self._call('advance',dt)
    def status(self): return self._call('status')
    def frame(self,view='overview'): return self._call('frame',view)
    def camera_jpeg(self): return self._call('camera_jpeg')
    def sensor_snapshot(self): return self._call('sensor_snapshot')
    def set_object(self,x,y): return self._call('set_object',x,y)
    def randomize_scene(self,seed=None): return self._call('randomize_scene',seed)
    def frame_jpeg(self,view='overview'): return self._call('frame_jpeg',view)
    def configure_scene(self,payload): return self._call('configure_scene',payload)
    def set_person(self,x,y): return self._call('set_person',x,y)
    def set_person_pose(self,pose): return self._call('set_person_pose',pose)
    def set_barrier(self,x,y): return self._call('set_barrier',x,y)
    @property
    def gesture(self): return self._call('get','gesture')
    @gesture.setter
    def gesture(self,value): self._call('set','gesture',value)


def build_sim_transport():
    return SimTransport()
