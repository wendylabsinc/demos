"""Virtual LiDAR and drifting odometry. Only this sensor adapter sees MuJoCo state.

AMCL runs in a separate ROS container and receives measurements, never truth.
"""
import time,math,json,urllib.request
import numpy as np,mujoco

class LocalizedRobot:
    def __init__(self,dog,url):
        self.dog=dog;self.url=url.rstrip('/');self.odom=np.zeros(3);self.pose=None
        self.rng=np.random.default_rng(83);self.tick=0;self.errors=[];self.started=False
        self.data=dog.data;self.model=dog.model
        self.trunk=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'base')
        if self.trunk<0:raise ValueError('No robot base body for virtual sensors')
        self.last_time=dog.data.time;self.rejections=[]
        self.next_wall=time.monotonic()
    def request(self,path,body=None):
        request=urllib.request.Request(self.url+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=2) as r:return json.loads(r.read() or '{}')
    def update(self):
        time.sleep(max(0,self.next_wall-time.monotonic()))
        self.next_wall=max(self.next_wall+.02,time.monotonic())
        # XBODY is the body frame. BODY uses the principal-inertia frame, whose
        # axes differ from the Go2's sensor mounting axes.
        velocity=np.zeros(6);mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_XBODY,self.trunk,velocity,1)
        dt=self.data.time-self.last_time;self.last_time=self.data.time
        # Measured local velocity and angular rate, with deliberate scale/bias errors.
        self.odom[2]+=dt*(velocity[2]+.008)
        c,s=math.cos(self.odom[2]),math.sin(self.odom[2])
        self.odom[:2]+=dt*1.08*np.array([c*velocity[3]-s*velocity[4],s*velocity[3]+c*velocity[4]])
        # The ROS bridge consumes at 10 Hz; leave headroom in its bounded queue.
        if self.tick%6==0:
            # Range generation belongs to the simulator's sensor model. Truth is not
            # sent to AMCL or made available through this robot's navigation properties.
            origin=np.array([*self.dog.base_xy,.8]);heading=self.dog.base_yaw
            ranges=[];gid=np.array([-1],np.int32)
            for a in np.linspace(-math.pi,math.pi,360,endpoint=False):
                direction=np.array([math.cos(a+heading),math.sin(a+heading),0.])
                r=mujoco.mj_ray(self.model,self.data,origin,direction,np.array([1,0,0,1,0,0],np.uint8),1,self.trunk,gid)
                ranges.append(float(r+self.rng.normal(0,.015)) if .35<r<=12 else None)
            self.request('/measurements',dict(capture_time=time.time(),odom=self.odom.tolist(),ranges=ranges))
            if not self.started:
                # Operator-like rough bay estimate. AMCL must correct this offset.
                self.request('/initialpose',dict(x=.3,y=-.25,yaw=.12,variance=[.25,.25,.09]));self.started=True
            status=self.request('/status')
            self.pose=status.get('pose') if status.get('ready') else None
            if self.pose:
                self.errors.append(dict(time=float(self.data.time),estimated=[self.pose['x'],self.pose['y'],self.pose['yaw']],
                    truth=[*self.dog.base_xy,self.dog.base_yaw],odometry=self.odom.tolist()))
            else:self.rejections.append(dict(time=float(self.data.time),status=status))
        self.tick+=1
    @property
    def ready(self):return self.pose is not None
    @property
    def base_xy(self):
        if not self.ready:raise RuntimeError('Localization unavailable')
        return self.pose['x'],self.pose['y']
    @property
    def base_yaw(self):
        if not self.ready:raise RuntimeError('Localization unavailable')
        return self.pose['yaw']
    def set_command(self,*args):self.dog.set_command(*args if self.ready else (0,0,0))
