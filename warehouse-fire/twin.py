"""MuJoCo warehouse geometry, calibrated thermal rays, and water dynamics."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

WIDTH, HEIGHT = 32, 24
RACKS = [(x, y) for x in (10, 20, 30) for y in (8, 16, 24, 32)]
RAY_GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

def zone_at(position):
    x, y = position[:2]
    return min(7, max(0, int(y / 5))) * 8 + min(7, max(0, int(x / 5)))

def zone_center(zone):
    return np.array([2.5 + 5 * (zone % 8), 2.5 + 5 * (zone // 8), 6.5])

def build_model(path):
    root = ET.Element('mujoco', model='40m thermal warehouse')
    ET.SubElement(root, 'option', timestep='.01', gravity='0 0 -9.81')
    world = ET.SubElement(root, 'worldbody')
    def geom(name, kind, pos, size, color, **kw):
        return ET.SubElement(world, 'geom', name=name, type=kind, pos=' '.join(map(str, pos)), size=' '.join(map(str, size)), rgba=color, **kw)
    geom('floor', 'plane', (20,20,0), (20,20,.1), '.13 .19 .23 1')
    for name, pos, size in [('north',(20,40,2),(20,.1,2)),('west',(0,20,2),(.1,20,2)),('east',(40,20,2),(.1,20,2)),('south',(20,0,.25),(20,.1,.25))]:
        geom(name,'box',pos,size,'.27 .35 .40 1')
    ET.SubElement(world,'light',pos='20 20 12',dir='0 0 -1',diffuse='.8 .8 .8')
    for x,y in RACKS:
        # Opaque storage: thermal rays must respect these occluders.
        geom(f'storage_{x}_{y}','box',(x,y,1.25),(.8,2,1.25),'.48 .32 .20 1')
        for sx in (-1,1):
            for sy in (-1,1):
                geom(f'post_{x}_{y}_{sx}_{sy}','box',(x+sx*.9,y+sy*2.05,1.5),(.07,.07,1.5),'.17 .40 .51 1',group='1')
        for z in (.1,1.3,2.6):
            geom(f'rack_{x}_{y}_{z}','box',(x,y,z),(1,2.2,.05),'.22 .40 .48 1',group='1')
    camera_specs = []
    def camera(pos, target, label):
        name = f'T{len(camera_specs)+1:02d}'
        back = np.array(pos,dtype=float)-target; back /= np.linalg.norm(back)
        right = np.cross([0,1,0] if abs(back[2])>.99 else [0,0,1],back); right /= np.linalg.norm(right)
        up = np.cross(back,right)
        ET.SubElement(world,'camera',name=name,pos=' '.join(map(str,pos)),xyaxes=' '.join(map(str,np.r_[right,up])),fovy='55')
        geom(f'housing_{name}','box',pos,(.12,.12,.08),'.15 .80 .70 1',group='1',contype='0',conaffinity='0')
        camera_specs.append(dict(id=name,position=list(pos),label=label))
    for y in range(8):
        for x in range(8): camera((2.5+x*5,2.5+y*5,7),np.array([2.5+x*5,2.5+y*5,0]),'Ceiling')
    for x,y in RACKS:
        for side in (-1,1): camera((x+side*1.2,y,1.8),np.array([x+side*1.8,y,0]),'Rack aisle')
    for zone in range(64):
        geom(f'sprinkler_{zone}','cylinder',zone_center(zone),(.16,.08),'.30 .52 .68 1',group='1',contype='0',conaffinity='0')
        geom(f'zone_{zone}','box',(zone_center(zone)[0],zone_center(zone)[1],.012),(2.42,2.42,.01),'.08 .45 .70 .08',group='2',contype='0',conaffinity='0')
    # Visible effects do not participate in thermal ray tracing.
    for n in range(8):
        body = ET.SubElement(world,'body',name=f'flame_{n}',mocap='true',pos='0 0 -5')
        ET.SubElement(body,'geom',type='ellipsoid',size='.55 .55 1',rgba='1 .24 .04 .95',group='2',contype='0',conaffinity='0')
        marker = ET.SubElement(world,'body',name=f'marker_{n}',mocap='true',pos='0 0 -5')
        ET.SubElement(marker,'geom',type='cylinder',size='.8 .02',rgba='1 .1 .05 .85',group='2',contype='0',conaffinity='0')
    # Free joints and gravity give water drops real MuJoCo motion and contacts.
    for n in range(64):
        body = ET.SubElement(world,'body',name=f'drop_{n}',pos='0 0 -10')
        ET.SubElement(body,'freejoint')
        ET.SubElement(body,'geom',type='sphere',size='.08',mass='.002',rgba='.15 .65 1 .8',group='1',friction='.1 .1 .1')
    ET.ElementTree(root).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, camera_specs

class ThermalTwin:
    def __init__(self, path):
        self.model, self.cameras = build_model(path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.hits = []; self.valid = []
        u,v = np.meshgrid(np.arange(WIDTH)+.5,np.arange(HEIGHT)+.5)
        focal = HEIGHT/(2*math.tan(math.radians(55)/2))
        local = np.stack([(u-WIDTH/2)/focal,-(v-HEIGHT/2)/focal,-np.ones_like(u)],axis=-1).reshape(-1,3)
        for cid,cam in enumerate(self.cameras):
            origin = self.data.cam_xpos[cid].copy()
            rays = np.einsum("ij,kj->ik", local, self.data.cam_xmat[cid].reshape(3,3))
            rays /= np.linalg.norm(rays,axis=1)[:,None]
            points = np.full((WIDTH*HEIGHT,3),np.nan)
            gid = np.array([-1],np.int32)
            for p,ray in enumerate(rays):
                dist = mujoco.mj_ray(self.model,self.data,origin,ray,RAY_GROUPS,1,-1,gid)
                if dist>=0: points[p]=origin+ray*dist
            self.hits.append(points.reshape(HEIGHT,WIDTH,3))
            self.valid.append(np.isfinite(points[:,0]).reshape(HEIGHT,WIDTH))
        self.hits = np.array(self.hits)
        self.valid = np.array(self.valid)
        self.rng = np.random.default_rng(42)

    def thermal_frames(self, sources):
        """Radiometric sensor adapter. The detector gets frames, never source poses."""
        frames = np.full(self.valid.shape,22.,dtype=float)
        for source in sources:
            if source['temperature']<=22: continue
            delta=self.hits-np.array(source['position'])
            dist2=np.nansum(delta*delta,axis=-1)
            heat=(source['temperature']-22)*np.exp(-dist2/(2*.75**2))
            frames += np.where(self.valid,heat,0)
        frames += self.rng.normal(0,.35,frames.shape)
        frames[~self.valid]=22
        return frames

    def surface(self,x,y):
        gid=np.array([-1],np.int32); origin=np.array([x,y,7.])
        distance=mujoco.mj_ray(self.model,self.data,origin,np.array([0.,0.,-1.]),RAY_GROUPS,1,-1,gid)
        if distance<0: raise ValueError('No warehouse surface under this ignition')
        return [x,y,float(7-distance)]

    def effects(self,sources,incidents,active_zones):
        for n in range(8):
            for kind in ('flame','marker'):
                body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'{kind}_{n}')
                mocap=self.model.body_mocapid[body]
                self.data.mocap_pos[mocap]=[0,0,-5]
            if n<len(sources) and sources[n]['temperature']>45:
                body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'flame_{n}')
                point=np.array(sources[n]['position']); point[2]+=.5+.2*math.sin(self.data.time*8+n)
                self.data.mocap_pos[self.model.body_mocapid[body]]=point
            if n<len(incidents):
                body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'marker_{n}')
                self.data.mocap_pos[self.model.body_mocapid[body]]=np.array(incidents[n]['position'])+[0,0,.08]
        for zone in range(64):
            gid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,f'zone_{zone}')
            self.model.geom_rgba[gid]=[.1,.55,1,.25 if zone in active_zones else .04]
        for n in range(64):
            body=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'drop_{n}')
            joint=self.model.body_jntadr[body]; q=self.model.jnt_qposadr[joint]; vel=self.model.jnt_dofadr[joint]
            if active_zones and self.data.qpos[q+2]<.2:
                zone=sorted(active_zones)[n%len(active_zones)]
                point=zone_center(zone)+self.rng.uniform([-1.6,-1.6,-.5],[1.6,1.6,0])
                self.data.qpos[q:q+3]=point; self.data.qpos[q+3:q+7]=[1,0,0,0]
                self.data.qvel[vel:vel+6]=[0,0,-2,0,0,0]
            elif not active_zones:
                self.data.qpos[q:q+3]=[0,0,-10]; self.data.qvel[vel:vel+6]=0
        for _ in range(10): mujoco.mj_step(self.model,self.data)
