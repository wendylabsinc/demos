"""Procedural city, MuJoCo planar vehicle dynamics, and sensor-driven taxi control."""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

DT = .01
ROAD_CENTERS = (-36., 0., 36.)
ROAD_HALF_WIDTH = 5.7
ROAD_EXTENT = 56.
TREES = [(x,y) for x in (-48.,48.) for y in (-48.,-24.,-12.,12.,24.,48.)] + [(x,y) for y in (-48.,48.) for x in (-24.,-12.,12.,24.)]

def road_clearance(x,y):
    return min(min(max(abs(y-c)-ROAD_HALF_WIDTH,abs(x)-ROAD_EXTENT),
                   max(abs(x-c)-ROAD_HALF_WIDTH,abs(y)-ROAD_EXTENT)) for c in ROAD_CENTERS)

CAMERAS = ['front', 'front_left', 'front_right', 'left', 'right', 'rear']
FONT = {
'S':['111','100','111','001','111'], 'T':['111','010','010','010','010'],
'O':['111','101','101','101','111'], 'P':['110','101','110','100','100'],
'3':['111','001','111','001','111'], '0':['111','101','101','101','111'],
'T':['111','010','010','010','010'], 'A':['010','101','111','101','101'],
'X':['101','101','010','101','101'], 'I':['111','010','010','010','111'],
}

def vec(v): return ' '.join(str(float(x)) for x in v)
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

class Route:
    def __init__(self, extent=38.4, reverse=False):
        a, r = extent, 8.
        points=[]
        # Counterclockwise right-hand outer lane with tangent quarter-circle turns.
        for rotation in range(4):
            angle=rotation*math.pi/2
            c,s=math.cos(angle),math.sin(angle)
            local=[(x,-a) for x in np.linspace(-a+r,a-r,160,endpoint=False)]
            local += [(a-r+r*math.cos(t),-a+r+r*math.sin(t)) for t in np.linspace(-math.pi/2,0,40,endpoint=False)]
            points += [(c*x-s*y,s*x+c*y) for x,y in local]
        self.points=np.array(points[::-1] if reverse else points)
        self.lengths=np.linalg.norm(np.roll(self.points,-1,axis=0)-self.points,axis=1)
        self.ss=np.r_[0,np.cumsum(self.lengths)[:-1]]
        self.length=float(self.lengths.sum())
        self.segments=np.roll(self.points,-1,axis=0)-self.points
        self.lengths_squared=self.lengths**2
        self.stop_lines=[((self.nearest(np.array(p))[0]-9.)%self.length,vertical)
                         for p,vertical in [([0,-a],False),([a,0],True),([0,a],False),([-a,0],True)]]
    def at(self,s):
        s=s%self.length; i=int(np.searchsorted(self.ss,s,side='right')-1)
        f=(s-self.ss[i])/self.lengths[i]
        p=self.points[i]*(1-f)+self.points[(i+1)%len(self.points)]*f
        d=self.points[(i+1)%len(self.points)]-self.points[i]
        return p,math.atan2(d[1],d[0])
    def nearest(self,p):
        segments=self.segments
        f=np.clip(np.sum((p-self.points)*segments,axis=1)/self.lengths_squared,0,1)
        distances=np.linalg.norm(self.points+segments*f[:,None]-p,axis=1)
        i=int(np.argmin(distances))
        return float((self.ss[i]+f[i]*self.lengths[i])%self.length),float(distances[i])

OUTER=Route(); INNER=Route(33.6,True)
STOPS=[dict(name='Central station',s=12.,kind='pickup'),dict(name='Riverside hotel',s=127.,kind='dropoff'),dict(name='City gardens',s=195.,kind='pickup'),dict(name='Central station',s=12.,kind='dropoff')]
SIGN_S=OUTER.nearest([38.4,14])[0]


def build_xml():
    root=ET.Element('mujoco',model='Wendy autonomous taxi city')
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep=str(DT),gravity='0 0 -9.81',integrator='implicitfast')
    visual=ET.SubElement(root,'visual'); ET.SubElement(visual,'global',offwidth='960',offheight='640')
    ET.SubElement(visual,'map',znear='.05',zfar='350',fogstart='120',fogend='230')
    ET.SubElement(visual,'rgba',fog='.64 .76 .83 1')
    asset=ET.SubElement(root,'asset')
    ET.SubElement(asset,'texture',type='skybox',builtin='gradient',rgb1='.25 .47 .65',rgb2='.84 .91 .96',width='512',height='3072')
    world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='-30 -40 90',dir='.3 .4 -1',directional='true',diffuse='.85 .82 .76',ambient='.35 .38 .4',castshadow='true')
    def geom(parent,name,pos,size,color,typ='box',collide=False,**kw):
        return ET.SubElement(parent,'geom',name=name,type=typ,pos=vec(pos),size=vec(size),rgba=color,contype='1' if collide else '0',conaffinity='1' if collide else '0',group='0',**kw)
    geom(world,'ground',(0,0,-.15),(68,68,.15),'.29 .40 .31 1')
    for i in ROAD_CENTERS:
        geom(world,f'road_x{i}',(0,i,.01),(ROAD_EXTENT,ROAD_HALF_WIDTH,.02),'.105 .13 .16 1')
        geom(world,f'road_y{i}',(i,0,.015),(ROAD_HALF_WIDTH,ROAD_EXTENT,.02),'.105 .13 .16 1')
        for q in range(-52,53,5):
            if all(abs(q-z)>6 for z in [-36,0,36]):
                for shift in [-.12,.12]:
                    geom(world,f'mark_x{i}_{q}_{shift}',(q,i+shift,.05),(1.35,.045,.012),'.98 .72 .27 1')
                    geom(world,f'mark_y{i}_{q}_{shift}',(i+shift,q,.05),(.045,1.35,.012),'.98 .72 .27 1')
    # Four dense blocks, sidewalks, pocket gardens and building windows.
    rng=np.random.default_rng(8)
    for bx in [-18,18]:
        for by in [-18,18]:
            geom(world,f'sidewalk{bx}_{by}',(bx,by,.12),(12.3,12.3,.12),'.67 .69 .66 1',collide=True)
            for ix in [-1,1]:
                for iy in [-1,1]:
                    x,y=bx+ix*6.2,by+iy*6.2; h=float(rng.uniform(5,14))
                    color=['.66 .51 .39 1','.43 .52 .56 1','.77 .72 .60 1','.48 .57 .59 1'][int(rng.integers(4))]
                    geom(world,f'building{x}_{y}',(x,y,h/2+.3),(4.3,4.2,h/2),color,collide=True)
                    geom(world,f'roof{x}_{y}',(x,y,h+.4),(4.5,4.4,.15),'.23 .28 .3 1')
                    for z in np.arange(1.6,h,2.1):
                        for off in [-2.7,0,2.7]:
                            for side in [-1,1]:
                                geom(world,f'winx{x}_{y}_{z}_{off}_{side}',(x+side*4.31,y+off,z),(.015,.7,.55),'.26 .55 .66 1')
                                geom(world,f'winy{x}_{y}_{z}_{off}_{side}',(x+off,y+side*4.21,z),(.7,.015,.55),'.26 .55 .66 1')
    for i,(x,y) in enumerate(TREES):
        assert road_clearance(x,y)>2., 'Tree canopy overlaps a road'
        geom(world,f'trunk{i}',(x,y,1.2),(.18,1.2),'.36 .25 .15 1','cylinder')
        geom(world,f'leaf{i}',(x,y,3),(1.6,1.6,2),'.18 .38 .23 1','ellipsoid')
    # Pedestrian crossings and roadside traffic signals at all grid intersections.
    for x in [-36,0,36]:
        for y in [-36,0,36]:
            for k in range(-4,5):
                for sign in [-1,1]:
                    geom(world,f'zebra_x{x}_{y}_{k}_{sign}',(x+sign*6.7,y+k,.058),(.8,.28,.01),'.85 .86 .79 1')
                    geom(world,f'zebra_y{x}_{y}_{k}_{sign}',(x+k,y+sign*6.7,.058),(.28,.8,.01),'.85 .86 .79 1')
            for n,(dx,dy) in enumerate([(-7,-6.5),(6.5,-7),(7,6.5),(-6.5,7)]):
                geom(world,f'pole{x}_{y}_{n}',(x+dx,y+dy,2),(.09,2),'.23 .27 .29 1','cylinder')
                facing=[(-1,0),(0,-1),(1,0),(0,1)][n]
                geom(world,f'signal{x}_{y}_{n}',(x+dx,y+dy,4),(.28,.24,.65),'.055 .065 .075 1',euler=vec([0,0,math.atan2(facing[1],facing[0])+math.pi/2]))
                for lamp,dz in enumerate([.38,0,-.38]):
                    geom(world,f'lamp_{x}_{y}_{n}_{lamp}',(x+dx+facing[0]*.27,y+dy+facing[1]*.27,4+dz),(.15,),'.15 .17 .13 1','sphere')
    def text(parent,word,pos,scale,color='.99 .98 .90 1'):
        for n,ch in enumerate(word):
            for row,line in enumerate(FONT.get(ch,FONT['0'])):
                for col,v in enumerate(line):
                    if v=='1': geom(parent,f'txt{word}{n}_{row}_{col}',(pos[0],pos[1]-(n*4+col)*scale,pos[2]-row*scale),(.012,scale*.43,scale*.43),color)
    stop=ET.SubElement(world,'body',pos='44 14 0',euler=f'0 0 {-math.pi/2}')
    geom(stop,'stop_pole',(0,0,1.6),(.08,1.6),'.5 .55 .57 1','cylinder')
    geom(stop,'stop_board',(0,0,3),(.05,.9,.65),'.78 .045 .055 1'); text(stop,'STOP',(.07,.73,3.23),.1)
    limit=ET.SubElement(world,'body',pos='-10 -44 0',euler=f'0 0 {math.pi/2}')
    geom(limit,'speed_pole',(0,0,1.4),(.07,1.4),'.5 .55 .57 1','cylinder')
    geom(limit,'speed_board',(0,0,2.8),(.05,.55,.6),'.95 .94 .85 1'); text(limit,'30',(.07,.36,3.02),.1,'.09 .11 .13 1')
    for n,stop in enumerate(STOPS[:3]):
        p,yaw=OUTER.at(stop['s']); right=np.array([math.sin(yaw),-math.cos(yaw)]); p+=right*5
        geom(world,f'taxi_bay{n}',(*p,.06),(2.8,1.5,.03),'.05 .62 .57 1',euler=vec([0,0,yaw]))
        geom(world,f'passenger{n}',(*p,1.2),(.22,.22,.65),'.91 .47 .19 1','ellipsoid')
        geom(world,f'head{n}',(*p,2),(.23,),'.84 .67 .51 1','sphere')
    actuators=ET.SubElement(root,'actuator')
    configs=[('taxi',OUTER,0.,'1 .70 .055 1'),('traffic_blue',OUTER,65.,'.1 .43 .72 1'),('traffic_white',OUTER,180.,'.84 .87 .9 1'),('traffic_red',INNER,28.,'.72 .13 .16 1'),('traffic_teal',INNER,145.,'.06 .56 .51 1')]
    for name,route,s,color in configs:
        body=ET.SubElement(world,'body',name=name,pos='0 0 .8')
        for joint,axis,typ in [('x','1 0 0','slide'),('y','0 1 0','slide'),('yaw','0 0 1','hinge')]:
            ET.SubElement(body,'joint',name=f'{name}_{joint}',type=typ,axis=axis,damping='0')
            ET.SubElement(actuators,'motor',name=f'{name}_{joint}',joint=f'{name}_{joint}',ctrllimited='true',ctrlrange='-15000 15000')
        ET.SubElement(body,'inertial',pos='0 0 0',mass='1100',diaginertia='450 1400 1600')
        geom(body,name+'_chassis',(0,0,0),(2.05,.87,.35),color,collide=True)
        geom(body,name+'_cabin',(-.15,0,.55),(1.12,.76,.36),'.12 .23 .28 1')
        geom(body,name+'_roof',(-.2,0,.93),(.9,.77,.07),color)
        geom(body,name+'_hood',(1.43,0,.25),(.6,.83,.12),color)
        for x in [-1.3,1.3]:
            for y in [-.89,.89]:
                geom(body,f'{name}_tire{x}_{y}',(x,y,-.38),(.39,.14),'.025 .03 .035 1','cylinder',euler=f'{math.pi/2} 0 0')
                geom(body,f'{name}_hub{x}_{y}',(x,y*1.16,-.38),(.19,.025),'.55 .60 .62 1','cylinder',euler=f'{math.pi/2} 0 0')
        for y in [-.6,.6]:
            geom(body,f'{name}_headlight{y}',(2.065,y,.08),(.016,.18,.1),'1 .94 .67 1')
            geom(body,f'{name}_tail{y}',(-2.065,y,.08),(.016,.18,.1),'.92 .07 .06 1')
        if name=='taxi':
            geom(body,'taxi_sign',(-.2,0,1.18),(.4,.32,.18),'1 .85 .24 1')
            geom(body,'lidar_unit',(.55,0,1.22),(.18,.15),'.09 .12 .14 1','cylinder')
            for cam,angle in zip(CAMERAS,[0,math.pi/4,-math.pi/4,math.pi/2,-math.pi/2,math.pi]):
                dx,dy=math.cos(angle),math.sin(angle)
                # Camera optical -Z points outwards, image Y points up.
                ET.SubElement(body,'camera',name=cam,pos=vec([dx*1.7,dy*.93,.65]),xyaxes=vec([dy,-dx,0,0,0,1]),fovy='68')
                geom(body,'cam_'+cam,(dx*1.72,dy*.96,.65),(.07,),'.035 .04 .045 1','sphere')
        for g in body.iter('geom'): g.set('group','2' if name=='taxi' else '1')
    return ET.tostring(root,encoding='unicode'),configs

class City:
    def __init__(self):
        xml,configs=build_xml(); self.xml=xml
        self.model=mujoco.MjModel.from_xml_string(xml); self.data=mujoco.MjData(self.model)
        self.cars=[]
        for name,route,s,color in configs:
            j=self.model.joint(name+'_x'); q=int(j.qposadr[0]); v=int(j.dofadr[0]); u=self.model.actuator(name+'_x').id
            p,yaw=route.at(s); self.data.qpos[q:q+3]=[*p,yaw]
            self.cars.append(dict(name=name,route=route,q=q,v=v,u=u,s=s,progress=s,error=0.,steer=0.,target=0.,reason='Cruising',sign_hold=0.,sign_done=-1,body=self.model.body(name).id))
        mujoco.mj_forward(self.model,self.data)
        self.frame=0; self.mission=0; self.completed=0; self.dwell=0.; self.distance=0.; self.events=[]; self.collisions=0; self.contact_pairs=set(); self.max_error=0.; self.last_scan=-1.; self.scan=np.full(180,60.); self.scan_geoms=np.full(180,-1,dtype=np.int32)
        self.angles=np.linspace(-math.pi,math.pi,180,endpoint=False)
        self.running=True; self.emergency=False; self.last_reason=''; self.sign_stops=0; self.red_wait=0.; self.lidar_wait=0.; self.trip_wait=0.
        self.lamps=[]
        for i in range(self.model.ngeom):
            name=mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
            if name.startswith('lamp_'):
                *_,n,lamp=name.split('_'); self.lamps.append((i,int(n)%2==1,int(lamp)))
        self.map_geometry=dict(roads=[dict(axis=axis,center=c,half_width=ROAD_HALF_WIDTH,extent=ROAD_EXTENT) for c in ROAD_CENTERS for axis in ('x','y')],trees=[dict(position=[x,y],radius=1.6) for x,y in TREES],buildings=[dict(position=self.data.geom_xpos[i,:2].tolist(),size=self.model.geom_size[i,:2].tolist()) for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('building')])
        self.update_sensors(); self.update_lights()
    def event(self,kind,**kw):
        self.events.append(dict(time=round(self.data.time,2),kind=kind,**kw)); self.events=self.events[-60:]
    def light_green(self,vertical=False):
        phase=self.data.time%36
        return 19<=phase<34 if vertical else 1<=phase<16
    def update_lights(self):
        for i,vertical,lamp in self.lamps:
            green=self.light_green(vertical)
            self.model.geom_rgba[i]=[.1,.95,.3,1] if lamp==2 and green else [.95,.08,.06,1] if lamp==0 and not green else [.07,.09,.07,1]
    def update_sensors(self):
        car=self.cars[0]; q=car['q']; yaw=self.data.qpos[q+2]; p=np.array([*self.data.qpos[q:q+2],1.35])
        mask=np.array([1,1,0,0,0,0],dtype=np.uint8); hit=np.zeros(1,dtype=np.int32)
        for i,a in enumerate(self.angles+yaw):
            d=mujoco.mj_ray(self.model,self.data,p,np.array([math.cos(a),math.sin(a),0.]),mask,1,car['body'],hit)
            self.scan[i]=min(d,60) if d>=0 else 60; self.scan_geoms[i]=hit[0] if 0<=d<60 else -1
        self.last_scan=float(self.data.time)
    def stop_speed(self,d): return min(math.sqrt(max(0.,2*2.2*(d-.4))), max(0.,(d-.4)*.9))
    def control(self,car):
        q,v,u=car['q'],car['v'],car['u']; x,y,yaw=self.data.qpos[q:q+3]
        vx,vy,w=self.data.qvel[v:v+3]; c,s=math.cos(yaw),math.sin(yaw)
        speed=c*vx+s*vy; lateral=-s*vx+c*vy
        route=car['route']; near,error=route.nearest(np.array([x,y])); delta=(near-car['s']+route.length/2)%route.length-route.length/2
        car['progress']+=delta; car['s']=near; car['error']=error
        look=4.+max(0,speed)*.6; target,heading=route.at(near+look)
        alpha=wrap(math.atan2(target[1]-y,target[0]-x)-yaw)
        steer=float(np.clip(math.atan2(2*2.6*math.sin(alpha),look),-.55,.55))
        target_speed=5.5 if abs(steer)<.12 else 3.2; reason='Following lane'
        # Traffic lights at the midpoints of each side, with mapped stop lines.
        for line,vertical in route.stop_lines:
            d=(line-near)%route.length
            if d<24 and not self.light_green(vertical):
                limit=self.stop_speed(d)
                if limit<target_speed: target_speed=limit; reason='Red light'
        # Every driver obeys the stop sign, once per lap.
        if route is OUTER:
            d=(SIGN_S-3.-near)%route.length; lap=round((car['progress']+d-(SIGN_S-3))/route.length)
            if d<16 and car['sign_done']!=lap:
                target_speed=min(target_speed,self.stop_speed(d)); reason='STOP sign'
                if d<.65 and abs(speed)<.18:
                    car['sign_hold']+=DT
                    if car['sign_hold']>1.5:
                        car['sign_done']=lap; car['sign_hold']=0.
                        if car is self.cars[0]: self.sign_stops+=1; self.event('stop_sign_observed')
        if car is self.cars[0]:
            self.distance+=max(delta,0.); self.max_error=max(self.max_error,error)
            # Real ray ranges, transformed to a forward collision corridor.
            forward=self.scan*np.cos(self.angles); side=self.scan*np.sin(self.angles)
            valid=(forward>1.8)&(np.abs(side)<1.1)&(self.scan<59.9)
            gap=float(np.min(forward[valid])-2.05) if np.any(valid) else 60.
            obstacle_speed=self.stop_speed(max(0,gap-2.))
            if obstacle_speed<target_speed: target_speed=obstacle_speed; reason='LiDAR yielding'
            if self.data.time-self.last_scan>.35 or not np.isfinite(self.scan).all(): target_speed=0.; reason='Sensor fault'
            stop=STOPS[self.mission]; d=(stop['s']-near)%route.length
            if d<25:
                limit=self.stop_speed(d)
                if limit<target_speed: target_speed=limit; reason='Approaching '+stop['name']
                if d<.75 and abs(speed)<.18:
                    self.dwell+=DT; target_speed=0.; reason='Boarding passenger' if stop['kind']=='pickup' else 'Dropping off passenger'
                    self.trip_wait+=DT
                    if self.dwell>=3.:
                        self.event(stop['kind'],destination=stop['name']); self.completed+=stop['kind']=='dropoff'; self.mission=(self.mission+1)%len(STOPS); self.dwell=0.
            if self.emergency or not self.running: target_speed=0.; reason='Emergency brake' if self.emergency else 'Taxi paused'
            if reason=='Red light' and speed<.3: self.red_wait+=DT
            if reason=='LiDAR yielding': self.lidar_wait+=DT
        else:
            # Traffic follows its own lane and brakes for vehicles ahead.
            for other in self.cars:
                if other is car: continue
                oq=other['q']; dx,dy=self.data.qpos[oq:oq+2]-np.array([x,y]); ahead=c*dx+s*dy; side=-s*dx+c*dy
                if 0<ahead<20 and abs(side)<1.8:
                    target_speed=min(target_speed,self.stop_speed(max(0,ahead-6.)))
        if car.get('parked'): target_speed=0.; reason='Parked traffic'
        # Dynamic bicycle tire approximation. MuJoCo integrates all live poses.
        # Longitudinal motor/braking force + front/rear lateral tire slip forces.
        drive=float(np.clip((target_speed-speed)*5000,-4400,2400))
        denom=max(abs(speed),2.)
        front=float(np.clip(5500*(steer-(lateral+1.3*w)/denom),-4400,4400))
        rear=float(np.clip(6500*(-(lateral-1.3*w)/denom),-4400,4400))
        fy=front+rear
        self.data.ctrl[u:u+3]=[c*drive-s*fy,s*drive+c*fy,1.3*(front-rear)]
        car.update(steer=steer,target=target_speed,speed=float(speed),reason=reason)
    def step(self):
        if self.frame%10==0: self.update_sensors(); self.update_lights()
        for car in self.cars: self.control(car)
        mujoco.mj_step(self.model,self.data); self.frame+=1
        for ct in self.data.contact:
            b1=int(self.model.geom_bodyid[ct.geom1]); b2=int(self.model.geom_bodyid[ct.geom2]); taxi=self.cars[0]['body']
            if taxi in (b1,b2) and b1!=b2:
                pair=tuple(sorted([int(ct.geom1),int(ct.geom2)]))
                if pair not in self.contact_pairs: self.contact_pairs.add(pair); self.collisions+=1; self.event('collision',geoms=list(pair))
    def state(self):
        car=self.cars[0]; q=car['q']; stop=STOPS[self.mission]
        return dict(time=round(float(self.data.time),2),frame=self.frame,position=self.data.qpos[q:q+3].tolist(),speed=round(car.get('speed',0)*3.6,1),steering=round(math.degrees(car['steer']),1),reason=car['reason'],destination=stop['name'],mission=stop['kind'],trips=self.completed,distance=round(self.distance),collisions=self.collisions,lane_error=round(car['error'],2),lidar=self.scan.round(2).tolist(),sensor_age=round(self.data.time-self.last_scan,3),cameras=CAMERAS,map_geometry=self.map_geometry,events=self.events,traffic=[dict(name=c['name'],position=self.data.qpos[c['q']:c['q']+3].tolist()) for c in self.cars[1:]],route=OUTER.points[::4].tolist(),stops=[dict(**p,position=OUTER.at(p['s'])[0].tolist()) for p in STOPS[:3]],green_horizontal=self.light_green(),green_vertical=self.light_green(True),running=self.running,emergency=self.emergency)
