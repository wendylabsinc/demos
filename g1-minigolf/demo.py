"""Floating-base G1 minigolf with joint servos and a physical hand-mounted club."""
import os, sys, time, math, json, threading, argparse
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
os.environ['WENDY_WORLDSIM_DIR']=str(ROOT/'runtime/minigolf-physics')
import numpy as np
import mujoco
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, Response
import uvicorn

TITLE='Unitree G1 • Physical minigolf'
COURSES={
    'garden': dict(title='Garden green', cup=2.65, speed=.65, color='.12 .48 .27 1', description='An open green with a long straight putt'),
    'gates': dict(title='Bend', cup=2.8, cup_y=.7, speed=.90, color='.16 .43 .36 1', description='A curved fairway turns toward an offset cup'),
    'bridge': dict(title='S curve', cup=3.9, cup_y=0., speed=1.1, acceleration=5., rolling=.00003, color='.23 .46 .22 1', description='Two opposing curves cross a sunken water garden'),
}
CUP=COURSES['garden']['cup']  # Compatibility for the original physics audit.
active={'course':'garden','slot':'minigolf-garden','ready':False}
shared={}; commands={'paused':False,'reset':False,'stop':False,'steps':0,'course':None}
def course_file(key): return COURSES[key]['title']
app=FastAPI()
@app.get('/ctl/sim-running')
def running():
    slot=active['slot']; course=active['course']
    return {'focus':slot if active['ready'] else None,'running':[{'file':course_file(course),'slot':slot,'title':f"G1 • {COURSES[course]['title']}"}] if active['ready'] else []}
@app.get('/ctl/sim-list')
def simlist(): return {'current':course_file(active['course']),'sims':[course_file(key) for key in COURSES]}
@app.post('/ctl/sim-open')
async def simopen(request:Request):
    file=(await request.json()).get('file')
    key=next((key for key in COURSES if course_file(key)==file),None)
    if key is None: return Response(status_code=400)
    commands.update(course=key,paused=False)
    return {'ok':True}
@app.get('/simslot/{slot}/{filename}')
def simfile(slot:str,filename:str):
    if slot!=active['slot'] or not active['ready'] or filename not in ('scene.json','state.json'): return Response(status_code=404)
    p=ROOT/'runtime'/slot/filename
    return FileResponse(p) if p.exists() else Response(status_code=503)
@app.get('/api/status')
def status(): return dict(shared)
@app.post('/ctl/sim-cmd')
async def simcmd(request:Request):
    cmd=(await request.json()).get('cmd')
    if cmd=='reset': commands.update(reset=True,paused=False)
    if cmd=='toggle': commands['paused']=not commands['paused']
    if cmd=='pause': commands['paused']=True
    if cmd=='resume': commands['paused']=False
    if cmd=='step': commands['paused']=True; commands['steps']+=1
    return {'ok':True,'control':{'paused':commands['paused']}}
@app.post('/ctl/sim-stop')
def stop(): commands['stop']=True; return {'ok':True}
@app.websocket('/ctl/sim-stream')
async def stream(ws:WebSocket):
    import asyncio
    await ws.accept()
    try:
        while not commands['stop']:
            slot=active['slot']; p=ROOT/'runtime'/slot/'state.json'
            if active['ready'] and p.exists(): await ws.send_text(json.dumps({'slot':slot,'state':json.loads(p.read_text())}))
            await asyncio.sleep(.025)
    except Exception: pass

def scene(course='garden'):
    config=COURSES[course]; cup=config['cup']; cy=config.get('cup_y',0.); end=cup+.65
    rolling=config.get('rolling',.00018)
    turf_friction=f'.18 .002 {rolling}'
    source=ROOT/'menagerie/unitree_g1'
    tree=ET.parse(source/'g1.xml'); root=tree.getroot()
    root.find('compiler').set('meshdir',str(source/'assets'))
    root.find('option').set('timestep','0.002')
    root.remove(root.find('keyframe'))
    world=root.find('worldbody')
    # Keep Menagerie's free base, masses, inertias, joint limits and collisions.
    # Foot friction must support the robot, independently of the turf's rolling loss.
    for g in world.findall(".//geom[@class='foot']"): g.set('friction','.8 .005 .0003')
    for side in ('left','right'):
        b=world.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(b,'site',name=f'{side}_grip',pos='.06 0 0',size='.008',rgba='0 0 0 0')
    asset=root.find('asset')
    ET.SubElement(asset,'material',name='turf',rgba=config['color'])
    def geom(name,typ,pos,size,rgba,**kw):
        return ET.SubElement(world,'geom',name=name,type=typ,pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba=rgba,**kw)
    geom('surround','plane',(0,0,-.31),(6,4,.1),'.06 .09 .13 1')
    def turf(name,x0,x1,y0,y1):
        geom(name,'box',((x0+x1)/2,(y0+y1)/2,-.025),((x1-x0)/2,(y1-y0)/2,.025),'1 1 1 1',material='turf',friction=turf_friction,condim='6')
    if course=='garden':
        turf('fairway',-.73,cup-.09,-.8,.8)
        turf('end',cup+.09,end,-.8,.8)
        turf('cup_north',cup-.09,cup+.09,.09,.8)
        turf('cup_south',cup-.09,cup+.09,-.8,-.09)
        for y in (-.84,.84): geom(f'rail{y}','box',((end-.73)/2,y,.04),((end+.73)/2+.03,.035,.065),'.86 .69 .36 1')
        for x in (-.78,end+.07): geom(f'end_rail{x}','box',(x,0,.04),(.035,.875,.065),'.86 .69 .36 1')
    else:
        turf('tee',-.73,.95,-.8,.8)
        geom('tee_back','box',(-.78,0,.04),(.035,.84,.065),'.86 .69 .36 1')
        for y in (-.84,.84): geom(f'tee_side{y}','box',(.08,y,.04),(.86,.035,.065),'.86 .69 .36 1')
        # The green and its rails are built along the actual curved centerline.
        # Segment boxes overlap at seams; the surrounding gaps stay open.
        xs=np.linspace(.9,cup-.3,45)
        u=(xs-.9)/(cup-.3-.9)
        ys=cy*(u*u*(3-2*u)) if course=='gates' else .28*np.sin(2*np.pi*u)*np.sin(np.pi*u)
        points=np.column_stack([xs,ys])
        width=.17
        for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
            width=.17-.075*max(0.,(i/43-.72)/.28)
            delta=b-a; length=float(np.linalg.norm(delta)); angle=math.atan2(delta[1],delta[0]); normal=np.array([-delta[1],delta[0]])/length; mid=(a+b)/2
            geom(f'curve_turf{i}','box',(*mid,-.025),(length/2+.012,width,.025),'1 1 1 1',material='turf',friction=turf_friction,condim='6',group='3',euler=f'0 0 {angle}')
            for side in (-1,1):
                rail=mid+side*normal*(width+.023)
                geom(f'curve_rail{i}_{side}','box',(*rail,.05),(length/2+.012,.023,.07),'.86 .69 .36 1',friction='.02 .001 .00003',priority='2',condim='3',solref='.004 1',group='3',euler=f'0 0 {angle}')
        # Continuous visible ribbons avoid depth fighting between collision boxes.
        # They carry no collisions; the original overlapping boxes handle physics.
        tangents=np.gradient(points,axis=0)
        normals=np.column_stack([-tangents[:,1],tangents[:,0]])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        widths=.17-.075*np.maximum(0.,(np.linspace(0,1,45)-.72)/.28)
        def ribbon(name,inner,outer,bottom,top,color):
            vertices=[]; faces=[]
            for point,normal,lo,hi in zip(points,normals,inner,outer):
                left=point+normal*lo; right=point+normal*hi
                vertices.extend([(*left,bottom),(*right,bottom),(*right,top),(*left,top)])
            for j in range(44):
                a=4*j; b=a+4
                for k in range(4):
                    n=(k+1)%4
                    faces.extend([(a+k,b+k,b+n),(a+k,b+n,a+n)])
            faces.extend([(0,1,2),(0,2,3),(176,179,178),(176,178,177)])
            ET.SubElement(asset,'mesh',name=name,vertex=' '.join(str(v) for row in vertices for v in row),face=' '.join(str(v) for a,b,c in faces for v in (a,c,b)))
            geom(name,'mesh',(0,0,0),(1,1,1),color,mesh=name,contype='0',conaffinity='0',mass='0')
        ribbon('green_ribbon',-widths,widths,-.05,.0002,config['color'])
        for side in (-1,1):
            lo=side*widths; hi=side*(widths+.046)
            ribbon(f'rail_ribbon{side}',np.minimum(lo,hi),np.maximum(lo,hi),-.02,.12,'.86 .69 .36 1')
        # A narrow cup landing prevents a direct shortcut through a broad green.
        turf('landing',cup-.31,cup-.09,cy-.095,cy+.095)
        turf('end',cup+.09,end,cy-.095,cy+.095)
        turf('cup_north',cup-.09,cup+.09,cy+.09,cy+.095)
        turf('cup_south',cup-.09,cup+.09,cy-.095,cy-.09)
        for y in (cy-.118,cy+.118): geom(f'landing_rail{y}','box',((cup-.3+end)/2,y,.05),((end-cup+.3)/2,.023,.07),'.86 .69 .36 1',friction='.02 .001 .00003',priority='2',condim='3')
        geom('landing_end','box',(end+.023,cy,.05),(.023,.216,.07),'.86 .69 .36 1')
        if course=='bridge':
            geom('water','box',((cup+.6)/2,0,-.26),((cup-.6)/2,.85,.018),'.08 .35 .55 1',contype='0',conaffinity='0')
            for i in (4,18,32):
                x,y=points[i]
                geom(f'bridge_support{i}','box',(x,y,-.16),(.05,.15,.14),'.45 .29 .17 1')
    # Every course has a real opening and a physical cup below the green.
    geom('cup_bottom','box',(cup,cy,-.18),(.10,.10,.01),'.015 .025 .03 1')
    for i,(pos,size) in enumerate([((cup-.10,cy,-.09),(.01,.11,.09)),((cup+.10,cy,-.09),(.01,.11,.09)),((cup,cy-.10,-.09),(.10,.01,.09)),((cup,cy+.10,-.09),(.10,.01,.09))]):
        geom(f'cup_wall{i}','box',pos,size,'.02 .03 .04 1',solref='.004 1')
    geom('flagpole','cylinder',(cup,cy+.25,.48),(.008,.48),'.9 .94 .97 1',contype='0',conaffinity='0')
    geom('flag','box',(cup+.13,cy+.25,.88),(.13,.008,.075),'1 .27 .22 1',contype='0',conaffinity='0')
    for i,(x,y) in enumerate([(-.7,1.15),(1.4,1.15),(end-.1,cy+1.0),(end-.1,cy-1.0)]):
        geom(f'pot{i}','cylinder',(x,y,-.12),(.15,.17),'.65 .32 .2 1')
        geom(f'plant{i}','sphere',(x,y,.1),(.23,),'.18 .57 .32 1')
    ball=ET.SubElement(world,'body',name='golf_ball',pos='.45 0 .036')
    ET.SubElement(ball,'freejoint',name='ball_free')
    ET.SubElement(ball,'geom',name='ball',type='sphere',size='.035',mass='.046',rgba='.98 .98 .91 1',friction=turf_friction,condim='6',solref='.004 1')
    # Compute the initial arm configuration in a separate planning state.
    # The live state is initialized once; subsequent motion comes only from actuators.
    preview=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    planned=mujoco.MjData(preview)
    planned.qpos[:]=preview.qpos0; planned.qpos[:3]=[0,.26,.79]
    for side in ('left','right'):
        planned.qpos[preview.joint(f'{side}_shoulder_pitch_joint').qposadr[0]]=.2
        planned.qpos[preview.joint(f'{side}_elbow_joint').qposadr[0]]=1.28
    ids=[preview.joint(f'right_{part}_joint').id for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
    site=preview.site('right_grip').id
    for _ in range(150):
        mujoco.mj_forward(preview,planned)
        error=np.array([.10,0,.73])-planned.site_xpos[site]
        if np.linalg.norm(error)<.0001: break
        jac=np.zeros((3,preview.nv)); mujoco.mj_jacSite(preview,planned,jac,None,site)
        j=jac[:,preview.jnt_dofadr[ids]]
        step=j.T@np.linalg.solve(j@j.T+.0001*np.eye(3),error)
        for jid,v in zip(ids,step):
            q=preview.jnt_qposadr[jid]; planned.qpos[q]=np.clip(planned.qpos[q]+np.clip(v,-.08,.08),*preview.jnt_range[jid])
    mujoco.mj_forward(preview,planned)
    wrist=preview.body('right_wrist_yaw_link').id
    rotation=planned.xmat[wrist].reshape(3,3)
    local_head=rotation.T@(np.array([.32,0,.043])-planned.xpos[wrist])
    local_quat=np.zeros(4); mujoco.mju_mat2Quat(local_quat,rotation.T.reshape(-1))
    hand=world.find(".//body[@name='right_wrist_yaw_link']")
    club=ET.SubElement(hand,'body',name='putter',pos=' '.join(map(str,local_head)),quat=' '.join(map(str,local_quat)))
    ET.SubElement(club,'geom',name='putter_head',type='box',size='.026 .09 .025',mass='.20',rgba='.85 .87 .93 1',friction='.1 .001 .0001',solref='.004 1')
    ET.SubElement(club,'site',name='club_tip',pos='0 0 0',size='.006',rgba='0 0 0 0')
    # Shaft contributes inertia and collisions. Its geometry never changes at runtime.
    ET.SubElement(hand,'geom',name='club_shaft',type='capsule',fromto=' '.join(map(str,np.r_[[.06,0,0],local_head])),size='.009',mass='.09',rgba='.72 .78 .85 1')
    ET.SubElement(world,'camera',name='overview',pos='4 -5 3.7',xyaxes='.8 .6 0 -.3 .4 .86')
    tree.write(ROOT/'course.xml')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode')), planned.qpos.copy()

class Golf:
    def __init__(self,course='garden'):
        self.course=course; self.config=COURSES[course]; self.cup=self.config['cup']; self.cup_y=self.config.get('cup_y',0.)
        self.m,self.pose=scene(course); self.d=mujoco.MjData(self.m); self.plan=mujoco.MjData(self.m)
        self.ball=self.m.joint('ball_free').qposadr[0]; self.bvel=self.m.joint('ball_free').dofadr[0]
        self.tip=self.m.site('club_tip').id
        self.joints=[self.m.joint(f'right_{part}_joint').id for part in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
        self.qs=self.m.jnt_qposadr[self.joints]; self.vs=self.m.jnt_dofadr[self.joints]
        self.aqs=self.m.jnt_qposadr[self.m.actuator_trnid[:,0]]
        self.avs=self.m.jnt_dofadr[self.m.actuator_trnid[:,0]]
        self.round=0; self.reset()
    def reset(self):
        mujoco.mj_resetData(self.m,self.d); self.d.qpos[:]=self.pose
        mujoco.mj_forward(self.m,self.d)
        self.target=self.pose.copy(); self.target_velocity=np.zeros(self.m.nq); self.reference_x=.32; self.reference_velocity=0.; self.phase='Settle'; self.stable_time=0.; self.t=0.
        self.path_y_min=0.; self.path_y_max=0.; self.rail_contacts=0
        self.hit=False; self.sunk=False; self.contacts=0; self.max_speed=0.; self.min_upright=1.; self.min_height=1.; self.fault=False
        self.round+=1
    def ik_target(self,target):
        # IK changes only a scratch state. Joint servos must realize the target
        # against inertia, gravity and contact forces in the live simulation.
        self.plan.qpos[:]=self.d.qpos
        self.plan.qpos[self.qs]=self.target[self.qs]
        def forward_error():
            mujoco.mj_kinematics(self.m,self.plan); mujoco.mj_comPos(self.m,self.plan)
            r=self.plan.site_xmat[self.tip].reshape(3,3)
            rotation_error=.5*sum((np.cross(r[:,i],np.eye(3)[:,i]) for i in range(3)),np.zeros(3))
            return np.r_[3*(target-self.plan.site_xpos[self.tip]),rotation_error]
        for _ in range(24):
            error=forward_error()
            if np.linalg.norm(error)<.0001: break
            jac=np.zeros((3,self.m.nv)); rot=np.zeros_like(jac)
            mujoco.mj_jacSite(self.m,self.plan,jac,rot,self.tip)
            j=np.vstack([3*jac[:,self.vs],rot[:,self.vs]])
            delta=j.T@np.linalg.solve(j@j.T+np.eye(6)*.0005,error)
            delta*=min(1,.12/max(np.linalg.norm(delta),1e-9))
            before=self.plan.qpos[self.qs].copy(); accepted=False
            for scale in (1.,.5,.25,.1):
                self.plan.qpos[self.qs]=np.clip(before+scale*delta,self.m.jnt_range[self.joints,0],self.m.jnt_range[self.joints,1])
                if np.linalg.norm(forward_error())<np.linalg.norm(error): accepted=True; break
            if not accepted:
                self.plan.qpos[self.qs]=before; break
        forward_error()
        self.target[self.qs]=self.plan.qpos[self.qs]
    def tick(self):
        tip=self.d.site_xpos[self.tip].copy(); pos=self.d.qpos[self.ball:self.ball+3]
        pelvis=self.d.xmat[self.m.body('pelvis').id].reshape(3,3)
        upright=float(pelvis[2,2]); height=float(self.d.qpos[2])
        self.min_upright=min(self.min_upright,upright); self.min_height=min(self.min_height,height)
        stable=upright>.98 and np.linalg.norm(self.d.qvel[:6])<.15
        self.stable_time=self.stable_time+self.m.opt.timestep if stable else 0.
        if self.phase=='Settle' and self.stable_time>.6: self.phase='Backswing'
        if self.phase=='Backswing' and abs(tip[0]-.19)<.008 and abs(self.reference_x-.19)<.008 and abs(tip[2]-.043)<.015: self.phase='Putt'
        if self.hit: self.phase='Watching the ball'
        if self.sunk: self.phase='Hole in one!'
        if upright<.8 or height<.60 or self.t>20 and not self.hit:
            self.fault=True; self.phase='Stopped: balance or reach fault'
        if self.phase=='Putt' and self.reference_x>.50 and not self.hit:
            self.fault=True; self.phase='Stopped: no club contact'
        if int(round(self.t/self.m.opt.timestep))%5==0:
            x={'Settle':.32,'Backswing':.19,'Putt':.47}.get(self.phase,.47 if self.hit else .32)
            # A feedback servo approaches the task goal with bounded speed and
            # acceleration. Targets are not applied to the simulation state.
            dt=.01
            speed_limit=self.config['speed'] if self.phase=='Putt' else .15
            velocity=speed_limit if self.phase=='Putt' else np.clip(5*(x-self.reference_x),-speed_limit,speed_limit)
            accel_limit=self.config.get('acceleration',3.) if self.phase=='Putt' or self.hit else 1.
            self.reference_velocity+=np.clip(velocity-self.reference_velocity,-accel_limit*dt,accel_limit*dt)
            self.reference_x+=self.reference_velocity*dt
            before=self.target.copy()
            if not self.fault: self.ik_target(np.array([self.reference_x,0,.043]))
            self.target_velocity=(self.target-before)/dt
        controls=self.target[self.aqs].copy()+self.d.qfrc_bias[self.avs]/self.m.actuator_gainprm[:,0]-self.m.actuator_biasprm[:,2]*self.target_velocity[self.aqs]/self.m.actuator_gainprm[:,0]
        # Measured torso tilt and angular velocity stabilize the free base through
        # the ankles. No force is applied to the floating base itself.
        pitch=math.atan2(pelvis[0,2],pelvis[2,2]); roll=math.atan2(-pelvis[1,2],pelvis[2,2])
        for side in ('left','right'):
            controls[self.m.actuator(f'{side}_ankle_pitch_joint').id]+=np.clip(.7*pitch+.10*self.d.qvel[4],-.20,.20)
            controls[self.m.actuator(f'{side}_ankle_roll_joint').id]+=np.clip(.7*roll+.10*self.d.qvel[3],-.15,.15)
        self.d.ctrl[:]=np.clip(controls,self.m.actuator_ctrlrange[:,0],self.m.actuator_ctrlrange[:,1])
        mujoco.mj_step(self.m,self.d)
        for c in self.d.contact:
            names={self.m.geom(c.geom1).name,self.m.geom(c.geom2).name}
            if 'ball' in names and any(name and name.startswith('curve_rail') for name in names): self.rail_contacts+=1
            if names=={'ball','putter_head'}:
                self.hit=True; self.contacts+=1
        pos=self.d.qpos[self.ball:self.ball+3]; speed=float(np.linalg.norm(self.d.qvel[self.bvel:self.bvel+3])); self.max_speed=max(self.max_speed,speed)
        if pos[0]>.95:
            self.path_y_min=min(self.path_y_min,float(pos[1])); self.path_y_max=max(self.path_y_max,float(pos[1]))
        if abs(pos[0]-self.cup)<.12 and abs(pos[1]-self.cup_y)<.12 and pos[2]<-.06: self.sunk=True
        self.t=float(self.d.time)
        shared.update(course=self.course,course_title=self.config['title'],phase=self.phase,round=self.round,ball=pos.tolist(),club=tip.tolist(),speed=round(speed,3),putter_contact=self.hit,contact_steps=self.contacts,sunk=self.sunk,time=round(self.t,2),upright=round(upright,4),height=round(height,4),fault=self.fault,mode='Full MuJoCo dynamics / joint control')
        return self.phase

def verification(course):
    golf=Golf(course)
    for _ in range(6500): golf.tick()
    result=dict(shared,max_speed=golf.max_speed,min_upright=golf.min_upright,min_height=golf.min_height,finite=bool(np.isfinite(golf.d.qpos).all()),mocap_bodies=golf.m.nmocap,rail_contact_steps=golf.rail_contacts,path_y_min=golf.path_y_min,path_y_max=golf.path_y_max)
    (ROOT/'runtime').mkdir(exist_ok=True)
    (ROOT/f'runtime/verification-{course}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    assert result['finite'] and result['putter_contact'] and result['sunk'] and abs(result['ball'][0]-golf.cup)<.12 and result['speed']<.05 and result['min_upright']>.9 and not result['fault'] and result['mocap_bodies']==0, f'{course}: putt did not complete'
    if course=='gates': assert golf.rail_contacts>0 and golf.path_y_max>.5, 'Ball did not round the bend'
    if course=='bridge': assert golf.rail_contacts>0 and golf.path_y_max>.12 and golf.path_y_min<-.12, 'Ball did not follow both S turns'
    return result

def open_course(course):
    runtime=Path(os.environ.get('WENDY_SIM_RUNTIME',ROOT.parent.parent/'wendy-sandbox/image/sim')).expanduser()
    if runtime.is_dir():sys.path.insert(0,str(runtime))
    try:
        import wendymujoco
    except ModuleNotFoundError as exc:
        if exc.name!='wendymujoco':raise
        raise RuntimeError(
            'Interactive playback requires the licensed Wendy Sandbox simulator runtime. '
            'Set WENDY_SIM_RUNTIME to its image/sim directory containing wendymujoco.py. '
            'Headless --verify-all and verify_physics.py do not require that runtime.'
        ) from exc
    active['ready']=False
    golf=Golf(course)
    slot=f'minigolf-curved-v2-{course}'
    directory=ROOT/'runtime'/slot; directory.mkdir(parents=True,exist_ok=True)
    # A distinct slot makes Wendy load the new collision geometry and ground.
    wendymujoco._DIR=str(directory)
    for attr,file in [('_SCENE','scene.json'),('_STATE','state.json'),('_CONTROL','control.json')]:
        setattr(wendymujoco,attr,str(directory/file))
    viewer=wendymujoco.launch_passive(golf.m,golf.d,title=f"{TITLE} • {golf.config['title']}",rerun=False)
    shared.clear()
    viewer.sync()
    active.update(course=course,slot=slot,ready=True)
    return golf,viewer

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--verify-all',action='store_true')
    parser.add_argument('--course',choices=COURSES,default='garden')
    args=parser.parse_args()
    if args.verify or args.verify_all:
        results=[verification(key) for key in (COURSES if args.verify_all else [args.course])]
        (ROOT/'runtime/verification.json').write_text(json.dumps(results if args.verify_all else results[0],indent=2))
        return
    golf,viewer=open_course(args.course)
    threading.Thread(target=lambda:uvicorn.run(app,host='127.0.0.1',port=18992,log_level='warning'),daemon=True).start()
    try:
        while not commands['stop']:
            start=time.monotonic()
            course=commands['course']
            if course is not None:
                commands.update(course=None,reset=False,steps=0)
                viewer.close(); golf,viewer=open_course(course)
            if commands['reset']: golf.reset(); commands['reset']=False
            if not commands['paused'] or commands['steps']:
                if commands['steps']: commands['steps']-=1
                for _ in range(10): golf.tick()
            viewer.hud(course=golf.config['title'],phase=shared.get('phase','Ready'),hole=f"{list(COURSES).index(golf.course)+1} • Par 1",strokes=1,round=golf.round,ball_speed=shared.get('speed',0),upright=shared.get('upright',1),mode='Full physics • joint servos',result='HOLE IN ONE' if golf.sunk else 'Putting')
            viewer.sync(); time.sleep(max(0,.02-(time.monotonic()-start)))
    finally: viewer.close()
if __name__=='__main__': main()
