"""G1 bar delivery: contact physics by default; scripted preview is optional."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from .sim import Plant, scene, ARM
from .walking_policy import WalkingPolicy
from .perception import SimCamera
from .task import execute, OPEN
from . import bar_film as bar

OUTPUT=bar.PROJECT/'results'/'bar-delivery'
POLICY=Path(__file__).resolve().parent/'assets'/'locomotion'
BODY_JOINTS=[f'{side}_{part}_joint' for side in ('left','right') for part in
    ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
BODY_JOINTS += [f'waist_{part}_joint' for part in ('yaw','roll','pitch')]
BODY_JOINTS += [f'{side}_{part}_joint' for side in ('left','right') for part in
    ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
REST=np.array([0,.08,0,1.45,0,0,0])

def camera(world,name,pos,target,fovy=48):
    z=np.asarray(pos)-target;z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    bar.add(world,'camera',name=name,pos=bar.vec(pos),xyaxes=bar.vec(np.r_[x,y]),fovy=fovy)

def decorate(root):
    bar.decorate(root,0)
    world=root.find('worldbody');pelvis=world.find("body[@name='pelvis']")
    bar.add(pelvis,'freejoint',name='floating_base_joint')
    for body in list(world.findall('body')):
        if body.get('name','').startswith(('customer_','actor_')):world.remove(body)
    for eq in root.findall('equality'):root.remove(eq)
    # Remove stools from the walking aisle and the booth beside the counter end.
    for g in list(world.findall('geom')):
        pos=np.fromstring(g.get('pos','0 0 0'),sep=' ')
        if len(pos)==3 and abs(pos[0]-1.09)<.001:world.remove(g)
    room=world.find("body[@name='bar_room']")
    for g in list(room.findall('geom')):
        pos=np.fromstring(g.get('pos','0 0 0'),sep=' ')
        if len(pos)==3 and abs(pos[1]+1.88)<.001:room.remove(g)
    bar.geom(room,'box',[1.95,2.73,1.5],[.80,.045,1.5],'wood')
    for g in world.findall('geom'):
        pos=np.fromstring(g.get('pos','0 0 0'),sep=' ')
        if len(pos)==3 and abs(pos[0]-.695)<.001 and abs(pos[2]-.40)<.001:
            g.set('contype','1');g.set('conaffinity','1');g.set('name','counter_front')
    table=bar.add(world,'body',name='delivery_table')
    bar.geom(table,'box',[1.87,-.42,.725],[.37,.43,.025],'wood',name='delivery_surface',contype='1',conaffinity='1',friction='1 .01 .001')
    for x in (1.59,2.15):
        for y in (-.74,-.10):
            bar.geom(table,'capsule',fromto=bar.vec([x,y,.035,x,y,.70]),size='.027',material='brass',contype='1',conaffinity='1')
    bar.geom(table,'cylinder',[1.62,-.43,.754],[.065,.004],'ivory',name='delivery_coaster')
    camera(world,'start_wide',[1.65,-2.4,1.75],[.20,-.15,.78],46)
    camera(world,'aisle_walk',[-.55,-2.43,1.8],[.12,-.65,.72],49)
    camera(world,'round_counter',[2.45,-2.40,2.15],[.62,-1.24,.62],56)
    camera(world,'route_wide',[2.45,-2.50,2.4],[.40,-1.35,.65],68)
    camera(world,'table_place',[2.57,-1.66,1.45],[1.43,-.5,.95],43)
    camera(world,'table_finish',[2.56,-1.5,1.7],[1.53,-.43,.68],47)

class MobilePlant(Plant):
    def __init__(self,model):
        super().__init__(model)
        self.policy=WalkingPolicy(POLICY)
        self.body_act=np.array([model.actuator(n).id for n in BODY_JOINTS])
        self.bqa=self.qadr[self.body_act];self.bva=self.dadr[self.body_act]
        self.right=np.array([model.actuator(n.replace('left_','right_')).id for n in ARM])
        self.upper=np.r_[self.arm,self.right]
        self.data.qpos[:7]=[0,0,.793,1,0,0,0]
        self.data.qpos[self.bqa]=self.policy.default
        self.data.qpos[self.qadr[self.arm]]=REST
        self.data.qpos[self.qadr[self.right]]=REST*np.array([1,-1,1,1,1,1,1])
        self.targets=self.data.qpos[self.qadr].copy()
        self.kp[self.upper]=80;self.kd[self.upper]=5
        self.body_target=self.policy.default.copy()
        self.command=np.zeros(3);self.applied=np.zeros(3);self.steps=0
        self.anchor=np.array([0.,0.]);self.navigating=False
        self.pelvis=model.body('pelvis').id
        self.min_height=10.;self.min_upright=1.;self.max_arm_error=0.
        mujoco.mj_forward(model,self.data)
        assert model.neq==0

    def step(self,seconds):
        for _ in range(round(seconds/self.model.opt.timestep)):
            m,d=self.model,self.data;rotation=d.xmat[self.pelvis].reshape(3,3)
            if not self.navigating:
                local=rotation[:2,:2].T@(self.anchor-d.qpos[:2])
                self.command[:2]=np.clip(2*local,-.3,.3)
                self.command[2]=np.clip(-np.arctan2(rotation[1,0],rotation[0,0]),-.2,.2)
            self.applied+=np.clip(self.command-self.applied,-np.array([1,1,2])*m.opt.timestep,np.array([1,1,2])*m.opt.timestep)
            if self.steps%10==0:
                policy_q=d.qpos[self.bqa].copy();policy_dq=d.qvel[self.bva].copy()
                # The manipulation controller owns both arms. Preserve the
                # locomotion policy's nominal arm observation for its leg loop.
                policy_q[15:]=self.policy.default[15:];policy_dq[15:]=0
                self.body_target=self.policy.targets(d.sensor('imu-pelvis-angular-velocity').data,
                    rotation.T@np.array([0,0,-1]),self.applied,policy_q,policy_dq)
            effort=self.kp*(self.targets-d.qpos[self.qadr])-self.kd*d.qvel[self.dadr]+d.qfrc_bias[self.dadr]
            effort[self.body_act]=self.policy.kp*(self.body_target-d.qpos[self.bqa])-self.policy.kd*d.qvel[self.bva]
            effort[self.upper]=self.kp[self.upper]*(self.targets[self.upper]-d.qpos[self.qadr[self.upper]])-self.kd[self.upper]*d.qvel[self.dadr[self.upper]]+d.qfrc_bias[self.dadr[self.upper]]
            d.ctrl[:]=np.clip(effort,-self.torque,self.torque)
            mujoco.mj_step(m,d);self.steps+=1
            if not np.isfinite(d.qpos).all():raise ValueError('Non-finite robot pose')
            if d.time>.5:
                self.min_height=min(self.min_height,float(d.qpos[2]));self.min_upright=min(self.min_upright,float(rotation[2,2]))
                if d.qpos[2]<.55 or rotation[2,2]<.7:raise ValueError('Walking controller lost standing balance')
            if self.frame_callback:self.frame_callback(self)

    def pose_collision(self,data):
        arm_prefix=('left_shoulder','left_elbow','left_wrist','left_hand')
        for c in data.contact:
            if c.dist>=-.002:continue
            names=[self.model.body(self.model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
            if 'can' in names:continue
            selected=[n.startswith(arm_prefix) for n in names]
            if any(selected) and not all(selected):return True
        return False

    def move(self,position,duration=2.,rotation=None):
        position=np.asarray(position,dtype=float)
        start=self.targets[self.arm].copy()
        goal=self.solve(position,rotation)
        for index,t in enumerate(np.linspace(0,1,round(duration/.02))):
            if index%5==0:goal=self.solve(position,rotation)
            alpha=t*t*(3-2*t)
            self.targets[self.arm]=start+alpha*(goal-start)
            self.step(.02)
        for _ in range(15):
            self.targets[self.arm]=self.solve(position,rotation)
            self.step(.1)
            if np.linalg.norm(self.data.site_xpos[self.site]-position)<.02:break
        error=float(np.linalg.norm(self.data.site_xpos[self.site]-position))
        if error>.025:raise ValueError(f'World-space arm tracking error {error:.3f}m at {self.data.site_xpos[self.site]}')

    def go(self,goal,timeout=45.):
        goal=np.array(goal,dtype=float);began=self.data.time;self.navigating=True
        while self.data.time-began<timeout:
            rotation=self.data.xmat[self.pelvis].reshape(3,3)
            delta=goal-self.data.qpos[:2]
            if np.linalg.norm(delta)<.075:break
            local=rotation[:2,:2].T@delta
            self.command[:2]=np.clip(1.2*local,-.3,.3)
            yaw=np.arctan2(rotation[1,0],rotation[0,0]);self.command[2]=np.clip(-yaw,-.2,.2)
            self.step(.02)
        self.anchor=goal.copy();self.navigating=False;self.command[:]=0;self.step(1.0)
        error=float(np.linalg.norm(goal-self.data.qpos[:2]))
        if error>.18:raise ValueError(f'Route waypoint not reached: {error:.3f}m')
        return error

def make_plant():
    return MobilePlant(scene(.85,.315,.19,decorate=decorate).model)

def storyboard_scene(root):
    decorate(root)
    world=root.find('worldbody')
    for body in world.iter('body'):
        if body.get('name')=='can':
            for geom in body.findall('geom'):geom.set('friction','2.5 .01 .001')
    bar.add(world,'body',name='route_support',mocap='true',pos='0 0 .793')
    equality=bar.add(root,'equality')
    bar.add(equality,'weld',body1='pelvis',body2='route_support',solref='.004 1',solimp='.99 .99 .001')
    bar.add(equality,'weld',name='carry_support',body1='left_wrist_yaw_link',body2='can',active='false',solref='.004 1')

class StoryboardPlant(Plant):
    """Explicitly supported, choreographed walk; can uses contact dynamics."""
    pose_collision=MobilePlant.pose_collision
    move=MobilePlant.move

    def __init__(self,model):
        super().__init__(model)
        self.right=np.array([model.actuator(n.replace('left_','right_')).id for n in ARM])
        self.body_act=np.array([model.actuator(n).id for n in BODY_JOINTS])
        self.bqa=self.qadr[self.body_act]
        self.nominal=np.array([-.10,0,0,.3,-.2,0]*2+[0]*3+REST.tolist()+(REST*np.array([1,-1,1,1,1,1,1])).tolist())
        self.data.qpos[:7]=[0,0,.793,1,0,0,0]
        self.data.qpos[self.bqa]=self.nominal
        self.targets=self.data.qpos[self.qadr].copy()
        self.support=int(model.body('route_support').mocapid[0])
        self.kp[self.fingers]=24;self.kd[self.fingers]=1
        self.min_height=.793;self.min_upright=1.
        self.data.mocap_pos[self.support]=[0,0,.793]
        mujoco.mj_forward(model,self.data)

    def go(self,goal,timeout=45.):
        goal=np.array(goal,dtype=float)
        start=self.data.mocap_pos[self.support,:2].copy()
        distance=np.linalg.norm(goal-start);duration=distance/.22+1.
        direction=(goal-start)/max(distance,.001)
        self.face(float(np.arctan2(direction[1],direction[0])))
        for t in np.linspace(0,1,round(duration/.02)):
            alpha=t*t*(3-2*t)
            speed=6*t*(1-t)*distance/duration
            gait=np.sin(2*np.pi*(self.data.time)*1.25)*min(speed/.22,1.)
            # A visual gait and root route, not a locomotion policy result.
            self.data.mocap_pos[self.support,:2]=start+alpha*(goal-start)
            self.data.mocap_pos[self.support,2]=.793+.006*abs(gait)
            for side,sign in ((0,1),(6,-1)):
                leg=self.nominal[side:side+6].copy()
                leg[0]+=-sign*.22*gait
                leg[3]+=.23*max(0,sign*gait)
                leg[4]-=.12*max(0,sign*gait)
                self.targets[self.body_act[side:side+6]]=leg
            self.step(.02)
        self.targets[self.body_act[:12]]=self.nominal[:12]
        self.data.mocap_pos[self.support,2]=.793
        self.step(.5)
        return float(np.linalg.norm(goal-self.data.qpos[:2]))

    def face(self,yaw):
        current=self.data.mocap_quat[self.support]
        start=2*np.arctan2(current[3],current[0])
        delta=np.arctan2(np.sin(yaw-start),np.cos(yaw-start))
        for t in np.linspace(0,1,60):
            angle=start+delta*t*t*(3-2*t)
            self.data.mocap_quat[self.support]=[np.cos(angle/2),0,0,np.sin(angle/2)]
            self.step(.02)

def make_storyboard():
    return StoryboardPlant(scene(.85,.315,.19,decorate=storyboard_scene).model)

def simulate(output):
    p=make_storyboard();frames=[];events=[];phase=['arms_at_hips'];last=[-1.]
    def record(plant):
        if plant.data.time-last[0]<1/30-.0001:return
        last[0]=plant.data.time
        frames.append(dict(time=float(plant.data.time),qpos=plant.data.qpos.copy(),mocap_pos=plant.data.mocap_pos.copy(),mocap_quat=plant.data.mocap_quat.copy(),eq_active=plant.data.eq_active.copy(),phase=phase[0]))
    def event(name):
        phase[0]=name;events.append(dict(phase=name,time=float(p.data.time),base=p.data.qpos[:3].tolist()))
        print(name,p.data.qpos[:3].round(3).tolist(),flush=True)
    p.frame_callback=record;record(p);p.step(2.)
    camera_rgbd=SimCamera(p);obs,_=camera_rgbd.observe()
    pickup=execute(p,obs,event,lambda:camera_rgbd.observe()[0]);camera_rgbd.close()
    event('carry_pose')
    rotation=p.data.site_xmat[p.site].reshape(3,3).copy()
    base=p.data.qpos[:3].copy()
    # Keep the verified pickup hold instead of regrasping during transit.
    if len(p.contacts())<2:raise ValueError('Can lost during carry transition')
    # The film is a choreographed preview: a temporary support keeps the can
    # in the hand during the scripted route. Pickup and release are simulated.
    support=p.model.equality('carry_support').id
    hand=p.model.body('left_wrist_yaw_link').id
    hand_r=p.data.xmat[hand].reshape(3,3)
    p.model.eq_data[support,3:6]=hand_r.T@(p.can_position()-p.data.xpos[hand])
    from scipy.spatial.transform import Rotation
    relative=hand_r.T@p.data.xmat[p.can_id].reshape(3,3)
    quat=Rotation.from_matrix(relative).as_quat()
    p.model.eq_data[support,6:10]=quat[[3,0,1,2]]
    p.data.eq_active[support]=1
    local_offset=rotation.T@(p.can_position()-p.data.site_xpos[p.site])
    grasp_relative=rotation.T@p.data.xmat[p.can_id].reshape(3,3)
    place_rotation=grasp_relative.T
    route=[]
    for name,target in [('walk_behind_counter',[0,-2.13]),('round_counter_end',[1.27,-2.13]),
        ('approach_table_corner',[1.27,-1.13]),('walk_along_table',[1.27,-.62])]:
        event(name);error=p.go(target)
        if p.can_position()[2]<.85:raise ValueError('Can below the carry clearance')
        route.append(dict(waypoint=target,error_m=error,actual=p.data.qpos[:2].tolist()))
    event('face_table');p.face(0.)
    event('align_over_table')
    # The destination is a 75 cm table, lower than the 85 cm pickup counter.
    target=np.array([1.62,-.43,.811])
    hand_target=target-place_rotation@local_offset
    p.move(hand_target+[0,0,.16],1.8,place_rotation)
    event('lower_can')
    p.move(hand_target+[0,0,.005],2.0,place_rotation)
    event('release_on_table');p.data.eq_active[support]=0;p.hand(OPEN,1.1);p.step(.6)
    event('withdraw_hand');p.move(hand_target+[0,.075,.16],1.5,place_rotation)
    p.step(1.5)
    position=p.can_position();axis=p.data.xmat[p.can_id].reshape(3,3)[:,2]
    if np.linalg.norm(position[:2]-target[:2])>.12 or abs(position[2]-.811)>.025 or axis[2]<.95:
        raise ValueError(f'Can did not settle upright on the table: {position.tolist()}')
    event('finished');p.step(1.5)
    result=dict(success=True,pickup=pickup,route=route,events=events,
        final_can_position=position.tolist(),can_upright_cosine=float(axis[2]),
        locomotion='Scripted walking animation with mocap pelvis support; not a validated locomotion controller',base_support_constraints=1,
        external_forces=False,object_attachment='temporary scripted carry weld; inactive at pickup and after release',
        grip_friction=2.5,initial_arm_targets=REST.tolist(),hardware_validated=False)
    np.savez_compressed(output/'states.npz',**{key:np.stack([f[key] for f in frames]) for key in frames[0]})
    (output/'simulation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return p,frames

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-walk',action='store_true')
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--simulate-only',action='store_true')
    parser.add_argument('--reuse-states',action='store_true')
    parser.add_argument('--scripted-preview',action='store_true')
    args=parser.parse_args()
    if not args.scripted_preview and not args.preview and not args.check_walk:
        from .bar_delivery_physics import main as physics_main
        physics_main();return
    OUTPUT.mkdir(parents=True,exist_ok=True)
    bar.textures();p=make_plant()
    if args.preview:
        renderer=mujoco.Renderer(p.model,width=1280,height=720)
        options=mujoco.MjvOption();options.geomgroup[3]=0;options.geomgroup[4]=1
        for view in ('start_wide','aisle_walk','round_counter','table_place'):
            renderer.update_scene(p.data,camera=view,scene_option=options)
            Image.fromarray(renderer.render().copy()).save(OUTPUT/(view+'.png'))
        renderer.close();return
    if args.check_walk:
        p.step(2.);start=p.data.qpos[:3].copy()
        p.navigating=True;p.command[:]=[0,-.3,0];p.step(4.);p.command[:]=0;p.step(1.)
        print(json.dumps(dict(start=start.tolist(),end=p.data.qpos[:3].tolist(),
            min_height=p.min_height,min_upright=p.min_upright,policy_updates=p.policy.updates),indent=2))
    elif args.reuse_states:
        archive=np.load(OUTPUT/'states.npz');frames=[{k:archive[k][i] for k in archive.files} for i in range(len(archive['time']))]
        render(make_storyboard(),frames,OUTPUT)
    else:
        plant,frames=simulate(OUTPUT)
        if not args.simulate_only:render(plant,frames,OUTPUT)

def render(plant,frames,output,label='MUJOCO  ·  SCRIPTED MOTION PREVIEW'):
    fps=20;width=1280;height=720
    indices=np.searchsorted([f['time'] for f in frames],np.arange(frames[0]['time'],frames[-1]['time'],1/fps))
    renderer=mujoco.Renderer(plant.model,width=width,height=height)
    options=mujoco.MjvOption();options.geomgroup[3]=0;options.geomgroup[4]=1
    path=output/'simulation-picture.mp4'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17)
    try:
        previous=None
        for count,index in enumerate(indices):
            f=frames[index];plant.data.qpos[:]=f['qpos'];plant.data.mocap_pos[:]=f['mocap_pos'];plant.data.mocap_quat[:]=f['mocap_quat'];mujoco.mj_forward(plant.model,plant.data)
            phase=str(f['phase'])
            if phase in ('arms_at_hips','raise_arm','prepare_reach'):view='aisle_walk'
            elif phase=='walk_behind_counter':view='aisle_walk' if f['time']<22 else 'route_wide'
            elif phase in ('round_counter_end','approach_table_corner','walk_along_table','face_table'):view='route_wide'
            elif phase in ('align_over_table','lower_can','release_on_table','withdraw_hand','restore_upright'):view='table_place'
            elif phase=='finished':view='table_finish'
            else:view='bar_close'
            renderer.update_scene(plant.data,camera=view,scene_option=options)
            picture=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(picture)
            draw.rectangle((24,height-50,412,height-20),fill=(12,15,14));draw.rectangle((24,height-50,27,height-20),fill=(159,226,191))
            draw.text((38,height-46),label,font=font,fill=(225,234,229))
            encoder.stdin.write(picture.tobytes())
            if phase!=previous:picture.save(output/f'frame-{phase}.jpg');previous=phase
            if count%100==0:print(f'Rendering {count}/{len(indices)}',flush=True)
    finally:
        encoder.stdin.close();encoder.wait();renderer.close()
    if encoder.returncode:raise RuntimeError('Video encoder failed')
    end=output/'end-tag.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(bar.MARKETING/'wendy-end-tag.mp4'),'-vf',f'scale={width}:{height},fps={fps}',
        '-an','-c:v','libx264','-crf','18',str(end)],check=True)
    (output/'concat.txt').write_text("file 'simulation-picture.mp4'\nfile 'end-tag.mp4'\n")
    picture_path=output/'picture.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(output/'concat.txt'),'-c','copy',str(picture_path)],check=True)
    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(picture_path)]))
    result=output/'wendy-g1-coke-table-delivery.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(picture_path),'-stream_loop','-1','-i',str(bar.MARKETING/'marketing-music.mp3'),
        '-map','0:v','-map','1:a','-t',str(duration),'-c:v','copy','-af',f'loudnorm=I=-20:TP=-2:LRA=9,afade=t=in:d=1,afade=t=out:st={duration-2}:d=2',
        '-c:a','aac','-b:a','192k','-movflags','+faststart',str(result)],check=True)
    print(f'VIDEO {result} ({duration:.2f}s)',flush=True)

if __name__=='__main__':main()
