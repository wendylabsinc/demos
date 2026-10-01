"""Render a MuJoCo bar film using contact grasps and scripted customer actors.

The existing observed-pose pickup controller drives the G1. Customer animation
and a recipient support weld stage the handoff; this is not a human-interaction
policy or a physical-hardware demonstration.
"""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .sim import scene
from .perception import SimCamera
from .task import execute, accept

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT / 'results' / 'bar-sim'
ASSETS = OUTPUT / 'assets'
MARKETING = PROJECT / 'results' / 'bartender' / 'assets'
FPS = 30
HEIGHT = .85

def vec(v):
    if isinstance(v,str):return v
    return ' '.join(str(float(x)) for x in v)

def add(parent, tag, **kwargs):
    return ET.SubElement(parent, tag, {k: str(v) for k,v in kwargs.items()})

def geom(parent, kind, pos=None, size=None, material=None, **kw):
    args = dict(type=kind, contype='0', conaffinity='0', group='0')
    if pos is not None: args['pos'] = vec(pos)
    if size is not None: args['size'] = vec(size)
    if material is not None: args['material'] = material
    return add(parent, 'geom', **(args | kw))

def textures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    y,x = np.indices((1024,1024))
    grain = np.sin(x*.09+np.sin(y*.013)*1.5)*3 + rng.normal(0,1.8,x.shape)
    planks = (x//170)%3
    wood = np.stack([100+grain+planks*6, 75+grain+planks*4, 48+grain+planks*2],-1)
    wood[(x%170)<3] *= .50
    Image.fromarray(np.clip(wood,0,255).astype('uint8')).save(ASSETS/'wood.png')
    brick = np.zeros((1024,1024,3),dtype='uint8')
    rows = y//82
    mortar = (y%82<7) | ((x+(rows%2)*100)%200<7)
    noise = rng.normal(0,5,x.shape)
    for i,base in enumerate([95,68,52]): brick[:,:,i]=np.clip(base+noise+((x//200+rows)%3)*5,0,255)
    brick[mortar] = [48,44,40]
    Image.fromarray(brick).save(ASSETS/'brick.png')
    floor = np.zeros((1024,1024,3),dtype='uint8')
    checks = (x//128+y//128)%2
    for i,base in enumerate([62,65,61]): floor[:,:,i] = base+checks*9
    floor[(x%128<4)|(y%128<4)]=[28,30,28]
    Image.fromarray(floor).save(ASSETS/'floor.png')

    label=Image.new('RGB',(1024,512),(182,10,21));d=ImageDraw.Draw(label)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/SnellRoundhand.ttc',150)
    small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',23)
    for center in (256,768):
        d.text((center,220),'Coca-Cola',font=font,anchor='mm',fill='white',stroke_width=1)
        d.text((center,335),'ORIGINAL TASTE',font=small,anchor='mm',fill='white')
        d.text((center,433),'330 ml',font=small,anchor='mm',fill='white')
    # UV side wall, explicitly textured rather than a painted screen overlay.
    label.save(ASSETS/'cola.png')
    n=96; lines=[]
    for i in range(n+1):
        a=i/n*2*np.pi
        for z in (-.058,.058): lines.append(f'v {.033*np.cos(a)} {.033*np.sin(a)} {z}')
    for i in range(n+1):
        for v in (1,0): lines.append(f'vt {i/n} {v}')
    for i in range(n):
        a=2*i+1;b=a+1;c=a+2;e=a+3
        lines += [f'f {a}/{a} {c}/{c} {b}/{b}',f'f {b}/{b} {c}/{c} {e}/{e}']
    (ASSETS/'cola-wall.obj').write_text('\n'.join(lines))
    logo=Image.open(MARKETING/'wendy-logo.jpg').convert('RGB')
    sign=Image.new('RGB',(1024,384),(232,228,210))
    logo=ImageOps.contain(logo,(850,230))
    sign.paste(logo,((1024-logo.width)//2,(384-logo.height)//2))
    sign.save(ASSETS/'wendy-sign.png')
    menu=Image.new('RGB',(512,768),(17,26,23));d=ImageDraw.Draw(menu)
    menu_font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',56)
    d.text((256,90),'ON THE HOUSE',font=menu_font,anchor='mm',fill='#9FE2BF')
    d.line((65,145,450,145),fill='#9FE2BF',width=3)
    d.text((256,290),'COLD COKE',font=menu_font,anchor='mm',fill='#F5F0DD')
    d.text((256,425),'GOOD COMPANY',font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',42),anchor='mm',fill='#F5F0DD')
    d.text((256,650),'WENDY',font=menu_font,anchor='mm',fill='#9FE2BF')
    menu.save(ASSETS/'menu.png')

def textured_plane(asset,world,name,file,center,half_y,half_z):
    # Wall decal normal points +X; UV gives a readable sign in the 3-D scene.
    vertex=vec([0,-half_y,-half_z,0,half_y,-half_z,0,half_y,half_z,0,-half_y,half_z])
    add(asset,'mesh',name=name+'_mesh',vertex=vertex,face='0 1 2 0 2 3',texcoord='0 1 1 1 1 0 0 0',inertia='shell')
    add(asset,'texture',name=name+'_tex',type='2d',file=str(file))
    add(asset,'material',name=name,texture=name+'_tex',texuniform='false',rgba='1 1 1 1',specular='.1')
    geom(world,'mesh',center,material=name,mesh=name+'_mesh')

def cola(parent,asset,name,position=None):
    target=parent if position is None else add(parent,'body',name=name,pos=vec(position))
    geom(target,'mesh',material='cola',mesh='cola_wall')
    for z in (-.06,.06):
        geom(target,'cylinder',[0,0,z],[.0335,.0018],'chrome')
    geom(target,'cylinder',[0,0,.0618],[.031,.0006],'chrome')
    geom(target,'ellipsoid',[0,-.005,.063],[.009,.015,.0007],'darkmetal')
    geom(target,'ellipsoid',[0,.006,.064],[.006,.010,.0011],'chrome')
    if position is not None:
        # Set dressing is outside the selected pickup observation layer.
        for g in target.findall('geom'):g.set('group','4')

CUSTOMERS=[dict(x=1.02,y=-.12,skin='.68 .43 .30 1',shirt='olive',hair='hair_dark'),
           dict(x=1.04,y=.72,skin='.78 .57 .40 1',shirt='navy',hair='hair_brown')]

def customer(world,i,c):
    x,y=c['x'],c['y'];skin=c['skin']
    b=add(world,'body',name=f'customer_{i}',pos=vec([x,y,0]))
    # Seated clothed figures, with face details and articulated reaching arms.
    geom(b,'ellipsoid',[-.035,0,.96],[.135,.18,.235],c['shirt'])
    geom(b,'ellipsoid',[.005,0,.70],[.14,.16,.10],'denim')
    for sign in (-1,1):
        geom(b,'capsule',fromto=vec([0,sign*.085,.67,-.28,sign*.085,.61]),size='.069',material='denim')
        geom(b,'capsule',fromto=vec([-.28,sign*.085,.61,-.28,sign*.085,.21]),size='.05',material='denim')
        geom(b,'ellipsoid',[-.32,sign*.085,.16],[.11,.055,.055],'darkmetal')
    geom(b,'capsule',[-.045,0,1.18],[.05,.07],rgba=skin)
    geom(b,'ellipsoid',[-.055,0,1.34],[.09,.08,.125],rgba=skin)
    geom(b,'ellipsoid',[-.027,0,1.40],[.093,.084,.083],c['hair'])
    if i==0:
        geom(b,'ellipsoid',[.017,0,1.29],[.075,.087,.15],c['hair'])
    else:
        geom(b,'ellipsoid',[-.085,0,1.25],[.072,.069,.029],'hair_brown')
    geom(b,'ellipsoid',[-.141,0,1.331],[.021,.021,.028],rgba=skin)
    for sy in (-1,1):
        geom(b,'ellipsoid',[-.131,sy*.036,1.36],[.008,.016,.009],'ivory')
        geom(b,'sphere',[-.138,sy*.034,1.36],[.005],'darkmetal')
        geom(b,'capsule',fromto=vec([-.135,sy*.020,1.381,-.134,sy*.052,1.38]),size='.004',material=c['hair'])
        geom(b,'sphere',[-.046,sy*.083,1.337],[.019],rgba=skin)
    geom(b,'capsule',fromto=vec([-.140,-.021,1.292,-.141,.021,1.292]),size='.004',rgba='.38 .16 .14 1')
    # Resting left arm.
    geom(b,'capsule',fromto=vec([-.035,.15,1.10,-.18,.20,.90]),size='.048',material=c['shirt'])
    geom(b,'capsule',fromto=vec([-.18,.20,.90,-.36,.13,.89]),size='.035',rgba=skin)
    geom(b,'ellipsoid',[-.37,.13,.89],[.045,.025,.018],rgba=skin)
    for part,r,mat in [('upper',.047,c['shirt']),('fore',.032,None)]:
        body=add(world,'body',name=f'actor_{i}_{part}',mocap='true',pos='0 0 0')
        args={'material':mat} if mat else {'rgba':skin}
        geom(body,'capsule',size=[r,.15],name=f'actor_{i}_{part}_geom',**args)
    hand=add(world,'body',name=f'actor_{i}_hand',mocap='true',pos=vec([.7,y,.91]))
    geom(hand,'ellipsoid',[0,0,0],[.017,.037,.045],rgba=skin)
    for z in (-.028,-.009,.010,.029):
        geom(hand,'capsule',fromto=vec([0,-.027,z,-.032,-.036,z]),size='.007',rgba=skin)
        geom(hand,'capsule',fromto=vec([-.032,-.036,z,-.049,-.010,z]),size='.007',rgba=skin)
    geom(hand,'capsule',fromto=vec([.001,.028,-.02,-.032,.031,-.03]),size='.009',rgba=skin)

def decorate(root, recipient=0):
    asset=root.find('asset');world=root.find('worldbody');visual=root.find('visual')
    for global_element in root.findall('visual/global'):
        global_element.set('offwidth','1280');global_element.set('offheight','720')
    quality=visual.find('quality')
    if quality is None: quality=add(visual,'quality')
    quality.set('shadowsize','1024');quality.set('offsamples','1')
    headlight=visual.find('headlight')
    if headlight is None:headlight=add(visual,'headlight')
    headlight.set('ambient','.18 .17 .16');headlight.set('diffuse','.40 .38 .34');headlight.set('specular','.35 .35 .35')
    # Render visual meshes once, keeping all original collision geometry active.
    for g in root.findall('.//geom'):
        if g.get('contype')!='0' and g.get('type')=='mesh':g.set('group','3')
    for site in root.findall('.//site'):site.set('rgba','0 0 0 0')
    mats={
        'brass':('.54 .37 .13 1','.8','.7'), 'chrome':('.76 .79 .8 1','.9','.65'),
        'darkmetal':('.025 .030 .031 1','.4','.4'), 'ivory':('.94 .9 .77 1','.2','.2'),
        'olive':('.20 .29 .20 1','.1','.1'), 'navy':('.08 .13 .17 1','.1','.1'),
        'denim':('.045 .07 .11 1','.1','.1'), 'hair_dark':('.045 .026 .02 1','.2','.2'),
        'hair_brown':('.11 .065 .036 1','.2','.2'), 'green_glass':('.08 .26 .16 1','.8','.8'),
        'amber_glass':('.37 .22 .05 1','.8','.8'), 'clear_glass':('.65 .76 .73 .30','.9','.9'),
        'warm_glow':('1 .63 .24 1','.1','.1'), 'mint_glow':('.4 .95 .78 1','.2','.2')}
    for name,(rgba,specular,shininess) in mats.items():
        kw={'emission':'.75'} if name.endswith('glow') else {}
        add(asset,'material',name=name,rgba=rgba,specular=specular,shininess=shininess,**kw)
    for name in ('wood','brick','floor'):
        add(asset,'texture',name=name+'_tex',type='2d',file=str(ASSETS/(name+'.png')))
        add(asset,'material',name=name,texture=name+'_tex',texrepeat='2 2',texuniform='true',
            specular='.45' if name=='wood' else '.08',shininess='.35' if name=='wood' else '.1',reflectance='.10' if name=='wood' else '.0')
    add(asset,'mesh',name='cola_wall',file=str(ASSETS/'cola-wall.obj'))
    add(asset,'texture',name='cola_tex',type='2d',file=str(ASSETS/'cola.png'))
    add(asset,'material',name='cola',texture='cola_tex',texuniform='false',specular='.75',shininess='.65')
    floor=root.find(".//geom[@name='floor']");floor.set('material','floor');floor.attrib.pop('rgba',None);floor.set('size','5 5 .1')
    table=world.find("geom[@name='table']");table.set('size','.25 1.85 .035');table.set('pos',vec([.48,.18,HEIGHT-.035]));table.set('material','wood');table.attrib.pop('rgba',None)
    geom(world,'box',[.695,.18,.40],[.022,1.85,.40],'wood')
    geom(world,'box',[.695,.18,.63],[.030,1.86,.015],'brass')
    geom(world,'box',[.695,.18,.14],[.032,1.86,.02],'brass')
    for yy in np.linspace(-1.6,1.9,8):geom(world,'box',[.725,yy,.38],[.006,.011,.22],'brass')
    geom(world,'capsule',fromto=vec([.9,-1.7,.18,.9,2.05,.18]),size='.023',material='brass')
    # Brick back wall, walnut bottle shelving, warm practical lights.
    geom(world,'box',[-1.1,.2,1.5],[.04,2.7,1.5],'brick')
    textured_plane(asset,world,'brick_backdrop',ASSETS/'brick.png',[-1.052,.2,1.5],2.7,1.5)
    geom(world,'box',[.2,2.8,1.5],[1.3,.04,1.5],'brick')
    for zz in (.40,.95,1.57,2.12):
        geom(world,'box',[-.87,.2,zz],[.20,2.55,.024],'wood')
        geom(world,'box',[-.67,.2,zz+.016],[.006,2.52,.010],'warm_glow')
    rng=np.random.default_rng(9)
    for zz in (.425,.975,1.595):
        for yy in np.arange(-2.2,2.65,.22):
            # Leave central space for the Wendy plaque.
            if zz>.9 and abs(yy-.18)<.70:continue
            h=float(rng.uniform(.17,.27));mat='green_glass' if rng.random()>.4 else 'amber_glass'
            geom(world,'cylinder',[-.84,yy,zz+h*.37],[.037,h*.37],mat)
            geom(world,'cylinder',[-.84,yy,zz+h*.83],[.015,h*.17],mat)
            geom(world,'cylinder',[-.84,yy,zz+h],[.017,.012],'brass')
            geom(world,'cylinder',[-.84,yy,zz+h*.37],[.0375,.025],'ivory')
    geom(world,'box',[-1.015,.18,1.84],[.018,.65,.21],'brass')
    textured_plane(asset,world,'wendy_sign',ASSETS/'wendy-sign.png',[-.992,.18,1.84],.61,.18)
    geom(world,'box',[-1.035,1.55,1.50],[.022,.29,.40],'wood')
    textured_plane(asset,world,'menu',ASSETS/'menu.png',[-1.008,1.55,1.50],.265,.375)
    # Stools, coasters and glasses establish a usable customer side.
    for yy in (-.12,.72,1.56):
        geom(world,'cylinder',[1.09,yy,.58],[.20,.045],'olive')
        geom(world,'cylinder',[1.09,yy,.29],[.029,.29],'brass')
        geom(world,'cylinder',[1.09,yy,.022],[.235,.020],'darkmetal')
    for yy in (-1.05,1.3):
        geom(world,'cylinder',[.54,yy,HEIGHT+.006],[.07,.005],'ivory')
        geom(world,'cylinder',[.54,yy,HEIGHT+.067],[.035,.06],'clear_glass')
    for yy in (-1.35,1.65):
        geom(world,'cylinder',[.45,yy,HEIGHT+.055],[.032,.053],'clear_glass')
        geom(world,'cylinder',[.45,yy,HEIGHT+.052],[.023,.027],'warm_glow')
    for yy in (-1.5,.1,1.75):
        geom(world,'capsule',fromto=vec([.08,yy,2.5,.08,yy,2.17]),size='.007',material='darkmetal')
        geom(world,'cylinder',[.08,yy,2.13],[.17,.035],'brass')
        geom(world,'sphere',[.08,yy,2.085],[.048],'warm_glow')
        add(world,'light',pos=vec([.10,yy,2.06]),dir='0 0 -1',diffuse='.8 .66 .45',specular='.45 .4 .25',cutoff='65')
    original=world.findall('light')[0];original.set('diffuse','.5 .46 .38');original.set('specular','.3 .3 .3')
    add(world,'light',pos='1.8 -1.4 2.8',dir='-1 0 -1',diffuse='.5 .52 .54',specular='.5 .5 .5')
    # One key shadow avoids six expensive shadow-map passes per film frame.
    for light in root.findall('.//light'):light.set('castshadow','false')
    world.findall('light')[-1].set('castshadow','true')
    can=world.find("body[@name='can']");shell=can.find("geom[@name='can_shell']");shell.set('group','3')
    can.remove(can.find("geom[@name='can_top']"));cola(can,asset,'primary')
    torso=root.find(".//body[@name='torso_link']")
    add(asset,'material',name='face_glow',rgba='.05 .5 1 1',emission='.8')
    geom(torso,'ellipsoid',[.067,0,.375],[.012,.054,.072],'darkmetal',group='1')
    angles=np.linspace(0,2*np.pi,45)
    for a,b in zip(angles[:-1],angles[1:]):
        geom(torso,'capsule',fromto=vec([.081,.054*np.cos(a),.375+.073*np.sin(a),.081,.054*np.cos(b),.375+.073*np.sin(b)]),size='.0018',material='face_glow',group='1')
    geom(torso,'capsule',fromto=vec([.083,-.030,.389,.083,.030,.389]),size='.002',material='face_glow',group='1')
    # A spare can stays behind the observation camera until the second take.
    cola(world,asset,'stock_can',[-.69,-.35,.488])
    if recipient==1:cola(world,asset,'served_can',[.63,-.12,HEIGHT+.063])
    for i,c in enumerate(CUSTOMERS):customer(world,i,c)
    equality=root.find('equality')
    if equality is None:equality=add(root,'equality')
    add(equality,'weld',name='recipient_support',body1=f'actor_{recipient}_hand',body2='can',active='false',solref='.015 1',relpose='-.052 0 -.02 1 0 0 0')
    for name,pos,target,fovy in [
        ('bar_wide',[2.45,-2.55,1.72],[.23,.20,1.06],43),
        ('bar_close',[1.40,-1.0,1.48],[.28,.18,1.015],40),
        ('bar_close_reverse',[1.40,1.55,1.48],[.30,.19,1.015],40),
        ('bar_reverse',[1.6,2.8,1.70],[.28,.32,1.03],43)]:
        z=np.array(pos)-target;z/=np.linalg.norm(z);xx=np.cross([0,0,1],z);xx/=np.linalg.norm(xx);yy=np.cross(z,xx)
        add(world,'camera',name=name,pos=vec(pos),xyaxes=vec(np.r_[xx,yy]),fovy=str(fovy))
    # Close the customer side of the room, including the reverse camera view.
    room=add(world,'body',name='bar_room')
    geom(room,'box',[2.78,.2,1.5],[.045,2.62,1.5],'wood')
    geom(room,'box',[.8,-2.62,1.5],[1.94,.045,1.5],'wood')
    for yy in (-1.1,.8):
        geom(room,'box',[2.72,yy,1.65],[.023,.56,.65],'brass')
        geom(room,'box',[2.69,yy,1.65],[.009,.53,.62],'darkmetal')
        geom(room,'box',[2.675,yy,1.65],[.013,.014,.62],'brass')
        geom(room,'box',[2.675,yy,1.65],[.013,.53,.014],'brass')
    geom(room,'box',[2.2,-1.88,.44],[.28,.62,.08],'olive')
    geom(room,'box',[2.46,-1.88,.81],[.07,.62,.30],'olive')
    geom(room,'box',[1.69,-1.88,.70],[.30,.55,.032],'wood')
    geom(room,'cylinder',[1.69,-1.88,.35],[.034,.35],'brass')

def quat_for_axis(axis):
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    q=np.zeros(4);mujoco.mju_quatZ2Vec(q,axis);return q

def actor_pose(plant,index,hand_pos,hand_quat=None):
    m,d=plant.model,plant.data;c=CUSTOMERS[index]
    shoulder=np.array([c['x']-.105,c['y']-.13,1.115])
    wrist=np.array(hand_pos)+np.array([.022,0,0])
    delta=wrist-shoulder;distance=np.linalg.norm(delta);direction=delta/max(distance,1e-8)
    l1,l2=.34,.32
    along=np.clip((l1*l1-l2*l2+distance*distance)/(2*max(distance,1e-8)),0,l1)
    bend=np.cross(direction,[0,0,1]);bend/=max(np.linalg.norm(bend),1e-8)
    elbow=shoulder+direction*along+bend*np.sqrt(max(0,l1*l1-along*along))
    for part,start,end in [('upper',shoulder,elbow),('fore',elbow,wrist)]:
        bid=m.body(f'actor_{index}_{part}').id;mid=m.body_mocapid[bid]
        d.mocap_pos[mid]=(start+end)/2;d.mocap_quat[mid]=quat_for_axis(end-start)
        m.geom_size[m.geom(f'actor_{index}_{part}_geom').id,1]=np.linalg.norm(end-start)/2
    mid=m.body_mocapid[m.body(f'actor_{index}_hand').id]
    d.mocap_pos[mid]=hand_pos;d.mocap_quat[mid]=[1,0,0,0] if hand_quat is None else hand_quat

def simulate(recipient, x, y, output):
    p=scene(HEIGHT,x,y,decorate=lambda root:decorate(root,recipient))
    for i,c in enumerate(CUSTOMERS):actor_pose(p,i,[.68,c['y']-.11,.92])
    p.step(.3)
    p.move([.28,.19,1.18],.6)
    camera=SimCamera(p)
    frames=[];events=[];last=[-1.];phase=['establish']
    def snapshot(plant):
        if plant.data.time-last[0] < 1/FPS-.0001:return
        last[0]=plant.data.time
        frames.append(dict(time=float(plant.data.time),qpos=plant.data.qpos.copy(),
            mocap_pos=plant.data.mocap_pos.copy(),mocap_quat=plant.data.mocap_quat.copy(),
            geom_size=plant.model.geom_size.copy(),phase=phase[0]))
    def event(name):
        phase[0]=name;events.append(dict(phase=name,time=float(p.data.time)))
        print(f'Customer {recipient+1}: {name}',flush=True)
    p.frame_callback=snapshot;p.step(1.5)
    obs,rgb=camera.observe();Image.fromarray(rgb).save(output/f'perception-{recipient+1}.png')
    result=execute(p,obs,event,lambda:camera.observe()[0])
    event('customer_reaches')
    can=p.can_position();target=can+[.052,0,.02]
    mid=p.model.body_mocapid[p.model.body(f'actor_{recipient}_hand').id]
    start=p.data.mocap_pos[mid].copy()
    cq=p.data.xquat[p.can_id].copy()
    for t in np.linspace(0,1,100):
        a=t*t*(3-2*t);actor_pose(p,recipient,start+(target-start)*a,cq);p.step(.02)
    # Recipient reaches the actual held can; activate support only at contact.
    support=p.model.equality('recipient_support').id
    hand_id=p.model.body(f'actor_{recipient}_hand').id
    mujoco.mj_forward(p.model,p.data)
    R=p.data.xmat[hand_id].reshape(3,3)
    p.model.eq_data[support,3:6]=R.T@(p.can_position()-p.data.xpos[hand_id])
    p.model.eq_data[support,6:10]=[1,0,0,0]
    p.data.eq_active[support]=1
    event('release_to_customer');accept(p,recipient_supports_can=True)
    received_start=p.data.mocap_pos[mid].copy()
    end=np.array([.72,CUSTOMERS[recipient]['y']-.05,1.04])
    event('customer_receives')
    for t in np.linspace(0,1,100):
        a=t*t*(3-2*t);actor_pose(p,recipient,received_start+(end-received_start)*a,cq);p.step(.02)
    # Withdraw the open Dex3 through the original collision-checked approach.
    event('arm_retracts')
    p.move([.28,.19,1.18],1.5)
    p.step(1.)
    expected=p.data.xpos[hand_id]+p.data.xmat[hand_id].reshape(3,3)@p.model.eq_data[support,3:6]
    support_error=float(np.linalg.norm(p.can_position()-expected))
    if support_error>.02:raise ValueError('Can failed to remain supported by the recipient actor')
    result.update(status='served_to_scripted_customer',success=True,
        recipient_support_error_m=support_error,customer=recipient+1,recipient_support='scripted_mocap_weld',
        pickup_controller='RGB-D observed pose and contact verification',
        final_can_position=p.can_position().tolist(),events=events,frames=len(frames))
    camera.close();return p,frames,result

def render(plant,frames,path,recipient,width=1280,height=720):
    renderer=mujoco.Renderer(plant.model,width=width,height=height)
    options=mujoco.MjvOption();options.geomgroup[3]=0;options.geomgroup[4]=1;options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=False
    ff=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    first=frames[0]['time']
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',18)
    try:
        for idx,f in enumerate(frames):
            plant.data.qpos[:]=f['qpos'];plant.data.mocap_pos[:]=f['mocap_pos'];plant.data.mocap_quat[:]=f['mocap_quat'];plant.model.geom_size[:len(f['geom_size'])]=f['geom_size']
            mujoco.mj_forward(plant.model,plant.data)
            phase=f['phase']
            if phase in ('align','reach','close','customer_reaches','release_to_customer'):
                view='bar_close' if recipient==0 else 'bar_close_reverse'
            else:view='bar_wide' if recipient==0 else 'bar_reverse'
            renderer.update_scene(plant.data,camera=view,scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=True
            image=Image.fromarray(renderer.render().copy())
            d=ImageDraw.Draw(image);d.rectangle((24,height-49,248,height-20),fill=(12,15,14))
            d.rectangle((24,height-49,27,height-20),fill=(159,226,191))
            d.text((38,height-46),'MUJOCO SIMULATION',font=font,fill=(225,234,229))
            ff.stdin.write(image.tobytes())
            if idx in (0,len(frames)//2,len(frames)-1):image.save(path.with_name(f'{path.stem}-{idx}.png'))
            if idx%150==0:print(f'Rendering customer {recipient+1}: {idx}/{len(frames)}',flush=True)
    finally:
        ff.stdin.close();ff.wait();renderer.close()
    if ff.returncode:raise RuntimeError('Video encoder failed')

def finish(output):
    paths=[output/'customer-1.mp4',output/'customer-2.mp4']
    end=output/'end-tag.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(MARKETING/'wendy-end-tag.mp4'),
        '-vf','scale=1280:720,fps=30','-an','-c:v','libx264','-crf','18',str(end)],check=True)
    paths.append(end)
    (output/'concat.txt').write_text(''.join(f"file '{p.name}'\n" for p in paths))
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(output/'concat.txt'),
        '-c','copy',str(output/'picture.mp4')],check=True)
    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(output/'picture.mp4')]))
    result=output/'wendy-g1-bartender-full-simulation.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(output/'picture.mp4'),
        '-stream_loop','-1','-i',str(MARKETING/'marketing-music.mp3'),'-map','0:v','-map','1:a',
        '-t',str(duration),'-c:v','copy','-af',f'loudnorm=I=-20:TP=-2:LRA=9,afade=t=in:d=1,afade=t=out:st={duration-2}:d=2',
        '-c:a','aac','-b:a','192k','-movflags','+faststart',str(result)],check=True)
    return edit_reel(output),29.0

def edit_reel(output):
    """Make a short film with cuts only; all action retains simulation speed."""
    output=Path(output)
    picture=output/'reel-picture.mp4'
    filters=(
        '[0:v]split=3[a][b][c];'
        '[a]trim=start=0:end=1,setpts=PTS-STARTPTS[a0];'
        '[b]trim=start=6.6:end=10.4,setpts=PTS-STARTPTS[a1];'
        '[c]trim=start=13:end=20.2,setpts=PTS-STARTPTS[a2];'
        '[1:v]split=2[d][e];'
        '[d]trim=start=0:end=1,setpts=PTS-STARTPTS[b0];'
        '[e]trim=start=10.5:end=20.5,setpts=PTS-STARTPTS[b1];'
        '[2:v]setpts=PTS-STARTPTS[end];'
        '[a0][a1][a2][b0][b1][end]concat=n=6:v=1:a=0[v]')
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(output/'customer-1.mp4'),
        '-i',str(output/'customer-2.mp4'),'-i',str(output/'end-tag.mp4'),
        '-filter_complex',filters,'-map','[v]','-an','-c:v','libx264',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p',str(picture)],check=True)
    result=output/'wendy-g1-bartender-simulation.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(picture),
        '-stream_loop','-1','-i',str(MARKETING/'marketing-music.mp3'),
        '-map','0:v','-map','1:a','-t','29','-c:v','copy','-af',
        'loudnorm=I=-20:TP=-2:LRA=9,afade=t=in:d=0.5,afade=t=out:st=27:d=2',
        '-c:a','aac','-b:a','192k','-movflags','+faststart',str(result)],check=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview',action='store_true');parser.add_argument('--reuse-states',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);textures()
    if args.preview:
        p=scene(HEIGHT,.315,.19,decorate=lambda root:decorate(root,0))
        for i,c in enumerate(CUSTOMERS):actor_pose(p,i,[.68,c['y']-.11,.92])
        p.step(.3);renderer=mujoco.Renderer(p.model,width=1280,height=720);opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
        for name in ('bar_wide','bar_close','bar_reverse'):
            renderer.update_scene(p.data,camera=name,scene_option=opt);Image.fromarray(renderer.render().copy()).save(args.output/(name+'.png'))
        renderer.close();print(str(args.output/'bar_wide.png'));return
    outcomes=[]
    for recipient,x,y in [(0,.315,.19),(1,.315,.19)]:
        cache=args.output/f'customer-{recipient+1}-states.npz'
        clip=args.output/f'customer-{recipient+1}.mp4'
        if args.resume and cache.exists() and clip.exists():
            probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',str(clip)],capture_output=True,text=True)
            if probe.returncode==0 and float(probe.stdout)>20:
                outcomes.append(json.loads((args.output/'simulation.json').read_text())[recipient])
                print(f'Reusing completed customer {recipient+1} clip',flush=True)
                continue
        if args.reuse_states and cache.exists():
            p=scene(HEIGHT,x,y,decorate=lambda root:decorate(root,recipient))
            saved=np.load(cache)
            frames=[{key:saved[key][i] for key in saved.files} for i in range(len(saved['time']))]
            outcome=json.loads((args.output/'simulation.json').read_text())[recipient]
            final=frames[-1]
            p.data.qpos[:]=final['qpos'];p.data.mocap_pos[:]=final['mocap_pos'];p.data.mocap_quat[:]=final['mocap_quat']
            mujoco.mj_forward(p.model,p.data)
            outcome.update(status='served_to_scripted_customer',success=True,
                final_can_position=p.can_position().tolist(),rendered_from_saved_simulation_states=True)
        else:
            p,frames,outcome=simulate(recipient,x,y,args.output)
            np.savez_compressed(cache,**{key:np.stack([frame[key] for frame in frames]) for key in frames[0]})
        outcomes.append(outcome)
        (args.output/'simulation.json').write_text(json.dumps(outcomes,indent=2))
        render(p,frames,args.output/f'customer-{recipient+1}.mp4',recipient)
    result,duration=finish(args.output)
    print(json.dumps(dict(video=str(result),duration_seconds=duration,customers=len(outcomes)),indent=2))

if __name__=='__main__':main()
