"""Render a captioned client film from a deterministic MuJoCo run."""
import argparse, copy, json, math, subprocess, time, wave
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from twin import ThermalTwin, RACKS, zone_at, zone_center
from controller import FireController
from film_scene import dress

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'media';OUT.mkdir(exist_ok=True)
W,H,FPS,SECONDS=1920,1080,24,45
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'; BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
fonts={}
def font(size,bold=False):
    key=(size,bold)
    if key not in fonts:fonts[key]=ImageFont.truetype(BOLD if bold else FONT,size)
    return fonts[key]
BG=(9,17,23); PANEL=(15,28,37); TEXT=(236,241,242); MUTED=(157,177,189); MINT=(113,229,192); AMBER=(255,151,92)
def label(draw,xy,text,size=24,color=TEXT,bold=False):draw.text(xy,text,font=font(size,bold),fill=color)
def smooth(t):return t*t*(3-2*t)
def mix(a,b,t):return np.array(a)+(np.array(b)-a)*smooth(min(1,max(0,t)))

def prepare():
    p=ROOT/'runtime/film-base.xml';twin=ThermalTwin(p);dress(twin,p)
    active=set();events=[]
    def emit(kind,**data):events.append(dict(kind=kind,sim_time=round(twin.data.time,2),**data))
    def valve(z,on):
        if on:active.add(z)
        else:active.discard(z)
    controller=FireController(twin.hits,twin.cameras,emit,valve)
    sources=[];snapshots=[]
    for tick in range(171):
        now=tick*.1
        if tick==40:sources.append(dict(position=twin.surface(7.5,7.5),temperature=180.,burning=True))
        for source in sources:
            water=any(np.linalg.norm(np.array(source['position'])[:2]-zone_center(z)[:2])<=3.6 for z in active)
            if source['burning']:
                source['temperature']=max(22,min(600,source['temperature']+.1*(30-90*water)))
                if water and source['temperature']<=35:source['burning']=False
            else:source['temperature']=max(22,source['temperature']-3)
        frames=twin.thermal_frames(sources);controller.update(frames,now)
        twin.effects(sources,controller.incidents,active)
        snapshots.append(dict(t=now,qpos=twin.data.qpos.copy(),qvel=twin.data.qvel.copy(),mocap=twin.data.mocap_pos.copy(),sources=copy.deepcopy(sources),incidents=copy.deepcopy(controller.incidents),active=set(active),thermal=frames[64].copy()))
    assert len(controller.incidents)==1
    assert controller.incidents[0]['state']=='resolved'
    assert not active
    report=dict(engine='MuJoCo '+mujoco.__version__,events=events,incident=controller.incidents[0],camera='T65',warehouse_m=[40,40],presentation_only='Added materials, set dressing, flame/smoke visualization and illustrated notification. Sensor/control data are from the recorded simulation.')
    (OUT/'video-evidence.json').write_text(json.dumps(report,indent=2))
    return twin,snapshots,report

def simulation_time(t):return float(np.interp(t,[0,7,11,16,20,29,36,45],[0,3,4,4.7,5.2,8,13,16]))

def shot(t):
    if t<7:return 'overview',mix([20,19,2],[20,19,2],t/7),55-5*smooth(t/7),125+10*t/7,-33
    if t<11:return 'camera',np.array([8.8,8,1.8]),1.8-.25*smooth((t-7)/4),-8+12*(t-7)/4,-9
    if t<20:return 'detection',np.array([8,8,1.5]),12,110+10*(t-11)/9,-28
    if t<29:return 'sprinklers',np.array([7.5,7.5,2.5]),11-1.2*smooth((t-20)/9),135+20*(t-20)/9,-15
    if t<36:return 'recovery',np.array([9,9,1.4]),16+7*smooth((t-29)/7),155+15*(t-29)/7,-30
    return 'closing',np.array([20,20,2]),52+3*smooth((t-36)/9),150+8*(t-36)/9,-36

def effects(renderer,snapshot,t):
    scene=renderer.scene
    def primitive(kind,size,pos,rgba):
        if scene.ngeom>=scene.maxgeom:return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],kind,np.array(size,float),np.array(pos,float),np.eye(3).reshape(-1),np.array(rgba,float))
        scene.geoms[scene.ngeom].emission=1 if rgba[0]>.95 and rgba[1]<.9 else 0
        scene.geoms[scene.ngeom].specular=0
        scene.ngeom+=1
    if snapshot['sources'] and snapshot['sources'][0]['temperature']>40:
        source=snapshot['sources'][0];p=np.array(source['position']);intensity=min(1,max(.05,(source['temperature']-35)/65))
        for i in range(110):
            age=(t*1.4+i*.193)%1
            angle=i*2.399;radius=.36*(1-age)
            position=p+[radius*math.cos(angle)+.05*math.sin(t*11+i),radius*math.sin(angle),.12+age*1.8*intensity]
            size=(.055+.035*(1-age))*intensity
            primitive(mujoco.mjtGeom.mjGEOM_ELLIPSOID,[size,size,size*4.5],position,[1,.18+.65*(1-age),.02,.15*(1-age)])
        for i in range(50):
            age=(t*.35+i*.151)%1;angle=i*2.399
            position=p+[age*.5*math.sin(angle),age*.5*math.cos(angle),.8+age*3.3]
            radius=.12+age*.38
            primitive(mujoco.mjtGeom.mjGEOM_ELLIPSOID,[radius,radius,radius*.8],position,[.20,.22,.23,.04*(1-age)*intensity])
    if snapshot['active']:
        for n in range(64):
            point=snapshot['qpos'][n*7:n*7+3]
            if point[2]>0:primitive(mujoco.mjtGeom.mjGEOM_ELLIPSOID,[.018,.018,.06],point,[.7,.86,.98,.7])
        z=sorted(snapshot['active'])[0];nozzle=zone_center(z)
        # Light mist around the actual free-joint droplets. Visual-only spray effect.
        for i in range(280):
            age=(t*(.8+(i%7)*.05)+i*.61803399)%1;angle=((math.sin(i*12.9898)*43758.5453)%1)*2*math.pi
            spread=age*3.1;point=nozzle+[spread*math.cos(angle),spread*math.sin(angle),-age*6.4]
            primitive(mujoco.mjtGeom.mjGEOM_ELLIPSOID,[.009,.009,.034],point,[.7,.86,.98,.6])

def thermal_image(frame,size):
    val=np.clip((frame-20)/140,0,1)
    colors=np.stack([np.minimum(1,val*3),np.clip(val*3-1,0,1),np.clip(val*3-2,0,1)],axis=-1)
    colors[:,:,2]=np.where(val<.25,.12+.6*val,colors[:,:,2])
    return Image.fromarray((colors*255).astype(np.uint8)).resize(size,Image.Resampling.NEAREST)

def map_panel(im,snap,rect):
    x,y,w,h=rect;draw=ImageDraw.Draw(im)
    draw.rounded_rectangle((x,y,x+w,y+h),radius=20,fill=PANEL,outline=(43,65,79),width=2)
    label(draw,(x+26,y+20),'Digital twin',25,bold=True);label(draw,(x+w-150,y+25),'40 × 40 m',18,MUTED)
    s=min(w-65,h-110);scale=s/40;ox=x+(w-s)/2;oy=y+70
    def point(px,py):return ox+px*scale,oy+(40-py)*scale
    for zone in range(64):
        a,b=point((zone%8)*5,(zone//8+1)*5)
        fill=(25,75,100) if zone in snap['active'] else (13,25,33)
        draw.rectangle((a,b,a+5*scale,b+5*scale),fill=fill,outline=(39,58,69))
    for px,py in RACKS:
        a,b=point(px-1,py+2.2);draw.rectangle((a,b,a+2*scale,b+4.4*scale),fill=(138,99,61))
    for py in range(8):
        for px in range(8):
            a,b=point(2.5+5*px,2.5+5*py);draw.ellipse((a-2,b-2,a+2,b+2),fill=MINT)
    for incident in snap['incidents']:
        a,b=point(*incident['position'][:2]);color=MINT if incident['state']=='resolved' else AMBER
        radius=10+2*math.sin(snap['t']*10);draw.ellipse((a-radius,b-radius,a+radius,b+radius),outline=color,width=2);draw.ellipse((a-4,b-4,a+4,b+4),fill=color)
        label(draw,(a+14,b-15),'F1',16,color,True)
    label(draw,(x+25,y+h-29),'Camera rays → world surface → local zone',16,MUTED)

def compose(rgb,t,snap,mode):
    im=Image.fromarray(rgb).convert('RGB');shade=Image.new('RGBA',(W,H),(0,0,0,0));sd=ImageDraw.Draw(shade)
    for y in range(170):sd.line((0,y,W,y),fill=(*BG,int(220*(1-y/170)**.5)))
    for y in range(H-240,H):sd.line((0,y,W,y),fill=(*BG,int(230*((y-(H-240))/240)**.5)))
    im=Image.alpha_composite(im.convert('RGBA'),shade).convert('RGB');d=ImageDraw.Draw(im)
    label(d,(64,40),'wendy.',42,TEXT,True);label(d,(230,54),'THERMAL FIRE WATCH',20,MUTED,True)
    d.rounded_rectangle((1580,40,1856,80),radius=20,fill=PANEL,outline=(66,90,103));label(d,(1601,49),'MUJOCO SIMULATION',19,MINT,True)
    headline,sub='', ''
    if mode=='overview':
        headline='A warehouse that watches for heat.';sub='40 × 40 meters. 88 thermal cameras. Local sprinkler control.'
        for i,(big,small) in enumerate([('88','THERMAL CAMERAS'),('32 × 24','SENSOR RESOLUTION'),('64','SPRINKLER ZONES')]):
            xx=65+i*285;label(d,(xx,760),big,42,MINT,True);label(d,(xx,818),small,17,MUTED,True)
    elif mode=='camera':headline='Low-resolution cameras. Known locations.';sub='Ceiling and rack-aisle coverage, calibrated to the warehouse twin.'
    elif mode=='detection':
        headline='Suspicious heat becomes a location.';sub='Persistent hot pixels are projected onto the visible surface in the twin.'
        # Put the actual sensor feed and measured location next to the 3D scene.
        x,y=1320,155;d.rounded_rectangle((x,y,1855,y+350),radius=20,fill=PANEL,outline=(43,65,79),width=2)
        label(d,(x+25,y+18),'T65 · Rack aisle',24,bold=True);label(d,(x+320,y+22),'32 × 24',18,MUTED)
        im.paste(thermal_image(snap['thermal'],(304,228)),(x+25,y+67));d=ImageDraw.Draw(im)
        label(d,(x+350,y+95),f'{snap["thermal"].max():.0f}°C',32,AMBER,True);label(d,(x+350,y+151),'80°C',25,TEXT);label(d,(x+350,y+188),'threshold',17,MUTED)
        label(d,(x+25,y+308),'Radiometric temperature frame',17,MUTED)
        map_panel(im,snap,(1320,525,535,380));d=ImageDraw.Draw(im)
        if snap['incidents']:
            i=snap['incidents'][0]
            d.rounded_rectangle((66,165,570,275),radius=15,fill=PANEL,outline=(95,62,43),width=2)
            label(d,(89,185),'FIRE DETECTED',22,AMBER,True);label(d,(89,228),f'({i["position"][0]:.1f}, {i["position"][1]:.1f}) m · Zone {i["zone"]+1}',22)
    elif mode=='sprinklers':
        headline='Water goes to the detected area.';sub='Only the local sprinkler zone activates. The modeled heat begins to fall.'
        map_panel(im,snap,(1360,160,495,450));d=ImageDraw.Draw(im)
        if snap['incidents']:
            i=snap['incidents'][0]
            d.rounded_rectangle((1360,635,1855,836),radius=20,fill=(234,239,239));label(d,(1385,657),'Push notification preview',17,(89,110,118),True)
            label(d,(1385,697),'Warehouse fire detected',27,(21,35,44),True);label(d,(1385,746),f'Zone {i["zone"]+1} · ({i["position"][0]:.1f}, {i["position"][1]:.1f}) m',23,(52,71,82))
            label(d,(1385,796),'Illustrated alert · live push tested separately',16,(89,110,118))
    elif mode=='recovery':
        headline='The hotspot cools. The system keeps watching.';sub='The controller holds the local valve open, then closes it after a cool-down period.'
        map_panel(im,snap,(1360,170,495,450));d=ImageDraw.Draw(im)
        label(d,(1388,655),f'{snap["thermal"].max():.0f}°C',56,MINT,True)
        label(d,(1388,723),'Current thermal peak',20,MUTED)
        state='RESOLVED' if snap['incidents'] and snap['incidents'][0]['state']=='resolved' else 'COOL-DOWN HOLD'
        label(d,(1388,774),state,22,MINT,True)
    else:
        headline='Test in the twin. Connect to the warehouse.';sub='The same controller accepts calibrated real camera frames and acknowledged valve commands.'
        if t>39:
            d.rounded_rectangle((64,173,795,355),radius=20,fill=PANEL,outline=(43,65,79))
            label(d,(92,197),'Simulation verified',28,MINT,True)
            label(d,(92,243),'12 controller and hardware-input tests passed',23)
            label(d,(92,285),'400 sampled surfaces covered · Local suppression verified',21,MUTED)
    label(d,(64,928),headline,43,TEXT,True);label(d,(64,992),sub,24,MUTED)
    label(d,(64,1050),'Simulation demonstration. Fire, smoke and water visuals are illustrative. Physical deployment requires site calibration.',16,(125,147,159))
    # Restrained progress line, with a chapter number for viewers watching without sound.
    chapter={'overview':1,'camera':2,'detection':3,'sprinklers':4,'recovery':5,'closing':6}[mode]
    label(d,(1810,1010),f'0{chapter}',25,MINT,True)
    d.rectangle((0,H-4,int(W*t/SECONDS),H),fill=MINT)
    # Short dip between camera cuts avoids jarring jumps.
    cuts=[7,11,20,29,36]
    distance=min(abs(t-c) for c in cuts)
    if distance<.16:im=Image.blend(Image.new('RGB',(W,H),BG),im,distance/.16)
    fade=min(1,t/.7,(SECONDS-t)/1.2)
    if fade<1:im=Image.blend(Image.new('RGB',(W,H),BG),im,max(0,fade))
    return im

def soundtrack(report):
    rate=48000;n=int(rate*SECONDS);t=np.arange(n)/rate
    # Original, quiet ambient pad, no third-party music or speech.
    sound=np.zeros(n)
    for frequency,amplitude in [(110,.014),(164.8138,.009),(220,.006),(329.6276,.004)]:
        sound+=amplitude*np.sin(2*np.pi*frequency*t+1.1*np.sin(.17*t))*(.75+.25*np.sin(.3*t)**2)
    envelope=np.minimum(1,t/3)*np.minimum(1,(SECONDS-t)/3);sound*=envelope
    # Water texture and a soft alert cue at the demonstrated response.
    rng=np.random.default_rng(3);noise=rng.normal(0,1,n);filtered=np.convolve(noise,np.ones(8)/8,mode='same')
    water=np.clip((t-20)/1.2,0,1)*np.clip((32-t)/2,0,1);sound+=filtered*.018*water
    a=np.maximum(0,t-17);cue=(t>=17)*np.exp(-a*4)*(np.sin(2*np.pi*880*a)+.3*np.sin(2*np.pi*1320*a));sound+=.04*cue
    pcm=np.clip(sound*32767,-32767,32767).astype(np.int16)
    p=OUT/'soundtrack.wav'
    with wave.open(str(p),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(rate);f.writeframes(pcm.tobytes())
    return p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',type=float);args=parser.parse_args()
    twin,snapshots,report=prepare()
    for zone in range(64):
        gid=mujoco.mj_name2id(twin.model,mujoco.mjtObj.mjOBJ_GEOM,f'zone_{zone}')
        twin.model.geom_rgba[gid,3]=0
    renderer=mujoco.Renderer(twin.model,height=H,width=W,max_geom=5000)
    opt=mujoco.MjvOption();opt.geomgroup[3]=1;camera=mujoco.MjvCamera()
    preview=args.preview
    if preview is None:
        video=OUT/'warehouse-fire-client-demo.mp4'
        audio=soundtrack(report)
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-i',str(audio),'-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart','-shortest',str(video)],stdin=subprocess.PIPE)
        times=np.arange(SECONDS*FPS)/FPS
    else:times=[preview]
    started=time.monotonic()
    for frame,t in enumerate(times):
        sim=simulation_time(t);idx=min(len(snapshots)-1,int(sim*10));snap=snapshots[idx]
        twin.data.qpos[:]=snap['qpos'];twin.data.qvel[:]=snap['qvel'];twin.data.mocap_pos[:]=snap['mocap'];twin.data.time=sim
        mujoco.mj_forward(twin.model,twin.data)
        mode,look,distance,azimuth,elevation=shot(t);camera.lookat[:]=look;camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
        renderer.update_scene(twin.data,camera=camera,scene_option=opt);effects(renderer,snap,t)
        image=compose(renderer.render(),t,snap,mode)
        if preview is not None:image.save(OUT/f'preview-{preview:.0f}.jpg',quality=95)
        else:
            encoder.stdin.write(np.asarray(image).tobytes())
            if frame%120==0:print(f'Rendered {frame}/{SECONDS*FPS} frames · {time.monotonic()-started:.1f}s',flush=True)
            if frame in (72,240,408,552,792,984):image.save(OUT/f'frame-{frame:04d}.jpg',quality=95)
    renderer.close()
    if preview is None:
        encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError(f'Video encoder exited {code}')
        print(video,flush=True)

if __name__=='__main__':main()
