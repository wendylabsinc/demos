"""Warehouse IP-camera search, YOLO perception, twin raycast and physical Go2."""
import os, sys, math, time, json, threading, heapq, wave
from pathlib import Path
ROOT = Path(__file__).resolve().parent
runtime = Path(os.environ.get('WENDY_SIM_RUNTIME', ROOT.parent.parent / 'wendy-sandbox/image/sim')).expanduser()
if runtime.is_dir():
    sys.path.insert(0, str(runtime))
os.environ['WENDY_WORLDSIM_DIR'] = str(ROOT / 'runtime/warehouse')
import numpy as np
import mujoco, cv2
try:
    import go2rl, wendymujoco
except ModuleNotFoundError as exc:
    if exc.name not in ('go2rl', 'wendymujoco'):
        raise
    raise RuntimeError(
        'The warehouse demo requires the licensed Wendy Sandbox simulator runtime. '
        'Set WENDY_SIM_RUNTIME to its image/sim directory, containing go2rl.py, '
        'wendymujoco.py and assets/go2_policy.onnx with its matching configuration.'
    ) from exc
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
import uvicorn

W, H = 960, 720
CORNERS = [(-5.7,-4.7,3.8),(5.7,-4.7,3.8),(5.7,4.7,3.8),(-5.7,4.7,3.8)]
SHELVES = [(-1.7,0,.55,2.0),(1.7,0,.55,2.0)]
shared = dict(phase='loading', cameras={}, events=[], route=[], position=[0,0], bark=0)
lock = threading.Lock()
commands = {'reset':False,'paused':False,'stop':False,'steps':0}
app = FastAPI()

def event(kind, **values):
    record = json.loads(json.dumps(dict(kind=kind, wall_time=time.time(), **values)))
    with lock:
        shared['events'].append(record)
        shared['events'] = shared['events'][-50:]
    with open(ROOT/'runtime/events.jsonl','a') as f: f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)

@app.get('/')
def dashboard(): return FileResponse(ROOT/'dashboard.html')
@app.get('/api/status')
def status():
    with lock: return {k:v for k,v in shared.items() if k!='cameras'}
@app.get('/camera/{name}/snapshot.jpg')
def snapshot(name:str):
    with lock: data = shared['cameras'].get(name)
    return Response(data or b'',media_type='image/jpeg',status_code=200 if data else 503)
@app.get('/camera/{name}/stream.mjpg')
def stream(name:str):
    def frames():
        while not commands['stop']:
            with lock: data = shared['cameras'].get(name)
            if data: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n'
            time.sleep(.25)
    return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')
@app.get('/bark.wav')
def bark(): return FileResponse(ROOT/'bark.wav')
@app.post('/api/reset')
def reset(): commands['reset']=True; commands['paused']=False; return {'ok':True}
@app.get('/ctl/sim-running')
def running(): return {'focus':'warehouse','running':[{'file':str(ROOT/'demo.py'),'slot':'warehouse','title':'Warehouse camera search • Go2 / MuJoCo'}]}
@app.get('/ctl/sim-list')
def simlist(): return {'current':str(ROOT/'demo.py'),'sims':[]}
@app.get('/simslot/{slot}/{filename}')
def simfile(slot:str,filename:str):
    if slot!='warehouse' or filename not in ('scene.json','state.json'): return Response(status_code=404)
    p=ROOT/'runtime/warehouse'/filename
    return FileResponse(p) if p.exists() else Response(status_code=503)
@app.post('/ctl/sim-cmd')
async def simcmd(request:__import__('fastapi').Request):
    body=await request.json(); cmd=body.get('cmd')
    if cmd=='reset': commands['reset']=True; commands['paused']=False
    if cmd=='toggle': commands['paused']=not commands['paused']
    if cmd=='step': commands['paused']=True; commands['steps']+=1
    if cmd=='pause': commands['paused']=True
    if cmd=='resume': commands['paused']=False
    return {'ok':True,'control':{'paused':commands['paused']}}
@app.post('/ctl/sim-stop')
def stop(): commands['stop']=True; return {'ok':True}
@app.websocket('/ctl/sim-stream')
async def ws(ws:WebSocket):
    import asyncio
    await ws.accept()
    try:
        while True:
            p=ROOT/'runtime/warehouse/state.json'
            if p.exists(): await ws.send_text(json.dumps({'slot':'warehouse','state':json.loads(p.read_text())}))
            await asyncio.sleep(.04)
    except Exception: pass

def make_scene(target=(3.5,2.7)):
    import xml.etree.ElementTree as ET
    source=ROOT/'menagerie/unitree_go2'
    tree=ET.parse(source/'go2.xml'); root=tree.getroot()
    root.find('compiler').set('meshdir', str(source/'assets'))
    visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth=str(W),offheight=str(H))
    ET.SubElement(visual,'headlight',diffuse='.7 .7 .7',ambient='.4 .4 .4')
    asset=root.find('asset')
    ET.SubElement(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1='.27 .31 .34',rgb2='.32 .36 .39',width='512',height='512')
    ET.SubElement(asset,'material',name='floor_mat',texture='floor_tex',texrepeat='12 10',texuniform='true')
    # Analytic sphere texture, used only by the renderer, never by perception.
    from PIL import Image
    yy,xx=np.mgrid[0:512,0:1024]; theta=xx/1024*2*np.pi; phi=yy/512*np.pi
    tex=np.zeros((512,1024,3),np.uint8); tex[:]=[209,91,24]
    seams=(abs(np.sin(2*theta))<.035)|(abs(phi-np.pi/2)<.025)|(abs(np.cos(theta)*np.sin(phi))<.025)
    tex[seams]=[26,24,22]; Image.fromarray(tex).save(ROOT/'ball.png')
    ET.SubElement(asset,'texture',name='ball_tex',type='2d',file=str(ROOT/'ball.png'))
    ET.SubElement(asset,'material',name='ball_mat',texture='ball_tex',specular='.1',shininess='.15')
    world=root.find('worldbody')
    def geom(name,typ,pos,size,rgba,**kw):
        return ET.SubElement(world,'geom',name=name,type=typ,pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),rgba=rgba,**kw)
    geom('floor','plane',(0,0,0),(6,5,.1),'1 1 1 1',material='floor_mat')
    ET.SubElement(world,'light',pos='0 0 7',dir='0 0 -1',diffuse='.9 .9 .9',castshadow='true')
    for name,pos,size in [('west',(-6,0,1.3),(.08,5,1.3)),('east',(6,0,1.3),(.08,5,1.3)),('north',(0,5,1.3),(6,.08,1.3)),('south',(0,-5,.25),(6,.08,.25))]:
        geom(name,'box',pos,size,'.38 .43 .49 1')
    for i,(x,y,hx,hy) in enumerate(SHELVES):
        # Each rack is a single collision volume in the twin, with visible shelves and crates.
        geom(f'rack_collision_{i}','box',(x,y,1),(hx,hy,1),'0 0 0 0',group='3')
        for px in (x-hx,x+hx):
            for py in (y-hy,y+hy): geom(f'post_{i}_{px}_{py}','box',(px,py,1.05),(.055,.055,1.05),'.15 .30 .43 1')
        for z in (.15,.95,1.8):
            geom(f'shelf_{i}_{z}','box',(x,y,z),(hx,hy,.055),'.22 .34 .43 1')
            for j in range(4): geom(f'crate_{i}_{z}_{j}','box',(x,y-1.45+j*.95,z+.26),(.36,.35,.21),'.57 .37 .19 1')
    geom('search_ball','sphere',(*target,.32),(.32,),'1 1 1 1',material='ball_mat',group='1')
    geom('destination','cylinder',(0,0,-.2),(.48,.012),'.13 .92 .65 .75',contype='0',conaffinity='0')
    for i,pos in enumerate(CORNERS):
        geom(f'camera_housing_{i}','box',pos,(.16,.12,.12),'.08 .12 .18 1',group='1',contype='0',conaffinity='0')
        backward=np.array(pos)-np.array([0,0,.25]); backward/=np.linalg.norm(backward)
        right=np.cross([0,0,1],backward); right/=np.linalg.norm(right)
        up=np.cross(backward,right)
        ET.SubElement(world,'camera',name=f'cam{i+1}',pos=' '.join(map(str,pos)),xyaxes=' '.join(map(str,np.r_[right,up])),fovy='65')
    p=ROOT/'warehouse.xml'; tree.write(p)
    return mujoco.MjModel.from_xml_path(str(p))

def blocked(x,y):
    return abs(x)>5.45 or abs(y)>4.45 or any(abs(x-a)<hx+.43 and abs(y-b)<hy+.43 for a,b,hx,hy in SHELVES)

def plan(start,goal):
    cell=.25
    def key(p): return tuple(int(round(v/cell)) for v in p)
    s,g=key(start),key(goal); frontier=[(0,s)]; cost={s:0}; parent={}
    while frontier:
        _,u=heapq.heappop(frontier)
        if u==g: break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            v=(u[0]+dx,u[1]+dy)
            if blocked(v[0]*cell,v[1]*cell): continue
            n=cost[u]+1
            if n<cost.get(v,1e9):
                cost[v]=n;parent[v]=u; heapq.heappush(frontier,(n+abs(v[0]-g[0])+abs(v[1]-g[1]),v))
    if g not in cost: raise RuntimeError('No collision-free route to detected location')
    path=[g]
    while path[-1]!=s: path.append(parent[path[-1]])
    path.reverse()
    # Keep only corners so the controller does not oscillate at every grid cell.
    corners=[path[0]]
    for i in range(1,len(path)-1):
        if np.any(np.subtract(path[i],path[i-1]) != np.subtract(path[i+1],path[i])): corners.append(path[i])
    corners.append(path[-1]); return [[x*cell,y*cell] for x,y in corners[1:]]

def localize(model,data,name,box):
    cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,name)
    u,v=(box[0]+box[2])/2,box[3]
    f=H/(2*math.tan(math.radians(model.cam_fovy[cid])/2))
    direction=data.cam_xmat[cid].reshape(3,3) @ np.array([(u-W/2)/f,-(v-H/2)/f,-1.])
    direction/=np.linalg.norm(direction); gid=np.array([-1],np.int32)
    distance=mujoco.mj_ray(model,data,data.cam_xpos[cid],direction,np.array([1,0,1,1,1,1],np.uint8),1,-1,gid)
    if distance<0: return None
    hit=data.cam_xpos[cid]+distance*direction
    if abs(hit[2])>.12: return None  # reject a wall/shelf hit rather than inventing a ground target
    return hit

def make_bark():
    """Use a recorded dog bark; playback is triggered by the arrival event."""
    import shutil
    source=ROOT/'audio/arrival-bark.wav'
    if not source.exists(): raise FileNotFoundError(f'Missing recorded bark: {source}')
    shutil.copyfile(source,ROOT/'bark.wav')


def main():
    ROOT.joinpath('runtime/warehouse').mkdir(parents=True,exist_ok=True); make_bark()
    threading.Thread(target=lambda:uvicorn.run(app,host='127.0.0.1',port=18992,log_level='warning'),daemon=True).start()
    model=make_scene(); dog=go2rl.Go2Controller(model=model)
    sys.path.insert(0,str(ROOT.parent/'warehouse-localization'))
    from sim_sensors import LocalizedRobot
    localization_url=os.environ.get('LOCALIZATION_URL','http://127.0.0.1:18790')
    localized=LocalizedRobot(dog,localization_url)
    detector=YOLO(str(ROOT/'yolo11n.pt'))
    renderer=mujoco.Renderer(model,height=H,width=W)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=True
    marker=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'destination')
    route=[]; destination=None; tick=0; scanned=set(); phase='searching'; confidence=None; camera=None; detection_box=None
    patrol=[[0,-3.4],[-4,-3.4],[-4,3.4],[0,3.4]]; patrol_index=0; next_scan=2.
    min_upright=1.; max_contacts=0
    with wendymujoco.launch_passive(model,dog.data,title='Warehouse camera search • Go2 / MuJoCo',rerun=False) as viewer:
        while not commands['stop']:
            began=time.monotonic()
            if commands['reset']:
                dog.reset();route=[];destination=None;scanned=set();phase='searching';tick=0;patrol_index=0;next_scan=2.;camera=None;confidence=None;detection_box=None;commands['reset']=False
                localized=LocalizedRobot(dog,localization_url)
                min_upright=1.; model.geom_pos[marker]=[0,0,-.2];event('reset')
            if commands['paused'] and commands['steps']==0: time.sleep(.04); continue
            if commands['steps']>0: commands['steps']-=1
            try:
                localized.update()
            except Exception as exc:
                localized.pose=None
                with lock:shared['localization_fault']=str(exc)
            if not localized.ready:
                dog.set_command(0,0,0);dog.step();tick+=1
                viewer.hud(phase='Waiting for LiDAR localization',physics='MuJoCo + Nav2 AMCL')
                viewer.sync()
                with lock:shared.update(phase='localization_waiting',localized_position=None,pose_source='Nav2 AMCL')
                time.sleep(max(0,.02-(time.monotonic()-began)))
                continue
            x,y=localized.base_xy; simtime=tick*.02
            if phase=='searching' and simtime>=next_scan:
                idx=len(scanned)%4; name=f'cam{idx+1}'
                renderer.update_scene(dog.data,camera=name);rgb=renderer.render().copy(); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                results=detector.predict(bgr,classes=[32],conf=.18,imgsz=960,verbose=False,device='cpu')[0]
                annotated=results.plot();cv2.imwrite(str(ROOT/f'runtime/{name}.jpg'),annotated)
                with lock: shared['cameras'][name]=cv2.imencode('.jpg',annotated)[1].tobytes()
                scanned.add(name);next_scan=simtime+2.
                event('camera_scanned',camera=name,detections=len(results.boxes),simulation_time=simtime)
                for box in sorted(results.boxes,key=lambda b:float(b.conf[0]),reverse=True):
                    hit=localize(model,dog.data,name,box.xyxy[0].cpu().numpy())
                    if hit is None: continue
                    # Stop short of the detected surface; no target pose lookup.
                    delta=np.array([x,y])-hit[:2];delta/=max(np.linalg.norm(delta),1e-6)
                    goal=hit[:2]+delta*.75
                    if blocked(*goal): continue
                    route=plan((x,y),goal);destination=hit[:2].tolist();camera=name;confidence=float(box.conf[0]);detection_box=box.xyxy[0].cpu().numpy().astype(int).tolist();phase='navigating'
                    model.geom_pos[marker]=[*goal,.015]
                    event('localized',camera=name,confidence=confidence,pixel_box=box.xyxy[0].cpu().numpy().tolist(),twin_hit=hit.tolist(),route=route)
                    break
                if len(scanned)==4 and phase=='searching': scanned=set()
            # Refresh every IP feed regardless of which feed is currently inspected by YOLO.
            if tick%100==0:
                for name in ['cam1','cam2','cam3','cam4']:
                    renderer.update_scene(dog.data,camera=name);bgr=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                    if name==camera and detection_box:
                        a,b,c,d=detection_box;cv2.rectangle(bgr,(a,b),(c,d),(100,240,100),2);cv2.putText(bgr,f'sports ball {confidence:.2f}',(a,max(20,b-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(100,240,100),2)
                    with lock: shared['cameras'][name]=cv2.imencode('.jpg',bgr)[1].tobytes()
            if phase=='navigating':
                if route and math.dist((x,y),route[0])<.26:route.pop(0)
                if not route:
                    phase='found';dog.set_command(0,0,0)
                    with lock: shared['bark']+=1
                    event('arrived_and_barked',position=[x,y],destination=destination,distance=math.dist((x,y),destination),min_upright=min_upright,contacts=max_contacts)
                goal=route[0] if route else None
            elif phase=='searching':
                if math.dist((x,y),patrol[patrol_index])<.35:patrol_index=(patrol_index+1)%len(patrol)
                goal=patrol[patrol_index]
            else: goal=None
            if goal:
                dx,dy=goal[0]-x,goal[1]-y;error=(math.atan2(dy,dx)-localized.base_yaw+math.pi)%(2*math.pi)-math.pi
                dog.set_command(.55*max(0,math.cos(error))*min(1,math.hypot(dx,dy)/.6),0,np.clip(1.8*error,-1,1))
            else: dog.set_command(0,0,0)
            dog.step();tick+=1;min_upright=min(min_upright,dog.upright);max_contacts=max(max_contacts,int(dog.data.ncon))
            if dog.base_height<.18 or dog.upright<.55:
                phase='fault';dog.set_command(0,0,0);event('robot_fell',height=dog.base_height,upright=dog.upright);commands['paused']=True
            if simtime>180 and phase not in ('found','fault'):
                phase='fault';dog.set_command(0,0,0);event('timeout');commands['paused']=True
            viewer.hud(phase=phase,camera=camera or f'{len(scanned)}/4 searched',YOLO=round(confidence or 0,3),x=round(x,2),y=round(y,2),height=round(dog.base_height,2),bark='WOOF! WOOF!' if phase=='found' else 'searching',physics='MuJoCo + RL joint torques')
            viewer.sync()
            with lock: shared.update(phase=phase,position=[x,y],localized_position=[x,y],pose_source='Nav2 AMCL',route=route,destination=destination,camera=camera,confidence=confidence,simulation_time=simtime,upright=dog.upright,contacts=int(dog.data.ncon))
            time.sleep(max(0,.02-(time.monotonic()-began)))
    renderer.close()

if __name__=='__main__': main()
