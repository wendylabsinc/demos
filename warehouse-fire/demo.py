"""Live thermal warehouse simulation and Wendy native viewer HTTP bridge."""
import argparse
import asyncio
import io
import json
import os
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, Response
import uvicorn
from twin import ThermalTwin, RACKS, WIDTH, HEIGHT, zone_at, zone_center
from controller import FireController
from hardware import HardwareAdapter
from push import PushDelivery

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('FIRE_DATA_DIR',ROOT/'runtime')); DATA.mkdir(parents=True,exist_ok=True)
app=FastAPI(); lock=threading.RLock(); jobs=queue.Queue()
shared={'phase':'loading','events':[],'cameras':[],'incidents':[],'active_zones':[],'frame':0,'mode':'simulation','paused':False}
images={}; scene={}; simstate={}; stop=threading.Event()

def record(kind,**values):
    event=dict(kind=kind,wall_time=time.time(),**values)
    with lock:
        shared['events'].append(event); shared['events']=shared['events'][-80:]
        with (DATA/'events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)

push=PushDelivery(DATA,record)

def emit(kind,**values):
    record(kind,**values)
    if kind=='fire_detected':
        x,y,z=values['position']
        push.send(dict(title='DEMO: Warehouse fire detected',body=f'Fire at ({x:.1f}, {y:.1f}) m, sprinkler zone {values["zone"]+1}.',incident_id=values['id'],position=values['position']))

@app.get('/')
def dashboard(): return FileResponse(ROOT/'dashboard.html')
@app.get('/sw.js')
def worker(): return FileResponse(ROOT/'sw.js',media_type='application/javascript',headers={'Service-Worker-Allowed':'/'})
@app.get('/api/status')
def status():
    with lock: return json.loads(json.dumps(shared))
@app.get('/api/push/key')
def pushkey(): return {'public_key':push.public_key,'subscribers':len(push.subscriptions)}
@app.post('/api/push/subscribe')
async def subscribe(request:Request):
    try: count=push.subscribe(await request.json())
    except (ValueError,KeyError) as exc: raise HTTPException(400,str(exc)) from exc
    return {'ok':True,'subscribers':count}
@app.post('/api/push/test')
def testpush():
    push.send(dict(title='Wendy fire demo connected',body='Push notifications are ready. This is a simulation.'))
    return {'ok':True}
@app.get('/camera/{name}.png')
def camera(name:str):
    with lock: content=images.get(name)
    if not content: raise HTTPException(503,'Thermal camera initializing')
    return Response(content,media_type='image/png',headers={'Cache-Control':'no-store'})
@app.post('/api/ignite')
async def ignite(request:Request):
    body=await request.json()
    if shared['mode']!='simulation': raise HTTPException(409,'Ignition is simulator-only')
    try: x,y=float(body['x']),float(body['y'])
    except (KeyError,TypeError,ValueError) as exc: raise HTTPException(400,'Expected x and y in meters') from exc
    if not np.isfinite([x,y]).all() or not .5<=x<=39.5 or not .5<=y<=39.5: raise HTTPException(400,'Fire must be within the warehouse')
    jobs.put(('ignite',(x,y))); return {'ok':True}
@app.post('/api/reset')
def reset():
    if shared['mode']!='simulation': raise HTTPException(409,'Reset is simulator-only')
    jobs.put(('reset',None)); return {'ok':True}
@app.post('/ctl/sim-cmd')
async def simcmd(request:Request):
    body=await request.json(); cmd=body.get('cmd')
    if shared['mode']!='simulation': raise HTTPException(409,'Simulation controls disabled in hardware mode')
    jobs.put((cmd,None))
    return {'ok':True,'control':{'paused':cmd in ('pause','step') or (cmd=='toggle' and not shared['paused'])}}
@app.post('/ctl/sim-stop')
def stop_sim(): stop.set(); return {'ok':True}
@app.get('/ctl/sim-running')
def running(): return {'focus':'warehouse-fire','running':[{'file':str(ROOT/'demo.py'),'slot':'warehouse-fire','title':'Warehouse fire watch · 40 × 40 m · MuJoCo'}]}
@app.get('/ctl/sim-list')
def simlist(): return {'current':str(ROOT/'demo.py'),'sims':[]}
@app.get('/simslot/{slot}/{filename}')
def simfile(slot:str,filename:str):
    if slot!='warehouse-fire': raise HTTPException(404)
    with lock:
        value=scene if filename=='scene.json' else simstate if filename=='state.json' else None
        if not value: raise HTTPException(503)
        return value
@app.websocket('/ctl/sim-stream')
async def simstream(socket:WebSocket):
    await socket.accept()
    try:
        while not stop.is_set():
            with lock: packet=json.dumps({'slot':'warehouse-fire','state':simstate})
            if simstate: await socket.send_text(packet)
            await asyncio.sleep(.05)
    except Exception: pass

def native_scene(model):
    kinds={mujoco.mjtGeom.mjGEOM_PLANE:'plane',mujoco.mjtGeom.mjGEOM_BOX:'box',mujoco.mjtGeom.mjGEOM_SPHERE:'sphere',mujoco.mjtGeom.mjGEOM_ELLIPSOID:'ellipsoid',mujoco.mjtGeom.mjGEOM_CYLINDER:'cylinder'}
    return dict(title='Thermal warehouse · 40 × 40 meters',up='z',engine='mujoco',meshes={},geoms=[dict(i=i,type=kinds[model.geom_type[i]],size=model.geom_size[i].tolist(),rgba=model.geom_rgba[i].tolist()) for i in range(model.ngeom)])

def native_state(twin,frame,hud):
    pose=[]; quat=np.zeros(4)
    for i,(pos,mat) in enumerate(zip(twin.data.geom_xpos,twin.data.geom_xmat)):
        pos=pos.copy()
        if pos[2]<-4: pos=np.array([20.,20.,-4.])
        mujoco.mju_mat2Quat(quat,mat)
        # The native orbit viewer uses a centered coordinate system.
        pose.append([float(pos[0]-20),float(pos[1]-20),float(pos[2]),*quat.tolist()])
    return dict(t=time.time(),frame=frame,engine='mujoco',pose=pose,hud=hud,contacts=[])

def thermal_png(frame):
    value=np.clip((frame-20)/180,0,1)
    colors=np.stack([np.minimum(1,value*3),np.clip(value*3-1,0,1),np.clip(value*3-2,0,1)],axis=-1)
    # Blue cool background and orange/white hot surfaces, with exact 32 × 24 pixels.
    colors[:,:,2]=np.where(value<.25,.18+.5*value,colors[:,:,2])
    image=Image.fromarray((colors*255).astype('uint8')); stream=io.BytesIO(); image.save(stream,format='PNG'); return stream.getvalue()

def simulation(args):
    global scene,simstate
    twin=ThermalTwin(DATA/'warehouse.xml'); sources=[]; active=set(); frame=0; elapsed=0.; paused=False; steps=0; fired=set()
    hardware=HardwareAdapter(json.loads(Path(args.hardware).read_text())) if args.hardware else None
    def actuate(zone,on):
        if hardware: hardware.actuate(zone,on)
        if on: active.add(zone)
        else: active.discard(zone)
    controller=FireController(twin.hits,twin.cameras,emit,actuate)
    def add_fire(x,y):
        if len(sources)>=8:
            emit('ignition_rejected',reason='Reset to clear the eight-source scenario'); return
        point=twin.surface(x,y)
        sources.append(dict(position=point,temperature=180.,burning=True,created_at=elapsed))
        emit('ignition',position=point,simulation_time=elapsed)
    with lock:
        scene=native_scene(twin.model)
        shared.update(phase='watching',cameras=twin.cameras,racks=RACKS,warehouse_m=[40,40],resolution=[WIDTH,HEIGHT],mode='hardware' if hardware else 'simulation')
    # Export the simulated calibration contract; hardware installations must match their actual poses.
    np.savez_compressed(DATA/'calibration.npz',surface_hits=twin.hits)
    while not stop.is_set():
        start=time.monotonic()
        while not jobs.empty():
            cmd,value=jobs.get()
            if cmd=='ignite': add_fire(*value)
            elif cmd=='reset':
                sources.clear(); active.clear(); controller=FireController(twin.hits,twin.cameras,emit,actuate); elapsed=0.; fired.clear(); paused=False
                mujoco.mj_resetData(twin.model,twin.data); mujoco.mj_forward(twin.model,twin.data); emit('reset')
            elif cmd=='toggle': paused=not paused
            elif cmd=='pause': paused=True
            elif cmd=='resume': paused=False
            elif cmd=='step': paused=True; steps+=1
        if paused and not steps:
            with lock: shared['paused']=True
            stop.wait(.1); continue
        if steps: steps-=1
        elapsed+=.1; frame+=1
        if args.auto and not hardware:
            for n,(at,x,y) in enumerate([(4,7.5,7.5),(14,30.4,23.5),(25,37.5,32.5)]):
                if elapsed>=at and n not in fired: fired.add(n); add_fire(x,y)
        if hardware:
            try:
                frames=hardware.frames(twin.valid.shape)
                controller.update(frames,time.monotonic())
                with lock: shared['sensor_fault']=None
            except Exception as exc:
                with lock:
                    if shared.get('sensor_fault')!=str(exc): emit('sensor_fault',error=str(exc))
                    shared['sensor_fault']=str(exc)
                stop.wait(.1); continue
        else:
            for source in sources:
                if source['burning']:
                    water=any(np.linalg.norm(np.array(source['position'])[:2]-zone_center(zone)[:2])<=3.6 for zone in active)
                    source['temperature']=max(22.,min(600.,source['temperature']+(.1*(30-90*water))))
                    if water and source['temperature']<=35:
                        source['burning']=False; emit('extinguished',position=source['position'],simulation_time=elapsed)
                else: source['temperature']=max(22.,source['temperature']-3)
            frames=twin.thermal_frames(sources)
            controller.update(frames,elapsed)
        twin.effects(sources,controller.incidents,active)
        maxes=frames.max(axis=(1,2)); top=np.argsort(maxes)[-6:][::-1]
        hud=dict(phase='FIRE · LOCAL SPRINKLERS ON' if active else 'Warehouse protected',cameras='88 thermal · 32 × 24',sprinklers=f'{len(active)} / 64 active',detected=str(len(controller.incidents)),time=f'{elapsed:.1f} s',engine='MuJoCo + radiometric thermal model')
        state=native_state(twin,frame,hud)
        with lock:
            simstate=state
            shared.update(phase='suppressing' if active else 'watching',frame=frame,simulation_time=round(elapsed,1),paused=paused,incidents=json.loads(json.dumps(controller.incidents)),active_zones=sorted(active),peak_c=round(float(maxes.max()),1),camera_temperatures=maxes.round(1).tolist(),featured_cameras=[twin.cameras[i]['id'] for i in top],push_subscribers=len(push.subscriptions))
            for n,cam in enumerate(twin.cameras): images[cam['id']]=thermal_png(frames[n])
        # Evidence exposes ground truth separately, never to the detector.
        if frame%10==0: (DATA/'truth.json').write_text(json.dumps(dict(sources=sources,simulation_time=elapsed)))
        if args.seconds and elapsed>=args.seconds: stop.set()
        stop.wait(max(0,.1-(time.monotonic()-start)))
    (DATA/'final-status.json').write_text(json.dumps(status(),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--auto',action='store_true'); parser.add_argument('--seconds',type=float); parser.add_argument('--hardware',help='Gateway configuration JSON for real thermal cameras and valves')
    args=parser.parse_args()
    worker=threading.Thread(target=simulation,args=(args,),daemon=True); worker.start()
    uvicorn.run(app,host=os.environ.get('FIRE_HOST','127.0.0.1'),port=int(os.environ.get('FIRE_PORT','8788')),log_level='warning')
