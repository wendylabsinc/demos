"""MuJoCo worker, camera server, and Wendy native simulation bridge."""
from __future__ import annotations
import argparse
import asyncio
import io
import json
import multiprocessing as mp
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
from world import City, CAMERAS

ROOT=Path(__file__).resolve().parent
app=FastAPI(); lock=threading.RLock(); commands=queue.Queue(); stopped=threading.Event()
shared={'phase':'loading'}; images={}; image_versions={}; image_capture={}; scene={}; poses={}; paused=False; native_packet=''; generation=0

@app.get('/')
def index(): return FileResponse(ROOT/'dashboard.html')
@app.get('/api/status')
def status():
    with lock: return shared
@app.get('/camera/{name}.jpg')
def camera(name:str):
    with lock: frame=images.get(name)
    if frame is None: raise HTTPException(503,'Camera warming up')
    return Response(frame,media_type='image/jpeg',headers={'Cache-Control':'no-store'})
@app.post('/api/control')
async def control(req:Request):
    cmd=(await req.json()).get('cmd')
    if cmd not in ('pause','resume','reset','brake','release','step','toggle'): raise HTTPException(400,'Unknown command')
    commands.put(cmd); return {'ok':True}
@app.post('/ctl/sim-cmd')
async def native_control(req:Request):
    cmd=(await req.json()).get('cmd')
    if cmd not in ('pause','resume','reset','step','toggle'): raise HTTPException(400,'Unknown command')
    commands.put(cmd); return {'ok':True,'control':{'paused':cmd in ('pause','step') or (cmd=='toggle' and not paused)}}
@app.post('/ctl/sim-stop')
def stop(): stopped.set(); return {'ok':True}
@app.get('/ctl/sim-running')
def running(): return {'focus':'autonomous-taxi','running':[{'file':str(ROOT/'demo.py'),'slot':'autonomous-taxi','title':'Autonomous taxi · MuJoCo city'}] if not stopped.is_set() else []}
@app.get('/ctl/sim-list')
def sim_list(): return {'current':str(ROOT/'demo.py'),'sims':[]}
@app.get('/simslot/{slot}/{filename}')
def sim_file(slot:str,filename:str):
    if slot!='autonomous-taxi': raise HTTPException(404)
    with lock:
        result=scene if filename=='scene.json' else poses if filename=='state.json' else None
    if not result: raise HTTPException(503)
    return result
@app.websocket('/ctl/sim-stream')
async def stream(ws:WebSocket):
    await ws.accept()
    try:
        while not stopped.is_set():
            with lock: result=native_packet
            if result: await ws.send_text(result)
            await asyncio.sleep(1/30)
    except Exception: pass

@app.websocket('/api/camera-stream')
async def camera_stream(ws:WebSocket):
    await ws.accept(); seen={}; names=['chase',*CAMERAS]
    try:
        while not stopped.is_set():
            with lock:
                updates=[(i,name,image_versions[name],images[name]) for i,name in enumerate(names)
                         if name in images and image_versions[name]!=seen.get(name)]
            for i,name,version,content in updates:
                await ws.send_bytes(bytes([i])+content); seen[name]=version
            await asyncio.sleep(.01)
    except Exception: pass


def native_scene(city):
    m=city.model; types={0:'plane',2:'sphere',3:'capsule',4:'ellipsoid',5:'cylinder',6:'box'}
    geoms=[]
    for i in range(m.ngeom):
        rgba=m.geom_rgba[i].tolist(); name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
        if name.startswith('lamp_'): rgba=[[.95,.08,.06,1],[1,.7,.05,1],[.1,.95,.3,1]][int(name[-1])]
        geoms.append(dict(i=i,type=types[int(m.geom_type[i])],size=m.geom_size[i].tolist(),rgba=rgba))
    return dict(title='Autonomous taxi city',engine='mujoco',geoms=geoms,meshes={})

def native_state(city,sequence,state):
    quat=np.zeros(4)
    if not hasattr(city,'native_static'):
        city.native_static=[]
        for p,r in zip(city.data.geom_xpos,city.data.geom_xmat):
            mujoco.mju_mat2Quat(quat,r)
            city.native_static.append([*p.round(5).tolist(),*quat.round(5).tolist()])
        city.native_dynamic=np.flatnonzero(city.model.body_dofnum[city.model.geom_bodyid]>0)
    pose=city.native_static.copy()
    for i in city.native_dynamic:
        mujoco.mju_mat2Quat(quat,city.data.geom_xmat[i])
        pose[i]=[*city.data.geom_xpos[i].round(5).tolist(),*quat.round(5).tolist()]
    for i,_,_ in city.lamps:
        if sum(city.model.geom_rgba[i,:3])<.4:
            pose[i]=pose[i].copy(); pose[i][2]=-20
    return dict(frame=sequence,engine='mujoco',title='Autonomous taxi',pose=pose,contacts=[],hud={'Destination':state['destination'],'Driving':state['reason'],'Speed km/h':state['speed'],'Trips':state['trips'],'LiDAR rays':180,'Cameras':6,'Collisions':state['collisions']})

def encode(pixels):
    out=io.BytesIO(); Image.fromarray(pixels).save(out,format='JPEG',quality=83); return out.getvalue()

def render_worker(xml,snapshots,output,quit_event):
    """Own a rendering-only model in a separate process; never block physics."""
    renderer=sensor=None
    try:
        model=mujoco.MjModel.from_xml_string(xml); data=mujoco.MjData(model)
        renderer=mujoco.Renderer(model,height=400,width=640)
        sensor=mujoco.Renderer(model,height=180,width=300)
        # Keep scene shadows in the chase view; onboard previews prioritize latency.
        sensor.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
        camera=mujoco.MjvCamera(); camera.distance=17; camera.elevation=-28
        latest=None; next_chase=0.; next_sensor=0.; index=0
        while not quit_event.is_set():
            changed=False
            while True:
                try: latest=snapshots.get_nowait(); changed=True
                except queue.Empty: break
            if latest is None: quit_event.wait(.005); continue
            gen,stamp,sim_time,qpos,lamp_ids,rgba=latest
            if changed:
                data.qpos[:]=qpos; data.time=sim_time; model.geom_rgba[lamp_ids]=rgba
                mujoco.mj_forward(model,data)
            now=time.monotonic(); frames=[]
            if now>=next_chase:
                camera.lookat[:]=[*data.qpos[:2],1.]
                camera.azimuth=math_degrees(data.qpos[2])+180
                renderer.update_scene(data,camera=camera)
                frames.append(('chase',encode(renderer.render())))
                next_chase=max(next_chase+.05,now+.002)
            if now>=next_sensor:
                name=CAMERAS[index%len(CAMERAS)]; index+=1
                sensor.update_scene(data,camera=name)
                sensor.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                frames.append((name,encode(sensor.render())))
                next_sensor=max(next_sensor+1/30,now+.002)
            for name,content in frames:
                try: output.put_nowait((gen,name,stamp,content))
                except queue.Full: pass  # Dropping an image is preferable to a growing delay.
            quit_event.wait(.002)
    except Exception as exc:
        try: output.put_nowait((-1,'error',time.monotonic(),str(exc)))
        except queue.Full: pass
    finally:
        if sensor: sensor.close()
        if renderer: renderer.close()


def worker():
    global shared,scene,poses,paused,native_packet,generation
    process=None; render_quit=None
    try:
        city=City(); (ROOT/'runtime').mkdir(exist_ok=True); (ROOT/'runtime/city.xml').write_text(city.xml)
        context=mp.get_context('spawn'); snapshots=context.Queue(maxsize=1); rendered=context.Queue(maxsize=16); render_quit=context.Event()
        process=context.Process(target=render_worker,args=(city.xml,snapshots,rendered,render_quit),daemon=True); process.start()
        with lock: scene=native_scene(city)
        seq=0; deadline=time.monotonic(); next_publish=deadline; metric_wall=deadline; metric_sim=0.; rtf=1.; camera_counts={}; camera_fps={}; render_error=None
        while not stopped.is_set():
            one_step=False
            while not commands.empty():
                cmd=commands.get()
                if cmd=='pause': paused=True
                elif cmd=='resume': paused=False
                elif cmd=='toggle': paused=not paused
                elif cmd=='step': paused=True; one_step=True
                elif cmd=='brake': city.emergency=True
                elif cmd=='release': city.emergency=False; paused=False
                elif cmd=='reset':
                    city=City(); paused=False; generation+=1; metric_wall=time.monotonic(); metric_sim=0.
                    with lock:
                        scene=native_scene(city); images.clear(); image_versions.clear(); image_capture.clear()
            if not paused or one_step: city.step()
            now=time.monotonic()
            while True:
                try: gen,name,stamp,content=rendered.get_nowait()
                except queue.Empty: break
                if name=='error': render_error=content; continue
                if gen!=generation: continue
                with lock:
                    images[name]=content; image_versions[name]=image_versions.get(name,0)+1; image_capture[name]=stamp
                camera_counts[name]=camera_counts.get(name,0)+1
            if now-metric_wall>=2:
                duration=now-metric_wall; rtf=(city.data.time-metric_sim)/duration
                camera_fps={k:round(v/duration,1) for k,v in camera_counts.items()}
                camera_counts={}; metric_wall=now; metric_sim=city.data.time
            if now>=next_publish:
                seq+=1; state=city.state()
                state.update(paused=paused,phase='running',performance=dict(realtime_factor=round(rtf,2),camera_fps=camera_fps,camera_age_ms={k:round((now-v)*1000) for k,v in image_capture.items()},render_error=render_error))
                native=native_state(city,seq,state)
                packet=json.dumps({'slot':'autonomous-taxi','state':native},separators=(',',':'))
                with lock: shared=state; poses=native; native_packet=packet
                lamp_ids=np.array([i for i,_,_ in city.lamps])
                snapshot=(generation,now,float(city.data.time),city.data.qpos.copy(),lamp_ids,city.model.geom_rgba[lamp_ids].copy())
                try: snapshots.put_nowait(snapshot)
                except queue.Full: pass
                next_publish=max(next_publish+1/30,now)
            deadline+=.01
            if now-deadline>.1: deadline=now  # Bound catch-up after an OS scheduling stall.
            stopped.wait(max(0,deadline-time.monotonic()))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        with lock: shared={'phase':'error','error':str(exc)}
    finally:
        if render_quit: render_quit.set()
        if process:
            process.join(timeout=3)
            if process.is_alive(): process.terminate(); process.join(timeout=1)

def math_degrees(r): return float(np.degrees(r))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8788); args=parser.parse_args()
    thread=threading.Thread(target=worker,daemon=True); thread.start()
    try: uvicorn.run(app,host='127.0.0.1',port=args.port,log_level='warning')
    finally: stopped.set(); thread.join(timeout=5)
