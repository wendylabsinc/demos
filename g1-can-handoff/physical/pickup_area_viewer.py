"""Read-only local pickup-placement guide; independent of operator controls.

Run with the project's Python environment. No camera lease, DDS, start, stop,
release, deployment or controller restart. The guide hides during a demo or
when its pose/calibration snapshot expires. Projection uses the accepted
stationary-base model, not an independent measurement of the workspace.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from operator_panel import Wendy
from pose_alignment import corrected_config

PORT = 8198

SCRIPT = r'''import ast,io,json,pathlib,time,urllib.request,zipfile
s=json.load(urllib.request.urlopen('http://127.0.0.1:8102/status',timeout=3))
if s['demo_running']:
 print(json.dumps(dict(demo_running=True)));raise SystemExit(0)
c=json.loads(pathlib.Path('/data/calibration.json').read_text())
h=json.load(urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=3))
raw=urllib.request.urlopen('http://127.0.0.1:8001/frame.recorder-rgbd?frame_id='+str(h['frame_id']),timeout=3).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:m=json.loads(z.read('metadata.json'))
bounds={}
for node in ast.walk(ast.parse(pathlib.Path('/app/g1_can_handoff/task.py').read_text())):
 if not isinstance(node,ast.Compare) or len(node.ops)!=2 or not all(isinstance(op,ast.LtE) for op in node.ops):continue
 a,b=node.comparators
 if (isinstance(node.left,ast.Constant) and isinstance(b,ast.Constant) and isinstance(a,ast.Subscript)
     and isinstance(a.value,ast.Name) and a.value.id=='c' and isinstance(a.slice,ast.Constant)
     and a.slice.value in (0,1)):bounds[str(a.slice.value)]=[node.left.value,b.value]
body=s.get('measured',{}).get('body')
print(json.dumps(dict(config=c,bounds=bounds,intrinsics=m['calibration']['intrinsics'],
 camera_serial=m['calibration']['serial'],phase=s['phase'],demo_running=s['demo_running'],body=body,
 body_age_s=None if body is None else time.monotonic()-body['time'],camera_health=h)))
'''


def project_area(snapshot):
    if snapshot['demo_running']:
        raise ValueError('Placement guide hidden while the demo is running')
    if not 0 <= snapshot['body_age_s'] <= .2:
        raise ValueError('Placement guide unavailable: body feedback is stale')
    config = snapshot['config']
    if not config.get('operator_accepted') or not config.get('stationary_base_confirmed'):
        raise ValueError('Placement guide requires accepted stationary-base calibration')
    if config['camera_serial'] != snapshot['camera_serial']:
        raise ValueError('Placement guide camera/calibration mismatch')
    h = snapshot['camera_health']
    if not h['healthy'] or not 0 <= h['frame_age_s'] <= .4:
        raise ValueError('Placement guide unavailable: camera is stale')
    body = snapshot['body']
    config = corrected_config(config, dict(timestamp=body['time'], rpy=body['imu_rpy'], waist_q=body['q'][12:15]))
    intrinsics = snapshot['intrinsics']
    # The verified NX2 aligned-color intrinsics have zero distortion. Fail
    # visibly if this camera contract changes rather than silently misproject.
    if any(float(c) != 0 for c in intrinsics['coeffs']):
        raise ValueError('Placement guide needs updated distortion projection')
    bounds = snapshot['bounds']
    x0, x1 = bounds['0']; y0, y1 = bounds['1']
    if not all(np.isfinite([x0, x1, y0, y1])) or not x0 < x1 or not y0 < y1:
        raise ValueError('Invalid pickup workspace')
    world = np.array([[x0,y0,config['table_height_m']], [x1,y0,config['table_height_m']],
                      [x1,y1,config['table_height_m']], [x0,y1,config['table_height_m']]])
    transform = np.asarray(config['camera_to_world'])
    camera = (world - transform[:3,3]) @ transform[:3,:3]
    if not np.isfinite(camera).all() or (camera[:,2] <= 0).any():
        raise ValueError('Pickup area is outside the camera view')
    pixels = np.column_stack((intrinsics['fx']*camera[:,0]/camera[:,2]+intrinsics['ppx'],
                              intrinsics['fy']*camera[:,1]/camera[:,2]+intrinsics['ppy']))
    return dict(polygon=pixels.tolist(),width=intrinsics['width'],height=intrinsics['height'],
                bounds_m=bounds,phase=snapshot['phase'],table_height_m=config['table_height_m'],
                label=f'Can base center: {x0*100:.0f}–{x1*100:.0f} cm forward · {y0*100:.0f}–{y1*100:.0f} cm robot-left')


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>G1 pickup placement</title>
<style>body{background:#10151c;color:#edf2f8;font:17px system-ui;margin:0}main{max-width:960px;margin:24px auto;padding:20px}h1{font-size:25px}.view{position:relative;max-width:800px}img{display:block;width:100%;border-radius:12px}svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}polygon{fill:#62ffab33;stroke:#62ffab;stroke-width:3;vector-effect:non-scaling-stroke}#message{color:#9df4c6}a{color:#a8cafa}</style>
</head><body><main><h1>G1 · Pickup placement guide</h1><p>Place the <b>center of the can’s base</b> inside the green area on the table.</p>
<div class="view"><img id="video" alt="Live camera"><svg id="overlay" preserveAspectRatio="none"><polygon id="area"/></svg></div>
<p id="message">Loading current calibration…</p><p id="video-state"></p><p>Stationary placement guide from the accepted camera/table calibration. Hidden during a demo or when calibration feedback is unavailable.</p><p><a href="http://127.0.0.1:8197/">Open demo controls</a></p>
<script>
const video=document.querySelector('#video'),svg=document.querySelector('#overlay'),area=document.querySelector('#area'),message=document.querySelector('#message'),vs=document.querySelector('#video-state');let pending=false,started=0,last=0,guideReceived=0,valid=false;
function visible(){svg.style.display=valid&&Date.now()-guideReceived<4500&&Date.now()-last<1000?'block':'none';}
video.onload=()=>{pending=false;last=Date.now();vs.textContent='Camera image received '+new Date(last).toLocaleTimeString();visible();};
video.onerror=()=>{pending=false;valid=false;visible();vs.textContent='Camera unavailable';};
function image(){if(pending&&Date.now()-started>3000)pending=false;if(!pending){pending=true;started=Date.now();video.src='http://10.10.20.238:8001/frame.jpg?t='+started;}visible();if(last&&Date.now()-last>1000)vs.textContent='Camera feed stalled · guide hidden';}
let polling=false;async function guide(){if(polling)return;polling=true;try{let r=await fetch('/overlay',{cache:'no-store'}),s=await r.json();guideReceived=Date.now();valid=r.ok&&s.valid;if(valid){svg.setAttribute('viewBox',`0 0 ${s.width} ${s.height}`);area.setAttribute('points',s.polygon.map(p=>p.join(',')).join(' '));message.textContent=s.label;}else message.textContent=s.error||'Placement guide unavailable';}catch(e){valid=false;message.textContent='Placement guide unavailable';}finally{polling=false;visible();}}
image();guide();setInterval(image,400);setInterval(guide,1000);
</script></main></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Host') != f'127.0.0.1:{PORT}':
            self.send_error(403); return
        if self.path == '/':
            body=PAGE.encode(); kind='text/html; charset=utf-8'; code=200
        elif self.path == '/overlay':
            with self.server.lock:
                result=dict(self.server.result)
                if time.monotonic()-self.server.updated > 4.5:
                    result=dict(valid=False,error='Placement feedback expired · guide hidden')
            body=json.dumps(result,allow_nan=False).encode();kind='application/json';code=200
        else:
            self.send_error(404); return
        self.send_response(code);self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def log_message(self,*args):pass


def main():
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    server.lock=threading.Lock();server.result=dict(valid=False,error='Loading calibration…');server.updated=0
    stop=threading.Event()
    def update():
        client=None
        try:
            client=Wendy()
            while not stop.is_set():
                try:
                    with client.lock:
                        result=client.tool('container_exec',dict(app_name='DEV-g1-handoff-physical',
                            command=['python3','-c',SCRIPT],timeout_seconds=12,max_bytes=18000))
                    data=result.get('structuredContent')
                    if data is None:data=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
                    if data.get('exit_code') != 0 or data.get('truncated'):
                        raise RuntimeError(data.get('stderr') or 'Calibration read unavailable')
                    guide=project_area(json.loads(data['stdout']));guide['valid']=True
                except Exception as exc:guide=dict(valid=False,error=str(exc))
                with server.lock:server.result=guide;server.updated=time.monotonic()
                stop.wait(2)
        except Exception as exc:
            with server.lock:server.result=dict(valid=False,error=str(exc));server.updated=time.monotonic()
        finally:
            if client is not None:client.process.terminate()
    worker=threading.Thread(target=update,daemon=True);worker.start()
    print(f'Pickup overlay: http://127.0.0.1:{PORT}/',flush=True)
    try:server.serve_forever()
    finally:stop.set();server.server_close();worker.join(3)


if __name__=='__main__':main()
