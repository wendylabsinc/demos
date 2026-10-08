"""Read-only calibrated placement overlay shared by the local controls page."""
import json
import threading
import time
import numpy as np
from pose_alignment import corrected_config


class PlacementGuide:
    """Separate read-only transport keeps projection off Stop/Release's path."""
    def __init__(self, client_factory):
        self.client_factory=client_factory;self.lock=threading.Lock()
        self.value=dict(valid=False,error='Loading calibrated pickup area…')
        self.updated=0.;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.update,daemon=True,name='placement-guide')

    def start(self):
        self.thread.start();return self

    def read(self):
        with self.lock:
            if time.monotonic()-self.updated>3:
                return dict(valid=False,error='Placement feedback expired — guide hidden')
            return dict(self.value)

    def update(self):
        client=None
        try:
            client=self.client_factory()
            while not self.stop.is_set():
                try:
                    result=client.tool('container_exec',dict(app_name='DEV-g1-handoff-physical',
                        command=['python3','-c',SCRIPT],timeout_seconds=10,max_bytes=18000))
                    payload=result.get('structuredContent') or json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
                    if payload.get('exit_code')!=0 or payload.get('truncated'):
                        raise RuntimeError(payload.get('stderr') or 'Placement feedback unavailable')
                    value=project_area(json.loads(payload['stdout']));value['valid']=True
                except Exception as exc:value=dict(valid=False,error=str(exc))
                with self.lock:self.value=value;self.updated=time.monotonic()
                self.stop.wait(1)
        finally:
            if client is not None:client.process.terminate()

    def close(self):
        self.stop.set();self.thread.join(1)

SCRIPT = r'''import ast,io,json,pathlib,time,urllib.request,zipfile
s=json.load(urllib.request.urlopen('http://127.0.0.1:8102/status',timeout=3))
if s['demo_running']:
 print(json.dumps(dict(demo_running=True)));raise SystemExit(0)
c=json.loads(pathlib.Path('/data/calibration.json').read_text())
h=json.load(urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=3))
raw=urllib.request.urlopen('http://127.0.0.1:8001/frame.tracking-rgbd',timeout=3).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:m=json.loads(z.read('metadata.json'))
source=m['source_metadata']
assert source['timestamp_valid'] and abs(source['depth_time']-source['color_time'])<=.012
h['frame_age_s']=time.monotonic()-source['depth_time']
bounds={}
for node in ast.walk(ast.parse(pathlib.Path('/app/g1_can_handoff/task.py').read_text())):
 if not isinstance(node,ast.Compare) or len(node.ops)!=2 or not all(isinstance(op,ast.LtE) for op in node.ops):continue
 a,b=node.comparators
 if (isinstance(node.left,ast.Constant) and isinstance(b,ast.Constant) and isinstance(a,ast.Subscript)
     and isinstance(a.value,ast.Name) and a.value.id=='c' and isinstance(a.slice,ast.Constant)
     and a.slice.value in (0,1)):bounds[str(a.slice.value)]=[node.left.value,b.value]
body=s.get('measured',{}).get('body')
print(json.dumps(dict(config=c,bounds=bounds,intrinsics=m['calibration']['intrinsics'],
 camera_serial=source['camera_serial'],phase=s['phase'],demo_running=s['demo_running'],body=body,
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
                label=f'Can base center: {x0*100:g}–{x1*100:g} cm forward · {y0*100:g}–{y1*100:g} cm robot-left')
