"""Feed stationary MuJoCo range measurements to SLAM Toolbox over its test-only API."""
import sys,time,math,json,urllib.request
from pathlib import Path
import numpy as np,mujoco
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'warehouse-demo'))
import demo
model=demo.make_scene();dog=demo.go2rl.Go2Controller(model=model)
ranges=[];gid=np.array([-1],np.int32)
for a in np.linspace(-math.pi,math.pi,360,endpoint=False):
    r=mujoco.mj_ray(model,dog.data,np.array([0.,0.,.8]),np.array([math.cos(a),math.sin(a),0.]),np.array([1,0,0,1,0,0],np.uint8),1,1,gid)
    ranges.append(float(r) if .35<r<=12 else None)
for _ in range(40):
    body=json.dumps(dict(capture_time=time.time(),odom=[0.,0.,0.],ranges=ranges)).encode()
    request=urllib.request.Request('http://127.0.0.1:18791/measurements',data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=2) as r:assert r.status==202
    time.sleep(.2)
with urllib.request.urlopen('http://127.0.0.1:18791/status') as r:status=json.load(r)
assert status['state']=='mapping' and status['pose'] is None and not status['motion_enabled'],status
print(json.dumps(status))
