"""Verify the live result, camera transport, calibrated localization and physics."""
import json, math, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
base='http://127.0.0.1:18992'
def get(path): return urllib.request.urlopen(base+path,timeout=5).read()
s=json.loads(get('/api/status'))
assert s['phase']=='found',s['phase']
arrived=next(e for e in reversed(s['events']) if e['kind']=='arrived_and_barked')
detection=next(e for e in reversed(s['events']) if e['kind']=='localized')
assert arrived['min_upright']>.95
assert arrived['contacts']>=2
assert .45<arrived['distance']<1.4
# This known simulator pose is exclusively a validation oracle, never a planner input.
from items import ITEMS
assert detection['item']==arrived['item']==s['requested_item']
oracle=next(item.position for item in ITEMS if item.name==s['requested_item'])
error=math.dist(detection['twin_hit'][:2],oracle)
assert error<.5,error
assert detection['confidence']>=.4
for name in ('cam1','cam2','cam3','cam4'):
    image=get(f'/camera/{name}/snapshot.jpg');assert image[:2]==b'\xff\xd8' and len(image)>10000
assert get('/bark.wav')[:4]==b'RIFF'
scene=json.loads(get('/simslot/warehouse/scene.json'))
state=json.loads(get('/simslot/warehouse/state.json'))
assert scene['engine']==state['engine']=='mujoco'
assert state['frame']>100 and state['contacts']
report=dict(passed=True,detector='YOLO11n simulated ten-item model',item=s['requested_item'],confidence=detection['confidence'],localization_error_m=error,arrival_distance_m=arrived['distance'],minimum_upright=arrived['min_upright'],peak_contacts=arrived['contacts'],ip_camera_snapshots=4,bark_events=s['bark'])
(ROOT/'runtime/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
