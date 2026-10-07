"""Check a running taxi server's native bridge, controls, and rendered cameras."""
import io
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image, ImageStat

BASE='http://127.0.0.1:8788'

def get(path):
    with urllib.request.urlopen(BASE+path,timeout=10) as response: return json.load(response)

def command(cmd):
    req=urllib.request.Request(BASE+'/ctl/sim-cmd',json.dumps({'cmd':cmd}).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as response: return json.load(response)

def until(predicate):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        state=get('/api/status')
        if predicate(state): return state
        time.sleep(.05)
    raise AssertionError(f'Timed out, last state: {state.get("phase")}, {state.get("time")}')

def main():
    out=Path(__file__).parent/'runtime';out.mkdir(exist_ok=True)
    command('pause');s=until(lambda s:s.get('paused'));a=s['time']
    time.sleep(.2);assert get('/api/status')['time']==a
    command('step');s=until(lambda s:s['time']>a);b=s['time'];assert abs(b-a-.01)<.001
    command('resume');until(lambda s:not s['paused'] and s['time']>b)
    scene=get('/simslot/autonomous-taxi/scene.json');state=get('/simslot/autonomous-taxi/state.json')
    assert len(scene['geoms'])==len(state['pose'])>1000
    cameras={}
    for name in ['front','front_left','front_right','left','right','rear','chase']:
        with urllib.request.urlopen(BASE+'/camera/'+name+'.jpg',timeout=10) as response: blob=response.read()
        im=Image.open(io.BytesIO(blob));assert max(ImageStat.Stat(im).stddev)>1
        assert im.size==((640,400) if name=='chase' else (300,180))
        (out/(name+'.jpg')).write_bytes(blob)
        cameras[name]={'dimensions':im.size,'bytes':len(blob)}
    command('reset');until(lambda s:s['time']<2 and s['trips']==0)
    result={'pause':True,'single_step_seconds':round(b-a,2),'resume':True,'reset':True,'native_geometries':len(scene['geoms']),'cameras':cameras}
    (out/'live-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__': main()
