"""Read-only native camera deployment readiness; no lease or trial start."""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from operator_panel import Wendy


def main():
    root=Path(__file__).resolve().parent/'src'
    expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
    script='''import hashlib,io,json,pathlib,time,urllib.request,zipfile
expected=EXPECTED
actual={name:hashlib.sha256((pathlib.Path('/app/src')/name).read_bytes()).hexdigest() for name in expected}
assert actual==expected,'Camera source hashes differ'
def get(path):return json.load(urllib.request.urlopen('http://127.0.0.1:8001'+path,timeout=3))
health=get('/health')
assert health['healthy'] and health['serial']=='338622073335' and health['fps']==15,health
assert health['tracking_rgbd_schema']=='wendy.g1.compact-tracking.v1',health
frames=[]
for index in range(2):
 path='/frame.tracking-rgbd'
 if frames:path+='?after_frame_id='+str(frames[-1]['frame_id'])+'&stream_id='+frames[-1]['stream_id']
 raw=urllib.request.urlopen('http://127.0.0.1:8001'+path,timeout=3).read()
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:
  cm=json.loads(archive.read('metadata.json'));meta=cm['source_metadata']
 age=(time.monotonic()-meta['depth_time'])*1000
 assert meta['timestamp_valid'] and 0<=age<=400 and abs(meta['color_time']-meta['depth_time'])<=.012,meta
 assert meta['frame_id']==cm['source_frame_id'] and meta['captured_at_unix_ns']==cm['captured_at_unix_ns']
 assert meta['camera_serial']=='338622073335' and meta['intrinsics']==cm['calibration']['intrinsics']
 frames.append(dict(frame_id=meta['frame_id'],stream_id=meta['stream_id'],exposure_age_ms=age,bytes=len(raw)))
assert frames[1]['frame_id']>frames[0]['frame_id']
print(json.dumps(dict(camera_verified=True,source_sha256=actual,frames=frames)))
'''.replace('EXPECTED',repr(expected))
    client=Wendy()
    try:
        for attempt in range(15):
            try:
                result=client.tool('container_exec',dict(app_name='g1-can-camera',command=['python3','-c',script],timeout_seconds=12,max_bytes=10000))
                payload=result.get('structuredContent') or json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
                if payload['exit_code']==0:
                    print(payload['stdout'],end='');return 0
                error=payload.get('stderr') or str(payload)
            except Exception as exc:error=str(exc)
            if attempt==14:raise RuntimeError(error)
            time.sleep(1)
    finally:client.process.terminate()


if __name__=='__main__':raise SystemExit(main())
