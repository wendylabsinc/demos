"""Capture a fresh real RGB-D/setup seed via Wendy; never opens a motor writer."""
import argparse
import json
from pathlib import Path
import tarfile
from operator_panel import Wendy
from trial_evidence import fetch_artifact,result_payload

PROBE=r'''
import hashlib,io,json,threading,time,tarfile
from pathlib import Path
import urllib.request
import numpy as np
from camera_input import Camera,geometry
from scene_refresh import refresh_scene
from pose_alignment import PoseHistory,corrected_config
from target_tracker import TargetTracker
def status():return json.load(urllib.request.urlopen('http://127.0.0.1:8102/status',timeout=3))
before=status()
allow_held=ALLOW_HELD
assert not before['demo_running'],before
assert (before['phase'] in ('PASSIVE','BLOCKED') and not before['adapter_initialized']) or (allow_held and before['phase']=='FROZEN' and before['adapter_initialized']),before
poses=PoseHistory();stop=threading.Event();errors=[]
def sample():
 while not stop.is_set():
  try:
   b=status()['measured']['body'];poses.add(b['time'],b['imu_rpy'],b['q'][12:15])
  except ValueError as e:
   if 'must advance' not in str(e):errors.append(str(e))
  except Exception as e:errors.append(str(e))
  stop.wait(.01)
thread=threading.Thread(target=sample,daemon=True);thread.start();camera=Camera()
try:
 camera.acquire();camera.frame();time.sleep(.5)
 frame=camera.frame();measured=status()['measured'];exposure_pose=poses.nearest(frame['timestamp'])
 config=refresh_scene(json.loads(Path('/data/calibration.json').read_text()),frame,measured,
                      frame['meta']['camera_serial'],exposure_pose=exposure_pose)
 frame=camera.frame();measured=status()['measured'];exposure_pose=poses.nearest(frame['timestamp'])
 aligned=corrected_config(config,exposure_pose)
 observation,vision=geometry(frame,aligned,tracker=TargetTracker())
 geometry_done=time.monotonic()
 after=status()
 assert not after['demo_running'] and after['phase']==before['phase'] and after['adapter_initialized']==before['adapter_initialized'],after
 if not before['adapter_initialized']:assert after['motion_commands_sent']==before['motion_commands_sent'],after
 else:assert np.max(np.abs(np.asarray(after['measured']['body']['q'][15:29])-before['measured']['body']['q'][15:29]))<.04,after
 setup=dict(schema='wendy.g1.real-sim-seed.v1',captured_unix_ns=time.time_ns(),
  captured_monotonic_s=time.monotonic(),config=config,measured=measured,
  exposure_age_at_geometry_ms=(geometry_done-frame['timestamp'])*1000,
  camera_intrinsics=camera.intrinsics,can_center_world_m=observation.center.tolist(),
  stationary_lower_body_support=True,exposure_pose=exposure_pose,vision=vision,
  source_metadata=frame['meta'],motion_commands_before=before['motion_commands_sent'],
  motion_commands_after=after['motion_commands_sent'],pose_poll_errors=errors,
  read_only_probe=True,controller_held=before['adapter_initialized'],
  limitations=['Real base is factory-balanced; simulation base is supported.',
   'Can texture, mass, friction and deformation are approximate in dynamics.',
   'Real camera and can geometry are measured; rendered RGB-D is synthetic.'])
 directory=Path('/data/sim-seeds')/str(setup['captured_unix_ns']);directory.mkdir(parents=True)
 (directory/'setup.json').write_text(json.dumps(setup,indent=2)+'\n')
 np.savez_compressed(directory/'frame.npz',color_bgr=frame['color_bgr'],depth_m=frame['depth_m'],rays=frame['rays'],
                     metadata_json=json.dumps(dict(meta=frame['meta'],timestamp=frame['timestamp'],config=aligned)))
 (directory/'frame.source.zip').write_bytes(frame['source_zip']) if 'source_zip' in frame else None
 files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in directory.iterdir()}
 (directory/'manifest.json').write_text(json.dumps(dict(files=files),indent=2))
 output=directory.with_suffix('.tar.gz')
 with tarfile.open(output,'w:gz') as tar:
  for p in directory.iterdir():tar.add(p,arcname=p.name)
 print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                      can_center_world_m=setup['can_center_world_m'],exposure_age_ms=(time.monotonic()-frame['timestamp'])*1000,
                      motion_commands_added=after['motion_commands_sent']-before['motion_commands_sent'])))
finally:camera.close();stop.set();thread.join(timeout=4)
'''


def capture(directory,allow_held=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    client=Wendy()
    try:
        metadata=json.loads(result_payload(client.tool('container_exec',dict(app_name='DEV-g1-handoff-physical',
            command=['python3','-c',PROBE.replace('ALLOW_HELD',repr(allow_held))],timeout_seconds=25,max_bytes=20000)))['stdout'])
        archive=directory/'seed.tar.gz'
        transfer=fetch_artifact(client,'DEV-g1-handoff-physical',metadata['path'],metadata,archive)
        with tarfile.open(archive) as tar:tar.extractall(directory,filter='data')
        (directory/'capture.json').write_text(json.dumps(transfer,indent=2)+'\n')
        print(json.dumps(transfer))
        return directory/'setup.json'
    finally:client.process.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True)
    parser.add_argument('--allow-held',action='store_true',help='Read only: capture a stable FROZEN controller hold')
    args=parser.parse_args();capture(args.directory,args.allow_held)
