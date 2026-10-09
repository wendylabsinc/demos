"""NX2 startup verification; optional idle budget update, never starts motion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from operator_panel import Wendy


def main(argv=()):
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configure-camera-max-age-ms',type=float)
    args=parser.parse_args(argv)
    configured_env=json.loads((root/'wendy.json').read_text())['env']
    configured_limit=float(configured_env['HANDOFF_CAMERA_MAX_AGE_MS'])
    configured_policy=configured_env['HANDOFF_PERCEPTION_POLICY']
    configured_identity=configured_env['HANDOFF_TARGET_IDENTITY']
    configured_kp=float(configured_env['HANDOFF_ARM_KP'])
    if args.configure_camera_max_age_ms is not None and args.configure_camera_max_age_ms!=configured_limit:
        parser.error('Requested budget must equal this deployment descriptor')
    copied={s.strip() for group in re.findall(r'paths:\s*\[([^\]]+)\]',
             (root/'build.stagefile.yaml').read_text()) for s in group.split(',')}
    paths=[p for p in root.glob('*.py') if p.name in copied]
    for directory in ['adapter','g1_can_handoff']:
        paths.extend(p for p in (root/directory).rglob('*.py') if '__pycache__' not in p.parts)
    paths.extend(root/name for name in ['yolo/frame_limits.py','yolo/confidence.py'])
    paths.append(root/'g1_can_handoff/noise_aware_planner.json')
    expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    script="""import hashlib,io,json,pathlib,time,urllib.request
from PIL import Image
import control_settle
import physical_core
import planner_tuning
from types import SimpleNamespace
import os
expected=EXPECTED
actual={name:hashlib.sha256((pathlib.Path('/app')/name).read_bytes()).hexdigest() for name in expected}
assert actual==expected,'Deployed source hashes differ from checkout'
assert os.environ.get('HANDOFF_PLANNER_PROFILE','legacy')==CONFIGURED_PROFILE
effective={}
planner_tuning.configure(SimpleNamespace(ik=SimpleNamespace(IK_SETTINGS={'x_scale':1.})),effective)
def get(port,path):return json.load(urllib.request.urlopen('http://127.0.0.1:'+str(port)+path,timeout=5))
status=get(8102,'/status')
assert status['phase']=='PASSIVE' and not status['adapter_initialized'] and not status['demo_running'] and status['motion_commands_sent']==0,status
assert status['telemetry_healthy'],status
assert physical_core.body_temperature_limit_c()==CONFIGURED_BODY_TEMP
assert os.environ.get('HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M')==CONFIGURED_TABLE_OFFSET
assert status['arm_gains']['configured_kp']==[CONFIGURED_KP]*4+[40.]*3+[CONFIGURED_KP]*4+[40.]*3,status
assert status['arm_gains']['configured_kd']==[3.]*4+[1.5]*3+[3.]*4+[1.5]*3,status
assert status['arm_gains']['active'] is None,status
budget=CONFIGURED_LIMIT
assert status['target_identity']==CONFIGURED_IDENTITY,status
yolo={'required':False}
if status['yolo_required']:
 yolo=get(8112,'/health')
 assert yolo['healthy'] and yolo['backend']=='pytorch_cuda' and yolo['model_sha256']=='fab6252b0abc0613ef39c36ea2d75a061d80bb087f5eb9034ca33614c26301e9',yolo
 assert yolo['camera_max_age_ms']==budget,yolo
previous_budget=status['camera_max_age_ms']
if APPLY_CONFIG and previous_budget!=budget:
 request=urllib.request.Request('http://127.0.0.1:8102/perception-config',data=json.dumps({'camera_max_age_ms':budget}).encode(),headers={'Content-Type':'application/json'},method='POST')
 with urllib.request.urlopen(request,timeout=5) as response:status=json.load(response)
 assert status['phase']=='PASSIVE' and not status['adapter_initialized'] and not status['demo_running'] and status['motion_commands_sent']==0,status
assert status['camera_max_age_ms']==budget and status['yolo_confidence_strictly_above']==.1,status
assert status['perception_policy']==CONFIGURED_POLICY,status
assert status['periodic_yolo_recheck_s']==(None if CONFIGURED_POLICY=='phase_boundary' or CONFIGURED_IDENTITY!='yolo_reacquire' else 1.),status
recorder=get(8101,'/status')
assert recorder['recording'] and not recorder['error'] and not sum(recorder['dropped'].values()) and not recorder['missing_or_stale_required'],recorder
assert recorder['queue_depth']<=1000 and all(recorder['last_written_age_s'].get(topic,999)<.5 for topic in ['rt/lowstate','rt/dex3/left/state','rt/dex3/right/state']),recorder
frames=[]
for _ in range(2):
 health=get(8001,'/health')
 assert health['healthy'] and health['serial']=='338622073335' and health['frame_age_s']<.4,health
 raw=urllib.request.urlopen('http://127.0.0.1:8001/frame.jpg',timeout=5).read()
 image=Image.open(io.BytesIO(raw));image.load()
 assert image.size==(640,480) and any(low<high for low,high in image.convert('RGB').getextrema())
 frames.append({'frame_id':health['frame_id'],'age_s':health['frame_age_s'],'jpeg_sha256':hashlib.sha256(raw).hexdigest()})
 time.sleep(.15)
assert frames[1]['frame_id']>frames[0]['frame_id'],'Camera frame counter did not advance'
print(json.dumps({'startup_verified':True,'planner_settings':effective,'motion_readiness':'not_checked','previous_camera_max_age_ms':previous_budget,'controller':status,'source_sha256':actual,'camera_frames':frames,'yolo':yolo,'recorder':recorder}))
""".replace('EXPECTED',repr(expected)).replace('CONFIGURED_BODY_TEMP',repr(float(configured_env['HANDOFF_BODY_TEMPERATURE_LIMIT_C']))).replace('CONFIGURED_TABLE_OFFSET',repr(configured_env['HANDOFF_PLANNING_TABLE_HEIGHT_OFFSET_M'])).replace('CONFIGURED_PROFILE',repr(configured_env.get('HANDOFF_PLANNER_PROFILE','legacy'))).replace('CONFIGURED_KP',repr(configured_kp)).replace('CONFIGURED_LIMIT',repr(configured_limit)).replace('CONFIGURED_POLICY',repr(configured_policy)).replace('CONFIGURED_IDENTITY',repr(configured_identity)).replace('APPLY_CONFIG',repr(args.configure_camera_max_age_ms is not None))
    client=Wendy()
    try:
        for attempt in range(8):
            try:
                result=client.tool('container_exec',{'app_name':'DEV-g1-handoff-physical',
                    'command':['python3','-c',script],'timeout_seconds':25,'max_bytes':40000})
            except RuntimeError:
                if attempt==7:raise
                time.sleep(1);continue
            payload=result.get('structuredContent')
            if payload is None:payload=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
            if payload.get('exit_code')==0:
                print(payload['stdout'],end='');return 0
            if attempt==7:raise RuntimeError(payload.get('stderr') or str(payload))
            time.sleep(1)
    finally:client.process.terminate()


if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
