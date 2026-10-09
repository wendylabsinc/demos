"""Local operator release button through the native Wendy MCP transport.

Run on the operator Mac: python3 operator_panel.py. No motion starts on load.
"""
import json
from operator_state import describe_state,ERROR_CASES
from placement_overlay import PlacementGuide
from pathlib import Path
import queue
import secrets
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE = '10.10.20.238:50052'
PORT = 8197
DEPLOYING = Path("/tmp/wendy-g1-nx2-deployment.lock")
TOKEN = secrets.token_urlsafe(32)

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>G1 operator controls</title>
<style>body{margin:0;background:#10151c;color:#edf2f8;font:18px system-ui}main{max-width:1000px;margin:32px auto;padding:24px}h1{font-size:26px}button{font:700 22px system-ui;background:#ffca68;color:#181510;border:0;border-radius:12px;padding:22px;width:100%;cursor:pointer}button:disabled{opacity:.4;cursor:default}img{display:block;width:100%;max-width:800px;border-radius:12px}#state{white-space:pre-wrap;padding:18px;background:#1c2530;border-radius:10px}#robot-card{padding:20px;border:2px solid #526275;border-radius:14px;margin:18px 0;background:#1c2530}#robot-title{font-size:28px;margin:0 0 10px}#robot-card[data-tone="ready"]{border-color:#8adeb2}#robot-card[data-tone="busy"]{border-color:#a8cafa}#robot-card[data-tone="hold"],#robot-card[data-tone="blocked"]{border-color:#ffca68}#robot-owner{display:inline-block;border-radius:24px;background:#303c4a;padding:6px 12px;font-size:16px}#robot-next{margin-bottom:0}#robot-error{color:#ffb0a8;overflow-wrap:anywhere}details{font-size:14px;color:#b7c4d2;margin:12px 0}.camera-view{position:relative;max-width:800px}#pickup-overlay{display:none;position:absolute;inset:0;width:100%;height:100%;pointer-events:none}#pickup-area{fill:#62ffab33;stroke:#62ffab;stroke-width:3;vector-effect:non-scaling-stroke}#pickup-overlay[data-reference="true"] #pickup-area{fill:#ffca681a;stroke:#ffca68;stroke-dasharray:9 6}#placement-message{color:#9df4c6}#placement-message[data-reference="true"]{color:#ffca68}</style></head><body><main>
<h1>G1 · Coke demo controls</h1><p>Start runs one pickup attempt. Stop holds the measured arm position. Release returns upper-body control to Unitree and relaxes both hands. Support the can and raised arms before releasing.</p>
<p>Pressing Release confirms that the can and raised arms are supported.</p>
<section id="robot-card" data-tone="unknown" aria-live="polite"><h2 id="robot-title">Checking robot state…</h2><p id="robot-activity">Checking attempt…</p><span id="robot-owner">Control ownership unknown</span><p id="perception-summary">Checking perception mode…</p><p id="robot-next">Connecting to the controller.</p><p id="robot-error"></p></section>
<style>.error-card{background:#241c21;border:1px solid #995b60;border-left:5px solid #ffb0a8;border-radius:12px;padding:20px;margin:18px 0;line-height:1.5}.error-card[hidden]{display:none}.error-card h2{font-size:23px;margin:6px 0;color:#ffd2cd}.error-badge{font-size:12px;letter-spacing:.08em;color:#ffb0a8;font-weight:700}.error-card p{margin:8px 0}.error-next{color:#ffda98}.error-card details{color:#d0bfc7}.error-card pre,#placement-error-raw{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;background:#10151c;padding:12px;border-radius:8px}.error-note{font-size:12px;color:#b6aab1}.error-card code{font-size:13px;color:#edf2f8}</style>
<section id="last-demo-error" class="error-card" role="status" aria-labelledby="demo-error-title" hidden><span class="error-badge">LAST ATTEMPT</span><h2 id="demo-error-title"></h2><p id="demo-error-summary"></p><p class="error-next"><strong>Next step:</strong> <span id="demo-error-next"></span></p><details><summary>Technical details</summary><p>Source: <code id="demo-error-source"></code></p><pre id="demo-error-raw"></pre></details><p class="error-note">Kept until the next Start is accepted, including after refresh.</p></section>
<button id="start" disabled style="background:#8adeb2;margin-bottom:12px">Start demo</button>
<button id="stop" style="background:#a8cafa;margin-bottom:12px">Stop and hold</button>
<button id="release">Release upper body + fingers to Unitree</button><details><summary>Controller details</summary><p id="state">Checking controller…</p></details><p id="action-status" aria-live="polite"></p>
<div class="camera-view"><img id="video" alt="Live color camera"><svg id="pickup-overlay" preserveAspectRatio="none" aria-label="Coke can pickup placement area"><polygon id="pickup-area"/></svg></div><p id="placement-message">Loading calibrated pickup area…</p><p id="video-state">Connecting to live camera…</p><p id="camera-health"></p>
<details id="placement-error-details" hidden><summary>Placement diagnostics</summary><pre id="placement-error-raw"></pre></details>
<script>const button=document.querySelector('#release'),state=document.querySelector('#state'),videoState=document.querySelector('#video-state'),cameraHealth=document.querySelector('#camera-health');let video=document.querySelector('#video');let busy=false,lastImage=0,imagePending=false,imageGeneration=0,cameraWaiting=false;
async function call(path,data){let r=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json','X-Operator-Token':'TOKEN_VALUE'},body:data?JSON.stringify(data):undefined});let s=await r.json();if(!r.ok)throw Error(s.error||'Controller request failed');return s;}
const actionStatus=document.querySelector('#action-status');const start=document.querySelector('#start'),stop=document.querySelector('#stop');let starting=false,polling=false,stateRevision=0;
const lastDemoError=document.querySelector('#last-demo-error');let savedDemoError='';
const demoErrorTitle=document.querySelector('#demo-error-title'),demoErrorSummary=document.querySelector('#demo-error-summary'),demoErrorNext=document.querySelector('#demo-error-next'),demoErrorSource=document.querySelector('#demo-error-source'),demoErrorRaw=document.querySelector('#demo-error-raw');const errorCases=ERROR_CASES_JSON;
function describeDemoError(message){return errorCases.find(rule=>rule.triggers.some(trigger=>message.includes(trigger)))||{title:'Demo stopped',summary:'The attempt stopped. Technical details contain the full error.',action:'Check the reported problem before retrying.',source:'Not identified'};}
function rememberDemoError(message){if(!message)return;if(savedDemoError!==message){savedDemoError=message;try{sessionStorage.setItem('g1-last-demo-error',message);}catch(e){}}const info=describeDemoError(savedDemoError);demoErrorTitle.textContent=info.title;demoErrorSummary.textContent=info.summary;demoErrorNext.textContent=info.action;demoErrorSource.textContent=info.source;demoErrorRaw.textContent=savedDemoError;lastDemoError.hidden=false;}
function clearDemoError(){savedDemoError='';lastDemoError.hidden=true;try{sessionStorage.removeItem('g1-last-demo-error');}catch(e){}}
try{rememberDemoError(sessionStorage.getItem('g1-last-demo-error'));}catch(e){}
function show(s){if(!s.demo_running)rememberDemoError(s.demo_error||s.error||s.presentation?.failure);renderRobotState(s);start.disabled=!!s.deployment_in_progress||starting||!!s.demo_running||s.phase==='HOLDING_CAN';state.textContent=(s.deployment_in_progress?'Updating demo — Start will be available after verification\n':'')+'Controller phase: '+s.phase+'\n'+(s.presentation?.activity||'Checking attempt')+(s.presentation?.failure?'\n'+s.presentation.failure:'')+(s.camera_max_age_ms?'\nCamera limit: '+s.camera_max_age_ms+' ms':'')+(s.target_identity==='color_depth'?'\nDetector: color/depth + tracking · YOLO not required':s.yolo_confidence_strictly_above?'\nYOLO confidence: above '+Math.round(s.yolo_confidence_strictly_above*100)+'%':'');}
start.onclick=async()=>{if(starting)return;const revision=++stateRevision;starting=true;start.disabled=true;demoActive=true;placementVisible();state.textContent='Starting one pickup attempt…';actionStatus.textContent='Sending Start request…';try{let result=await call('/start',{});if(revision!==stateRevision)return;clearDemoError();actionStatus.textContent='Start accepted — one pickup attempt requested.';show(result);}catch(e){if(revision!==stateRevision)return;actionStatus.textContent='Start request failed: '+e.message;rememberDemoError('Start request failed: '+e.message);state.textContent=e.message;}finally{starting=false;poll();}};
stop.onclick=async()=>{const revision=++stateRevision;state.textContent='Stopping and holding…';try{const result=await call('/stop',{});if(revision===stateRevision)show(result);}catch(e){if(revision===stateRevision)state.textContent=e.message;}};
button.onclick=async()=>{if(busy)return;const revision=++stateRevision;busy=true;button.disabled=true;state.textContent='Releasing upper body and fingers…';try{const result=await call('/release',{can_supported:true});if(revision===stateRevision)show(result);}catch(e){if(revision===stateRevision)state.textContent=e.message;}finally{busy=false;button.disabled=false;}};
function cameraDisplayState(){const age=lastImage?Math.max(0,Date.now()-lastImage):null;videoState.textContent=age===null?'Waiting for the first camera image…':cameraWaiting||age>1000?'Camera display paused · showing last image received '+(age/1000).toFixed(1)+' s ago · retrying…':'Camera display updating · image received '+age+' ms ago';}
function refreshVideo(){
 cameraDisplayState();if(imagePending)return;
 imagePending=true;const generation=++imageGeneration,next=new Image();let finished=false;
 const timeout=setTimeout(failed,3000);
 function failed(){if(finished||generation!==imageGeneration)return;finished=true;clearTimeout(timeout);imagePending=false;cameraWaiting=true;guideValid=false;cameraDisplayState();placementVisible();}
 next.onerror=failed;
 next.onload=async()=>{
  try{await next.decode();if(!next.naturalWidth)throw Error('Empty camera image');}catch(e){failed();return;}
  if(finished||generation!==imageGeneration)return;
  finished=true;clearTimeout(timeout);next.id='video';next.alt='Live color camera';
  // Swap the already decoded image. Failed or late requests never touch the
  // displayed frame, so camera contention cannot turn it into a broken image.
  video.replaceWith(next);video=next;lastImage=Date.now();imagePending=false;cameraWaiting=false;cameraDisplayState();placementVisible();
 };
 next.src='http://10.10.20.238:8001/frame.jpg?t='+Date.now();
}
refreshVideo();setInterval(refreshVideo,250);
async function health(){try{let h=await call('/camera-health');cameraHealth.textContent=(h.healthy&&h.frame_age_s>=0&&h.frame_age_s<=.4?'Source camera fresh':'Source camera stale')+' · frame '+h.frame_id+' · exposure '+Math.round(h.frame_age_s*1000)+' ms old · '+h.fps+' fps';}catch(e){cameraHealth.textContent='Source camera health unavailable';}}
health();setInterval(health,1000);
async function poll(){if(polling||busy||starting)return;polling=true;const revision=stateRevision;try{const result=await call('/status');if(revision===stateRevision)show(result);}catch(e){if(revision!==stateRevision)return;state.textContent=e.message;robotTitle.textContent='Controller unavailable';robotActivity.textContent='Attempt state unknown';robotCard.dataset.tone='unknown';robotOwner.textContent='Control ownership unknown';robotNext.textContent='Check the controller connection.';robotError.textContent='';guideValid=false;placementVisible();}finally{polling=false;}}
poll();setInterval(poll,1000);
const perceptionSummary=document.querySelector('#perception-summary');const robotCard=document.querySelector('#robot-card'),robotTitle=document.querySelector('#robot-title'),robotActivity=document.querySelector('#robot-activity'),robotOwner=document.querySelector('#robot-owner'),robotNext=document.querySelector('#robot-next'),robotError=document.querySelector('#robot-error'),pickupOverlay=document.querySelector('#pickup-overlay'),pickupArea=document.querySelector('#pickup-area'),placementMessage=document.querySelector('#placement-message');let guideValid=false,guidePresent=false,guideReceived=0,demoActive=false,placementPolling=false;
function placementVisible(){const current=guideValid&&Date.now()-guideReceived<2500&&lastImage&&Date.now()-lastImage<1000;pickupOverlay.style.display=guidePresent&&lastImage?'block':'none';pickupOverlay.dataset.reference=String(!current);placementMessage.dataset.reference=String(!current);if(guidePresent&&!current&&!placementMessage.textContent.startsWith('Reference only'))placementMessage.textContent='Reference only — current placement is not verified. '+placementMessage.textContent;}
function renderRobotState(s){demoActive=!!s.demo_running;perceptionSummary.textContent=s.target_identity==='color_depth'?'Perception: color/depth tracking · YOLO not required':s.target_identity==='yolo_once'?'Perception: initial YOLO identity, then visual/depth tracking':'Perception: YOLO and visual/depth tracking';const p=s.presentation;if(p){robotTitle.textContent=p.title;robotActivity.textContent=p.activity;robotCard.dataset.tone=p.tone;robotOwner.textContent=p.ownership;robotNext.textContent=p.next_action;robotError.textContent=p.failure&&p.failure_label==='Current problem'?(p.failure_summary||p.failure):'';}placementVisible();}
function placementError(message){const details=document.querySelector('#placement-error-details'),raw=document.querySelector('#placement-error-raw');details.hidden=!message;raw.textContent=message||'';if(!message)return 'Current placement is not verified.';const info=describeDemoError(message);return info.source==='Not identified'?'Placement feedback unavailable. Open diagnostics for details.':info.summary;}
async function placement(){if(placementPolling)return;placementPolling=true;try{const p=await call('/placement');guideReceived=Date.now();guideValid=!!p.valid;const explanation=placementError(p.error);if(p.polygon){guidePresent=true;pickupOverlay.setAttribute('viewBox',`0 0 ${p.width} ${p.height}`);pickupArea.setAttribute('points',p.polygon.map(point=>point.join(',')).join(' '));placementMessage.textContent=guideValid?'Place the center of the can base inside the green area. '+p.label:'Reference only — amber outline uses the last calibration. '+explanation;}else{guidePresent=false;placementMessage.textContent=explanation;}}catch(e){guideValid=false;placementMessage.textContent='Reference only — '+placementError(e.message);}finally{placementPolling=false;placementVisible();}}
placement();setInterval(placement,1000);setInterval(placementVisible,250);

</script></main></body></html>'''.replace('ERROR_CASES_JSON',json.dumps(ERROR_CASES))

class Wendy:
    def __init__(self):
        self.process = subprocess.Popen(['wendy','mcp','serve'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.responses=queue.Queue();self.lock=threading.Lock();self.sequence=0
        threading.Thread(target=self.read,daemon=True).start()
        try:
            self.rpc('initialize',{'protocolVersion':'2024-11-05','capabilities':{},
                                  'clientInfo':{'name':'g1-operator-release','version':'1'}})
            self.process.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'})+'\n');self.process.stdin.flush()
            self.tool('wendy_status',{})
            self.tool('device_connect',{'device':DEVICE})
        except Exception:
            self.process.terminate()
            raise
    def read(self):
        for line in self.process.stdout:
            try:self.responses.put(json.loads(line))
            except json.JSONDecodeError:continue
    def rpc(self,method,params):
        self.sequence+=1;ident=self.sequence
        self.process.stdin.write(json.dumps({'jsonrpc':'2.0','id':ident,'method':method,'params':params})+'\n');self.process.stdin.flush()
        while True:
            row=self.responses.get(timeout=30)
            if row.get('id')==ident:
                if 'error' in row:raise RuntimeError(row['error'].get('message','Wendy transport failed'))
                return row['result']
    def tool(self,name,arguments):
        result=self.rpc('tools/call',{'name':name,'arguments':arguments})
        if result.get('isError'):raise RuntimeError(str(result.get('content')))
        return result
    def request(self,release=False,*,action=None):
        # Fixed target, fixed app, fixed endpoints: never accept command text.
        action='release' if release else (action or 'status')
        paths={'status':'status','release':'release','start':'demo/start','stop':'stop'}
        if action not in paths:raise ValueError('Unknown operator action')
        if action in ('start','start_force') and DEPLOYING.is_file():raise RuntimeError('Demo update in progress; Start will be available after verification')
        data=",data=b'{\"can_supported\":true}',headers={'Content-Type':'application/json'}" if action=='release' else (",data=b'{}',headers={'Content-Type':'application/json'}" if action!='status' else '')
        script="""import json,urllib.request,urllib.error
url='http://127.0.0.1:8102/"""+paths[action]+"""'
request=urllib.request.Request(url"""+data+""")
try:
 response=urllib.request.urlopen(request,timeout=12);s=json.load(response)
 print(json.dumps({k:s.get(k) for k in ['phase','error','motion_commands_sent','adapter_initialized','telemetry_healthy','camera_max_age_ms','yolo_confidence_strictly_above','target_identity','yolo_required','demo_running','demo_result','demo_error']}))
except urllib.error.HTTPError as e:
 print(json.dumps({'error':json.load(e).get('error')}))
"""
        with self.lock:
            result=self.tool('container_exec',{'app_name':'DEV-g1-handoff-physical',
                  'command':['python3','-c',script],'timeout_seconds':20,'max_bytes':4000})
        payload=result.get('structuredContent')
        if payload is None:
            payload=json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
        if payload.get('exit_code')!=0:raise RuntimeError(payload.get('stderr') or 'Controller unavailable')
        state=json.loads(payload['stdout'])
        state['deployment_in_progress']=DEPLOYING.is_file()
        state['presentation']=describe_state(state)
        if state.get('error') and 'phase' not in state:raise RuntimeError(state['error'])
        if action=='release' and (state.get('phase')!='PASSIVE' or state.get('adapter_initialized')):
            raise RuntimeError(state.get('error') or 'Release not confirmed; controller retains hold')
        return state

class Handler(BaseHTTPRequestHandler):
    def send(self,data,status=200,kind='application/json'):
        body=data.encode() if isinstance(data,str) else json.dumps(data).encode()
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        self.end_headers();self.wfile.write(body)
    def valid(self):
        return (self.headers.get('Host')==f'127.0.0.1:{PORT}' and
                self.headers.get('X-Operator-Token')==TOKEN and
                self.headers.get('Origin',f'http://127.0.0.1:{PORT}')==f'http://127.0.0.1:{PORT}')
    def do_GET(self):
        if self.headers.get('Host')!=f'127.0.0.1:{PORT}':self.send({'error':'Local operator only'},403);return
        if self.path=='/':self.send(PAGE.replace('TOKEN_VALUE',TOKEN),kind='text/html; charset=utf-8');return
        if self.path not in ('/status','/camera-health','/placement') or not self.valid():self.send({'error':'Invalid operator request'},403);return
        try:
            if self.path=='/placement':
                guide=getattr(self.server,'placement',None)
                self.send(guide.read() if guide is not None else {'valid':False,'error':'Placement guide unavailable'})
            elif self.path=='/camera-health':
                with urllib.request.urlopen('http://10.10.20.238:8001/health',timeout=2) as response:health=json.load(response)
                self.send({k:health.get(k) for k in ['healthy','frame_id','frame_age_s','fps']})
            else:self.send(self.server.wendy.request())
        except Exception as exc:self.send({'error':str(exc)},503)
    def do_POST(self):
        if self.path not in ('/release','/start','/stop') or not self.valid():self.send({'error':'Invalid operator request'},403);return
        if self.path=='/start' and DEPLOYING.is_file():self.send({'error':'Demo update in progress; Start will be available after verification'},409);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<1024:raise ValueError('Invalid operator request')
            data=json.loads(self.rfile.read(size))
            if self.path=='/release':
                if data.get('can_supported') is not True:raise ValueError('Support must be confirmed before release')
                self.send(self.server.wendy.request(release=True))
            else:self.send(self.server.wendy.request(action=self.path[1:]))
        except Exception as exc:self.send({'error':str(exc)},409)
    def log_message(self,*args):pass

def main():
    server=ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    server.wendy=Wendy()
    server.placement=PlacementGuide(Wendy).start()
    print(f'G1 operator panel: http://127.0.0.1:{PORT}/',flush=True)
    try:server.serve_forever()
    finally:server.placement.close();server.wendy.process.terminate();server.server_close()

if __name__=='__main__':main()
