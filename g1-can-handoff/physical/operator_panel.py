"""Local operator release button through the native Wendy MCP transport.

Run on the operator Mac: python3 operator_panel.py. No motion starts on load.
"""
import json
import queue
import secrets
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEVICE = '10.10.20.238:50052'
PORT = 8197
TOKEN = secrets.token_urlsafe(32)

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>G1 operator controls</title>
<style>body{margin:0;background:#10151c;color:#edf2f8;font:18px system-ui}main{max-width:1000px;margin:32px auto;padding:24px}h1{font-size:26px}button{font:700 22px system-ui;background:#ffca68;color:#181510;border:0;border-radius:12px;padding:22px;width:100%;cursor:pointer}button:disabled{opacity:.4;cursor:default}img{display:block;width:100%;max-width:800px;border-radius:12px}#state{white-space:pre-wrap;padding:18px;background:#1c2530;border-radius:10px}</style></head><body><main>
<h1>G1 · Coke demo controls</h1><p>Start runs one pickup attempt. Stop holds the measured arm position. Release returns upper-body control to Unitree and relaxes both hands. Support the can and raised arms before releasing.</p>
<p>Pressing Release confirms that the can and raised arms are supported.</p>
<button id="start" style="background:#8adeb2;margin-bottom:12px">Start demo</button>
<button id="stop" style="background:#a8cafa;margin-bottom:12px">Stop and hold</button>
<button id="release">Release upper body + fingers to Unitree</button><p id="state">Checking controller…</p>
<img id="video" alt="Live color camera"><p id="video-state">Connecting to live camera…</p><p id="camera-health"></p>
<script>const button=document.querySelector('#release'),state=document.querySelector('#state'),video=document.querySelector('#video'),videoState=document.querySelector('#video-state'),cameraHealth=document.querySelector('#camera-health');let busy=false,lastImage=0,imagePending=false,imageStarted=0;
async function call(path,data){let r=await fetch(path,{method:data?'POST':'GET',headers:{'Content-Type':'application/json','X-Operator-Token':'TOKEN_VALUE'},body:data?JSON.stringify(data):undefined});let s=await r.json();if(!r.ok)throw Error(s.error||'Controller request failed');return s;}
const start=document.querySelector('#start'),stop=document.querySelector('#stop');let starting=false,polling=false;
function show(s){start.disabled=starting||!!s.demo_running||s.phase==='HOLDING_CAN';state.textContent=(s.demo_running?'Demo running · ':'Controller: ')+s.phase+(s.demo_error||s.error?'\n'+(s.demo_error||s.error):'')+(s.camera_max_age_ms?'\nCamera limit: '+s.camera_max_age_ms+' ms':'')+(s.yolo_confidence_strictly_above?'\nYOLO confidence: above '+Math.round(s.yolo_confidence_strictly_above*100)+'%':'');if(s.demo_result?.physical_lift_verified)state.textContent+='\nLift verified; support the can before releasing.';}
start.onclick=async()=>{if(starting)return;starting=true;start.disabled=true;state.textContent='Starting one pickup attempt…';try{show(await call('/start',{}));}catch(e){state.textContent=e.message;}finally{starting=false;poll();}};
stop.onclick=async()=>{state.textContent='Stopping and holding…';try{show(await call('/stop',{}));}catch(e){state.textContent=e.message;}};
button.onclick=async()=>{if(busy)return;busy=true;button.disabled=true;state.textContent='Releasing upper body and fingers…';try{show(await call('/release',{can_supported:true}));}catch(e){state.textContent=e.message;}finally{busy=false;button.disabled=false;}};
video.onload=()=>{imagePending=false;lastImage=Date.now();videoState.textContent='Live camera · received '+new Date(lastImage).toLocaleTimeString();};
video.onerror=()=>{imagePending=false;videoState.textContent='Camera image unavailable; reconnecting…';};
function refreshVideo(){if(lastImage&&Date.now()-lastImage>3000)videoState.textContent='Camera feed stalled';if(imagePending&&Date.now()-imageStarted>3000)imagePending=false;if(!imagePending){imagePending=true;imageStarted=Date.now();video.src='http://10.10.20.238:8001/frame.jpg?t='+imageStarted;}}
refreshVideo();setInterval(refreshVideo,250);
async function health(){try{let h=await call('/camera-health');cameraHealth.textContent=(h.healthy&&h.frame_age_s<=.4?'Fresh camera':'Camera stale')+' · frame '+h.frame_id+' · '+Math.round(h.frame_age_s*1000)+' ms · '+h.fps+' fps';}catch(e){cameraHealth.textContent='Camera health unavailable';}}
health();setInterval(health,1000);
async function poll(){if(polling||busy||starting)return;polling=true;try{show(await call('/status'));}catch(e){state.textContent=e.message;}finally{polling=false;}}
poll();setInterval(poll,1000);
</script></main></body></html>'''

class Wendy:
    def __init__(self):
        self.process = subprocess.Popen(['wendy','mcp','serve'], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.responses=queue.Queue();self.lock=threading.Lock();self.sequence=0
        threading.Thread(target=self.read,daemon=True).start()
        self.rpc('initialize',{'protocolVersion':'2024-11-05','capabilities':{},
                              'clientInfo':{'name':'g1-operator-release','version':'1'}})
        self.process.stdin.write(json.dumps({'jsonrpc':'2.0','method':'notifications/initialized'})+'\n');self.process.stdin.flush()
        self.tool('wendy_status',{})
        self.tool('device_connect',{'device':DEVICE})
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
        data=",data=b'{\"can_supported\":true}',headers={'Content-Type':'application/json'}" if action=='release' else (",data=b'{}',headers={'Content-Type':'application/json'}" if action!='status' else '')
        script="""import json,urllib.request,urllib.error
url='http://127.0.0.1:8102/"""+paths[action]+"""'
request=urllib.request.Request(url"""+data+""")
try:
 response=urllib.request.urlopen(request,timeout=12);s=json.load(response)
 print(json.dumps({k:s.get(k) for k in ['phase','error','motion_commands_sent','adapter_initialized','telemetry_healthy','camera_max_age_ms','yolo_confidence_strictly_above','demo_running','demo_result','demo_error']}))
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
        if self.path not in ('/status','/camera-health') or not self.valid():self.send({'error':'Invalid operator request'},403);return
        try:
            if self.path=='/camera-health':
                with urllib.request.urlopen('http://10.10.20.238:8001/health',timeout=2) as response:health=json.load(response)
                self.send({k:health.get(k) for k in ['healthy','frame_id','frame_age_s','fps']})
            else:self.send(self.server.wendy.request())
        except Exception as exc:self.send({'error':str(exc)},503)
    def do_POST(self):
        if self.path not in ('/release','/start','/stop') or not self.valid():self.send({'error':'Invalid operator request'},403);return
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
    print(f'G1 operator panel: http://127.0.0.1:{PORT}/',flush=True)
    try:server.serve_forever()
    finally:server.wendy.process.terminate();server.server_close()

if __name__=='__main__':main()
