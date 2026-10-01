"""Free-form command interpretation by a local, pretrained Qwen language model."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from urllib.request import Request, urlopen

ACTIONS = ['takeoff','hover','land','goto','follow','baseline','survey','clear','reject']
SCHEMA = {'type':'object','properties':{'actions':{'type':'array','minItems':1,'maxItems':5,
          'items':{'type':'object','properties':{'action':{'type':'string','enum':ACTIONS},
          'zone':{'type':'string','enum':['A','B','C','D','home','none']},
          'hazard':{'type':'string','enum':['fire','oil','none']}},
          'required':['action','zone','hazard'],'additionalProperties':False}}},
          'required':['actions'],'additionalProperties':False}
PROMPT = '''You translate instructions for a construction inspection drone into JSON actions.
Zone A is the materials storage. Zone B is the fuel depot. Zone C is equipment. Zone D is excavation.
Use goto with zone A/B/C/D/home for destinations. Use follow for following, staying with, keeping up with, tracking, or accompanying the worker. A request to stay WITH a person means follow, not hover.
Use baseline for recording a reference mission. Use survey for scanning and comparing a new mission.
Use takeoff for launching. Use hover to stop, wait, stay where you are, keep your current position or pause.
Use land ONLY when the user says to land, return to base, or return home. Stopping movement does not mean landing.
Use clear for removing simulated incidents. Incident creation requires the visual placement controls.
If the user reports a hazard and asks to investigate, use goto for its named area. Never create a hazard from a report.
For an unsupported request use reject. Never invent actions. Preserve the requested order of multiple instructions.
Only act on the user instruction. Usually return ONE action. Do not visit other zones or do a baseline unless requested. All objects need zone and hazard; use none if irrelevant.
Examples:
"Stay with that worker and keep a safe distance" -> {"actions":[{"action":"follow","zone":"none","hazard":"none"}]}
"Follow me" -> {"actions":[{"action":"follow","zone":"none","hazard":"none"}]}
"Inspect the fuel storage" -> {"actions":[{"action":"goto","zone":"B","hazard":"none"}]}
"There is a fire near the materials. Go investigate." -> {"actions":[{"action":"goto","zone":"A","hazard":"none"}]}
"Go to zone A" -> {"actions":[{"action":"goto","zone":"A","hazard":"none"}]}
"Scan for changes" -> {"actions":[{"action":"survey","zone":"none","hazard":"none"}]}
"Stop flying and stay where you are" -> {"actions":[{"action":"hover","zone":"none","hazard":"none"}]}
"Wait here" -> {"actions":[{"action":"hover","zone":"none","hazard":"none"}]}
"Go home and land" -> {"actions":[{"action":"land","zone":"none","hazard":"none"}]}
"Take a reference flight" -> {"actions":[{"action":"baseline","zone":"none","hazard":"none"}]}
"Fly to excavation and land" -> {"actions":[{"action":"goto","zone":"D","hazard":"none"},{"action":"land","zone":"none","hazard":"none"}]}
Return JSON only.
'''


class IntentService:
    def __init__(self):
        self.url = os.environ.get('YARD_DRONE_LLM_URL','http://127.0.0.1:9895')
        self.process = None
        self.ready = False
        self.error = None
        self._lock = threading.Lock()

    def start(self):
        binary = os.environ.get('YARD_DRONE_LLAMA_BIN','/opt/llama/llama-server')
        model = os.environ.get('YARD_DRONE_LLM_MODEL','/opt/models/Qwen3-0.6B-Q8_0.gguf')
        if Path(binary).is_file() and Path(model).is_file():
            self.process = subprocess.Popen([binary,'-m',model,'--host','127.0.0.1','--port','9895',
                '-c','2048','-t','2','-ngl','0','--jinja','--chat-template-kwargs',
                '{"enable_thinking":false}','--no-webui','--parallel','1','--cache-ram','0'], stdout=subprocess.DEVNULL,
                stderr=open('/tmp/yard-language.log','a'))
        deadline = time.monotonic()+90
        while time.monotonic()<deadline:
            try:
                with urlopen(self.url+'/health',timeout=2) as response:
                    if response.status == 200:
                        self.ready=True; return
            except Exception: pass
            if self.process is not None and self.process.poll() is not None: break
            time.sleep(.5)
        self.error='Local language model did not become ready. See /tmp/yard-language.log.'

    def interpret(self, text):
        if not self.ready: raise ValueError(self.error or 'Local language model is loading.')
        if not isinstance(text,str) or not text.strip() or len(text)>500:
            raise ValueError('Enter a command of at most500 characters.')
        body={'model':'Qwen3-0.6B','messages':[{'role':'system','content':PROMPT},
              {'role':'user','content':text}], 'temperature':0, 'max_tokens':180,
              'response_format':{'type':'json_schema','json_schema':{'name':'flight_plan',
                                  'strict':True,'schema':SCHEMA}}}
        with self._lock:
            try:
                request=Request(self.url+'/v1/chat/completions',data=json.dumps(body).encode(),
                                headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=65) as response: result=json.load(response)
                parsed=json.loads(result['choices'][0]['message']['content'])
            except Exception as error:
                raise ValueError(f'Command interpretation failed: {type(error).__name__}') from error
        actions=parsed.get('actions',[])
        if not 1<=len(actions)<=5: raise ValueError('The language model returned an invalid plan.')
        for action in actions:
            if action.get('action') not in ACTIONS: raise ValueError('The language model returned an unsupported action.')
            if action['action']=='goto' and action.get('zone') not in ['A','B','C','D','home']:
                raise ValueError('The requested destination is unclear.')
        return actions

    def status(self):
        return {'ready':self.ready,'model':'Qwen3-0.6B Q8_0','backend':'local llama.cpp', 'error':self.error}

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try: self.process.wait(timeout=3)
            except subprocess.TimeoutExpired: self.process.kill()
