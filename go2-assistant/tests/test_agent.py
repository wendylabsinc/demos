from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from go2_assistant.agent import OpenAICompatibleProvider, ProviderConfig, RobotAgent, tool_definitions, validate_call
from go2_assistant.memory import Memory
from go2_assistant.server import create_app


class FakeAssistant:
    def __init__(self):
        self.memory = Memory(":memory:", "test-room")
        self.lock = threading.RLock()
        self.calls = []
        self.ticks = 0
        self.views = []
        self.backend = self

    def command(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "remember_room":
            self.memory.remember_room(arguments["room"], 1, 2)
        return {"reply": "Stopped." if name == "stop" else name + " accepted."}

    def status(self):
        return {"task": self.calls[-1][0] if self.calls else "idle", "ticks": self.ticks}

    def frame(self, view):
        self.views.append(view)
        return Image.new("RGB", (24, 16), "green")

    def tick(self, dt):
        self.ticks += 1


class FakeProvider:
    config = SimpleNamespace(timeout=1, transcription_model="test-transcribe")

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, messages, tools, *, timeout):
        self.requests.append((messages.copy(), tools, timeout))
        return next(self.responses)

    def transcribe(self, audio, filename, content_type):
        assert audio == b"recorded-audio"
        return "go to kitchen"


def call_response(name, arguments, *, call_id="call_1"):
    call = {"id": call_id, "name": name, "arguments": json.dumps(arguments)}
    return {"content": "", "calls": [call], "message": {"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": call["arguments"]}}]}}


def final_response(text):
    return {"content": text, "calls": [], "message": {"role": "assistant", "content": text}}


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    monkeypatch.delenv("GO2_LLM_MODEL", raising=False)
    monkeypatch.delenv("GO2_HTTP_TOKEN", raising=False)


@pytest.mark.parametrize("message,name,args", [
    ("Please go to the kitchen", "go_to_room", {"room": "kitchen"}),
    ("Remember this room as kitchen", "remember_room", {"room": "kitchen"}),
    ("Find the red mug", "find_object", {"object": "red mug"}),
    ("Find the red mug in the kitchen", "find_object", {"object": "red mug", "room": "kitchen"}),
    ("Follow me", "follow_person", {}),
    ("Come here", "come_here", {}),
    ("Hey Go2, please come to me!", "come_here", {}),
    ("Come over here", "come_here", {}),
    ("Come with me", "follow_person", {}),
    ("Spin", "perform_action", {"action": "spin"}),
    ("Could you please turn around?", "perform_action", {"action": "spin"}),
    ("Lie down", "perform_action", {"action": "lie_down"}),
    ("Rest", "perform_action", {"action": "lie_down"}),
    ("Sit", "perform_action", {"action": "lie_down"}),
    ("Sit down", "perform_action", {"action": "lie_down"}),
    ("Stand up", "perform_action", {"action": "stand_up"}),
    ("Get up", "perform_action", {"action": "stand_up"}),
    ("Where are you?", "status", {}),
    ("Stop following me", "stop", {}),
])
def test_offline_commands(message, name, args):
    assistant = FakeAssistant()
    result = RobotAgent(assistant).respond(message)
    assert result["mode"] == "offline"
    assert assistant.calls == [(name, args)]
    assert assistant.memory.history()[-1]["role"] == "assistant"


def test_offline_unknown_text_does_not_issue_motion():
    assistant = FakeAssistant()
    result = RobotAgent(assistant).respond("Tell me a story")
    assert "Offline command mode" in result["reply"]
    assert assistant.calls == []


@pytest.mark.parametrize("message", ["Hello!", "Hey Go2!", "Good dog", "Good boy", "Greet", "Say hello"])
@pytest.mark.parametrize("llm", [False, True])
def test_greeting_and_praise_never_start_motion_or_call_provider(message, llm):
    assistant = FakeAssistant()
    provider = FakeProvider([]) if llm else None
    agent = RobotAgent(assistant, provider)
    result = agent.respond(message)
    assert assistant.calls == []
    assert result["tools"] == []
    assert result["reply"] in {"Hi! Ready when you are.", "Thanks! Happy to help."}
    assert assistant.memory.history()[-1]["content"] == result["reply"]
    if provider:
        assert provider.requests == []


def test_greeting_does_not_replace_a_running_task():
    assistant = FakeAssistant()
    agent = RobotAgent(assistant)
    agent.respond("Follow me")
    result = agent.respond("Good dog")
    assert assistant.calls == [("follow_person", {})]
    assert result["status"]["task"] == "follow_person"


def test_bounded_action_schema_never_accepts_motor_parameters():
    definitions = {tool["function"]["name"]: tool["function"] for tool in tool_definitions()}
    assert definitions["come_here"]["parameters"]["properties"] == {}
    action = definitions["perform_action"]["parameters"]
    assert action["properties"]["action"]["enum"] == ["spin", "lie_down", "stand_up"]
    assert action["required"] == ["action"]
    assert action["additionalProperties"] is False
    for name, args in [("come_here", {}), ("perform_action", {"action": "spin"}),
                       ("perform_action", {"action": "lie_down"}),
                       ("perform_action", {"action": "stand_up"})]:
        assert validate_call(name, args) == args


@pytest.mark.parametrize("message", ["Spin twice", "Spin forever", "Set velocity to 2", "Come here at 4 metres per second"])
def test_offline_unbounded_or_parameterized_requests_do_not_move(message):
    assistant = FakeAssistant()
    result = RobotAgent(assistant).respond(message)
    assert result["tools"] == []
    assert assistant.calls == []


def test_llm_tools_receive_results_and_persist_dialogue():
    assistant = FakeAssistant()
    provider = FakeProvider([call_response("go_to_room", {"room": "kitchen"}), final_response("I'm on my way to the kitchen.")])
    result = RobotAgent(assistant, provider).respond("Could you check the kitchen for me?")
    assert assistant.calls == [("go_to_room", {"room": "kitchen"})]
    assert result["mode"] == "llm"
    assert provider.requests[1][0][-1]["role"] == "tool"
    assert json.loads(provider.requests[1][0][-1]["content"])["reply"] == "go_to_room accepted."
    assert assistant.memory.history()[-1]["content"] == result["reply"]


@pytest.mark.parametrize("name,args", [
    ("execute_code", {}), ("go_to_room", {"room": "kitchen", "velocity": 100}),
    ("go_to_room", {"room": 2}), ("find_object", {"object": ""}),
    ("follow_person", {"track_id": "invented"}), ("come_here", {"x": 1, "y": 2}),
    ("come_here", {"track_id": "invented"}), ("perform_action", {}),
    ("perform_action", {"action": "dance"}), ("perform_action", {"action": "sit"}),
    ("perform_action", {"action": "spin", "degrees": 720}),
    ("perform_action", {"action": "spin", "duration": 60}),
    ("perform_action", {"action": "spin", "speed": 5}),
    ("perform_action", {"action": None}), ("perform_action", {"action": True}),
    ("perform_action", {"action": "spin\n"}),
])
def test_untrusted_provider_tools_do_not_reach_robot(name, args):
    assistant = FakeAssistant()
    result = RobotAgent(assistant, FakeProvider([call_response(name, args)])).respond("Move")
    assert assistant.calls == []
    assert result["tools"] == []


def test_tool_limit_rejects_batch_before_execution():
    assistant = FakeAssistant()
    response = call_response("status", {})
    response["calls"] *= 5
    result = RobotAgent(assistant, FakeProvider([response])).respond("Keep moving")
    assert "limit" in result["reply"]
    assert assistant.calls == []


@pytest.mark.parametrize("name,args", [("go_to_room", {"room": "kitchen"}),
                                       ("come_here", {}),
                                       ("perform_action", {"action": "spin"})])
def test_stop_bypasses_provider_and_cancels_pending_tool_call(name, args):
    assistant = FakeAssistant()
    entered, release = threading.Event(), threading.Event()

    class SlowProvider(FakeProvider):
        def complete(self, messages, tools, *, timeout):
            entered.set()
            assert release.wait(3)
            return call_response(name, args)

    agent = RobotAgent(assistant, SlowProvider([]), lock=assistant.lock)
    with TestClient(create_app(assistant, agent=agent, tick_interval=0)) as client:
        with ThreadPoolExecutor() as pool:
            request = pool.submit(client.post, "/api/chat", json={"message": "Go to kitchen"})
            assert entered.wait(2)
            result = client.post("/api/stop")
            assert result.status_code == 200
            assert assistant.calls == [("stop", {})]
            release.set()
            assert "cancelled" in request.result(3).json()["reply"]
            assert assistant.calls == [("stop", {})]


@pytest.mark.parametrize("message", ["Please stop", "Stop spinning", "Hey Go2, please stop now", "Stay"])
def test_text_stop_bypasses_language_model(message):
    assistant = FakeAssistant()
    provider = FakeProvider([])
    response = RobotAgent(assistant, provider).respond(message)
    assert response["reply"] == "Stopped."
    assert provider.requests == []
    assert assistant.calls == [("stop", {})]


@pytest.mark.parametrize("name,args", [("come_here", {}),
                                       ("perform_action", {"action": "spin"}),
                                       ("perform_action", {"action": "lie_down"})])
def test_playful_llm_tools_receive_controller_reply_without_claiming_completion(name, args):
    assistant = FakeAssistant()
    provider = FakeProvider([call_response(name, args), final_response("I'm starting now.")])
    result = RobotAgent(assistant, provider).respond("Can you help me?")
    assert assistant.calls == [(name, args)]
    assert result["tools"][0]["result"]["reply"] == name + " accepted."
    instructions = provider.requests[0][0][0]["content"]
    assert "Unitree Go2 robot dog" in instructions and "Wendy" in instructions
    assert "not that it has completed" in instructions
    assert "Do not claim a distinct hindquarters sitting pose" in instructions
    assert json.loads(provider.requests[1][0][-1]["content"])["reply"] == name + " accepted."


def test_action_rejection_is_passed_to_llm_without_fabricated_success():
    assistant = FakeAssistant()
    assistant.command = lambda name, args: {"reply": "This backend does not support that posture.",
                                            "result": {"state": "failed"}}
    provider = FakeProvider([call_response("perform_action", {"action": "lie_down"}),
                             final_response("That posture isn't available here.")])
    result = RobotAgent(assistant, provider).respond("Lie down")
    assert result["tools"][0]["result"]["result"]["state"] == "failed"
    assert "isn't available" in result["reply"]
    assert json.loads(provider.requests[1][0][-1]["content"])["result"]["state"] == "failed"


def test_phone_server_chat_auth_frames_and_history():
    assistant = FakeAssistant()
    with TestClient(create_app(assistant, token="test-token", tick_interval=0)) as client:
        assert client.get("/").status_code == 200
        assert "Wendy Go2" in client.get("/").text
        assert client.get("/api/config").json()["mode"] == "offline"
        assert client.post("/api/stop").status_code == 401
        auth = {"Authorization": "Bearer test-token"}
        assert client.post("/api/chat", headers=auth, json={"message": "Remember this as kitchen"}).status_code == 200
        status = client.get("/api/status", headers=auth).json()
        assert status["rooms"][0]["name"] == "kitchen"
        assert len(status["history"]) == 2
        picture = client.get("/api/frame/camera", headers=auth)
        assert picture.status_code == 200 and picture.headers["content-type"] == "image/jpeg"
        assert picture.content[:2] == b"\xff\xd8"
        assert assistant.views == ["robot"]
        assert client.get("/api/frame/arbitrary", headers=auth).status_code == 404
        assert client.post("/api/chat", headers=auth, json={"message": "x" * 2001}).status_code == 422
        assert client.post("/api/stop", headers={**auth, "Origin": "https://other.example"}).status_code == 403
        assert client.post("/api/stop", headers={**auth, "Origin": "http://testserver"}).status_code == 200


def test_server_runs_live_ticks_and_stops_on_shutdown():
    assistant = FakeAssistant()
    with TestClient(create_app(assistant, tick_interval=.01)) as client:
        end = time.monotonic() + 1
        while assistant.ticks == 0 and time.monotonic() < end:
            time.sleep(.01)
        assert client.get("/api/status").json()["status"]["ticks"] > 0
    assert assistant.calls[-1] == ("stop", {})


class FakePersonBackend:
    is_simulator = True

    def __init__(self):
        self.person = {"x": 2.0, "y": -1.3}
        self.moves = []
        self.frame_thread = None
        self.move_thread = None

    def status(self):
        return {"person": self.person.copy()}

    def set_person(self, x, y):
        if x > 3:
            raise ValueError("person marker must be inside free floor space")
        self.moves.append((x, y))
        self.person = {"x": x, "y": y}
        self.move_thread = threading.get_ident()

    def frame(self, view):
        self.frame_thread = threading.get_ident()
        return Image.new("RGB", (24, 16), "green")


def test_simulator_person_controls_validate_auth_floorplan_and_renderer_thread():
    assistant = FakeAssistant()
    backend = assistant.backend = FakePersonBackend()
    with TestClient(create_app(assistant, token="operator-token", tick_interval=0)) as client:
        assert client.get("/api/config").json()["simulator"] is True
        assert client.post("/api/sim/person", json={"x": 2.35, "y": -1.3}).status_code == 401
        assert backend.moves == []
        auth = {"Authorization": "Bearer operator-token"}
        assert client.get("/api/frame/overview", headers=auth).status_code == 200
        response = client.post("/api/sim/person", headers=auth, json={"x": 2.35, "y": -1.3})
        assert response.status_code == 200
        assert response.json()["person"] == {"x": 2.35, "y": -1.3}
        assert client.get("/api/status", headers=auth).json()["simulator_person"] == response.json()["person"]
        assert backend.move_thread == backend.frame_thread
        assert assistant.calls == []  # Moving the test person never commands the robot.
        assert client.post("/api/sim/person", headers=auth, json={"x": 4, "y": 0}).status_code == 422
        for position in ({"x": True, "y": 0}, {"x": "2", "y": 0}, {"x": 0, "y": 0, "z": 1}, {"x": 100, "y": 0}):
            assert client.post("/api/sim/person", headers=auth, json=position).status_code == 422
        assert client.post("/api/sim/person", headers={**auth, "Origin": "https://another.example"}, json={"x": 2, "y": -1}).status_code == 403
        assert backend.moves == [(2.35, -1.3)]


def test_person_controls_are_denied_for_physical_backend_even_if_method_exists():
    assistant = FakeAssistant()
    backend = assistant.backend = FakePersonBackend()
    backend.is_simulator = False
    with TestClient(create_app(assistant, tick_interval=0)) as client:
        assert client.get("/api/config").json()["simulator"] is False
        assert client.get("/api/status").json()["simulator_person"] is None
        assert client.post("/api/sim/person", json={"x": 2, "y": -1}).status_code == 403
        assert backend.moves == []


def test_uploaded_audio_returns_transcript_without_executing_command():
    assistant = FakeAssistant()
    agent = RobotAgent(assistant, FakeProvider([]))
    with TestClient(create_app(assistant, agent=agent, tick_interval=0)) as client:
        response = client.post("/api/transcribe", headers={"Content-Type": "audio/webm"}, content=b"recorded-audio")
        assert response.json() == {"text": "go to kitchen"}
        assert assistant.calls == []
        assert client.post("/api/transcribe", headers={"Content-Type": "text/plain"}, content=b"audio").status_code == 415


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed to execute the phone UI script")
@pytest.mark.parametrize("page_url,prefix", [("https://robot.example/", "/"), ("https://robot.example/out/", "/out/")])
def test_phone_ui_fetches_from_its_sandbox_prefix(page_url, prefix):
    html = (Path(__file__).parents[1] / "go2_assistant/static/index.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const vm = require('node:vm');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const requests = [];
const element = () => ({style:{},classList:{add(){},remove(){},toggle(){}},addEventListener(){},append(){},replaceChildren(){},value:'',checked:false});
const nodes = new Map();
const context = {
  URL, Blob, console, AbortController, DOMException, performance:{now:()=>0}, Image:class{decode(){return Promise.resolve();}removeAttribute(){}},setTimeout(){return 1;},clearTimeout(){},clearInterval(){},
  window:{location:{href:input.url},isSecureContext:false,addEventListener(){}}, navigator:{language:'en-US'},
  sessionStorage:{getItem(){return '';},setItem(){}}, setInterval(){},
  document:{hidden:false,addEventListener(){},getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},querySelectorAll(){return [];},createElement:element},
  fetch:async url => {
    requests.push(url);
    return {ok:true,status:200,json:async () => url.endsWith('api/config') ? {mode:'offline',token_required:false,transcription:false} : {status:{task:null},rooms:[],objects:[],history:[],mode:'offline'},blob:async () => new Blob(['image'])};
  },
};
vm.runInNewContext(input.script,context);
setImmediate(() => process.stdout.write(JSON.stringify(requests)));
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", harness], input=json.dumps({"script": script, "url": page_url}),
        capture_output=True, text=True, check=True, timeout=5,
    )
    assert json.loads(result.stdout) == [
        "https://robot.example" + prefix + "api/config",
        "https://robot.example" + prefix + "api/status",
        "https://robot.example" + prefix + "api/frame/overview",
    ]


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed to execute the phone UI script")
def test_phone_ui_person_controls_and_task_completion_feedback():
    html = (Path(__file__).parents[1] / "go2_assistant/static/index.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const vm=require('node:vm'),input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const nodes=new Map(),requests=[],spoken=[];
let person={x:2,y:-1.3},completed={id:'history',state:'succeeded',reply:'Historical result.'};
const element=()=>({style:{},classList:{add(){},remove(){},toggle(){}},children:[],events:{},value:'',checked:false,addEventListener(event,handler){this.events[event]=handler;},append(...items){this.children.push(...items);},replaceChildren(...items){this.children=items;}});
const speaker={cancel(){},speak(utterance){spoken.push(utterance.text);}};
const context=vm.createContext({
  URL,Blob,console,AbortController,DOMException,performance:{now:()=>0},Image:class{decode(){return Promise.resolve();}removeAttribute(){}},setTimeout(){return 1;},clearTimeout(){},clearInterval(){},window:{location:{href:'https://robot.example/out/'},isSecureContext:false,speechSynthesis:speaker,addEventListener(){}},
  speechSynthesis:speaker,SpeechSynthesisUtterance:function(text){this.text=text;},navigator:{language:'en-US'},
  sessionStorage:{getItem(){return '';},setItem(){}},setInterval(){},
  document:{hidden:false,addEventListener(){},getElementById(id){if(!nodes.has(id))nodes.set(id,element());return nodes.get(id);},querySelectorAll(){return [];},createElement:element},
  fetch:async(url,options={})=>{
    requests.push({url,body:options.body});
    let body={status:{task:null,last_result:completed},simulator_person:person,rooms:[],objects:[],history:[],mode:'offline'};
    if(url.endsWith('api/config'))body={mode:'offline',simulator:true,token_required:false,transcription:false};
    if(url.endsWith('api/sim/person')){person=JSON.parse(options.body);body={person};}
    if(url.endsWith('api/chat'))body={reply:completed.reply,status:{last_result:completed},tools:[]};
    if(url.endsWith('api/stop'))body={reply:'Stopped.',status:{last_result:completed}};
    return{ok:true,json:async()=>body,blob:async()=>new Blob(['image'])};
  },
});
vm.runInContext(input.script,context);
const settle=()=>new Promise(resolve=>setImmediate(resolve));
const replies=()=>nodes.get('messages').children.map(item=>item.children[1].textContent);
(async()=>{
  await settle();context.document.getElementById('speak').checked=true;
  const historical=replies().filter(text=>text==='Historical result.').length;
  await vm.runInContext('movePerson(0.35,0,false)',context);await settle();
  const move=requests.find(request=>request.url.endsWith('api/sim/person'));
  completed={id:'spin-1',state:'succeeded',reply:'One spin, coming right up. Done!'};
  await vm.runInContext('refreshStatus()',context);await vm.runInContext('refreshStatus()',context);
  completed={id:'rest-1',state:'succeeded',reply:'All comfy.'};
  await vm.runInContext("sendMessage('Lie down')",context);await settle();await vm.runInContext('refreshStatus()',context);
  completed={id:'follow-1',state:'cancelled',reply:'Stopped.'};
  await nodes.get('stop').events.click();await settle();await vm.runInContext('refreshStatus()',context);
  process.stdout.write(JSON.stringify({historical,move,person,spoken,replies:replies()}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", harness], input=json.dumps({"script": script}),
        capture_output=True, text=True, check=True, timeout=5,
    )
    output = json.loads(result.stdout)
    assert output["historical"] == 0
    assert output["move"]["url"] == "https://robot.example/out/api/sim/person"
    assert json.loads(output["move"]["body"]) == {"x": 2.35, "y": -1.3}
    assert output["person"] == {"x": 2.35, "y": -1.3}
    assert output["spoken"] == ["One spin, coming right up. Done!", "All comfy."]
    assert output["replies"].count("One spin, coming right up. Done!") == 1
    assert output["replies"].count("All comfy.") == 1
    assert output["replies"].count("Stopped.") == 1


@pytest.mark.skipif(not shutil.which("node"), reason="Node is needed to execute the phone UI script")
@pytest.mark.parametrize("scenario", ["pace", "view", "visibility", "errors", "auth", "timeout", "hung_decode", "decode_timeout"])
def test_phone_frame_scheduler(scenario):
    html = (Path(__file__).parents[1] / "go2_assistant/static/index.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const vm=require('node:vm'),assert=require('node:assert/strict');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const nodes=new Map(),timers=new Map(),intervals=new Map(),documentEvents={},windowEvents={};
const frames=[],decodes=[],displayed=[],created=[],revoked=[];
let now=0,nextTimer=1,activeFetch=0,activeDecode=0;
function element(id=''){
  const item={style:{},classList:{add(){},remove(){},toggle(){}},events:{},value:'',checked:false,
    addEventListener(event,handler){this.events[event]=handler;},append(){},replaceChildren(){},removeAttribute(){this.source='';}};
  Object.defineProperty(item,'src',{set(value){this.source=value;if(id==='frame')displayed.push(value);},get(){return this.source;}});
  return item;
}
const views=['overview','camera'].map(view=>({...element(),dataset:{view}}));
class FrameImage{
  decode(){
    assert.equal(activeFetch,0,'decode must follow the completed fetch');assert.equal(activeDecode,0,'decodes must not overlap');
    activeDecode++;
    return new Promise((resolve,reject)=>{this.job={image:this,url:this.src,done:false,resolve,reject};decodes.push(this.job);});
  }
  removeAttribute(){if(this.job && !this.job.done){this.job.done=true;activeDecode--;if(!['hung_decode','decode_timeout'].includes(input.scenario))this.job.reject(new DOMException('Cancelled','AbortError'));}}
}
class FrameURL extends URL{
  static createObjectURL(){const url='blob:frame-'+(created.length+1);created.push(url);return url;}
  static revokeObjectURL(url){revoked.push(url);}
}
const context=vm.createContext({
  URL:FrameURL,Blob,AbortController,DOMException,Image:FrameImage,console,performance:{now:()=>now},
  window:{location:{href:'https://robot.example/out/'},isSecureContext:false,addEventListener(event,handler){windowEvents[event]=handler;}},navigator:{language:'en-US'},
  document:{hidden:false,addEventListener(event,handler){documentEvents[event]=handler;},getElementById(id){if(!nodes.has(id))nodes.set(id,element(id));return nodes.get(id);},querySelectorAll(selector){return selector==='[data-view]' ? views : [];},createElement:element},
  sessionStorage:{getItem(){return '';},setItem(){}},
  setTimeout(callback,delay){const id=nextTimer++;timers.set(id,{callback,at:now+delay});return id;},clearTimeout(id){timers.delete(id);},
  setInterval(callback,delay){const id=nextTimer++;intervals.set(id,{callback,delay});return id;},clearInterval(id){intervals.delete(id);},
  fetch:async(url,options={})=>{
    if(!url.includes('/api/frame/'))return{ok:true,json:async()=>url.endsWith('api/config') ? {mode:'offline',frame_rate:100,token_required:input.scenario==='auth'} : {status:{task:null},rooms:[],objects:[],history:[],mode:'offline'}};
    assert.equal(activeFetch,0,'frame requests must not overlap');assert.equal(activeDecode,0,'fetch must wait for decode');activeFetch++;
    return new Promise((resolve,reject)=>{
      const frame={url,options,at:now,done:false,resolve,reject};frames.push(frame);
      options.signal.addEventListener('abort',()=>{if(!frame.done){frame.done=true;activeFetch--;reject(new DOMException('Cancelled','AbortError'));}},{once:true});
    });
  },
});
const settle=()=>new Promise(resolve=>setImmediate(resolve));
async function finishFetch(index,status=200){const frame=frames[index];assert(!frame.done);frame.done=true;activeFetch--;frame.resolve({ok:status===200,status,json:async()=>({detail:'Camera unavailable'}),blob:async()=>new Blob(['image'])});await settle();}
async function finishDecode(index){const job=decodes[index];assert(!job.done);job.done=true;activeDecode--;job.resolve();await settle();}
async function advance(ms){now+=ms;for(const [id,timer]of [...timers].sort((a,b)=>a[1].at-b[1].at)){if(timer.at<=now && timers.has(id)){timers.delete(id);timer.callback();}}await settle();}
async function visibility(hidden){context.document.hidden=hidden;documentEvents.visibilitychange();await settle();}
async function token(value){context.document.getElementById('token').value=value;nodes.get('token-box').events.submit({preventDefault(){}});await settle();}
vm.runInContext(input.script,context);
(async()=>{
  await settle();
  assert.equal(intervals.size,1,'one status interval starts');
  if(input.scenario==='auth'){
    assert.equal(frames.length,0,'startup waits for the required token');
    await token('wrong');assert.equal(frames.length,1);await finishFetch(0,401);
    await advance(5000);assert.equal(frames.length,1,'bad tokens must not cause a retry loop');
    await token('valid');assert.equal(frames.length,2);assert.equal(frames[1].options.headers.Authorization,'Bearer valid');
    await finishFetch(1);await finishDecode(0);assert.equal(displayed.length,1);
  }else{
    assert.equal(frames.length,1,'startup issues one frame request');
    assert(frames[0].url.endsWith('/out/api/frame/overview'));
    if(input.scenario==='pace'){
      assert.equal(displayed.length,0);await finishFetch(0);assert.equal(displayed.length,0,'decode must finish before swap');
      await advance(49);assert.equal(frames.length,1,'a pending decode prevents another fetch');
      await finishDecode(0);assert.equal(displayed.length,1);await advance(0);assert.equal(frames.length,1);
      await advance(1);assert.equal(frames.length,2);assert.equal(frames[1].at,50,'configured rates above20 are capped');
      await finishFetch(1);assert.equal(displayed.length,1);await finishDecode(1);assert.equal(displayed.length,2);assert(revoked.includes(displayed[0]));
      windowEvents.pagehide();await settle();assert.equal(timers.size,0);assert.equal(intervals.size,0);assert(revoked.includes(displayed[1]));
      await advance(5000);assert.equal(frames.length,2);windowEvents.pageshow();await settle();assert.equal(frames.length,3);assert.equal(intervals.size,1);
    }
    if(input.scenario==='view'){
      views[1].events.click();await settle();assert(frames[0].options.signal.aborted);assert.equal(frames.length,2);assert(frames[1].url.endsWith('/camera'));
      await finishFetch(1);assert.equal(decodes.length,1);views[0].events.click();await settle();
      assert.equal(activeDecode,0);assert.equal(displayed.length,0,'obsolete decoded views must not swap');assert.equal(frames.length,3);assert(revoked.includes(created[0]));
      await finishFetch(2);await finishDecode(1);assert.equal(displayed.length,1);assert(frames[2].url.endsWith('/overview'));
    }
    if(input.scenario==='visibility'){
      await visibility(true);assert.equal(activeFetch,0);await advance(1000);assert.equal(frames.length,1);
      await visibility(false);assert.equal(frames.length,2);await finishFetch(1);await visibility(true);
      assert.equal(activeDecode,0);assert.equal(displayed.length,0);assert(revoked.includes(created[0]));await advance(1000);assert.equal(frames.length,2);
      await visibility(false);assert.equal(frames.length,3);await finishFetch(2);await finishDecode(1);assert.equal(displayed.length,1);
    }
    if(input.scenario==='errors'){
      await finishFetch(0,503);await advance(249);assert.equal(frames.length,1);await advance(1);assert.equal(frames.length,2);
      await finishFetch(1,503);await advance(499);assert.equal(frames.length,2);await advance(1);assert.equal(frames.length,3);
      await finishFetch(2);await finishDecode(0);await advance(49);assert.equal(frames.length,3);await advance(1);assert.equal(frames.length,4,'success restores normal pacing');
    }
    if(input.scenario==='timeout'){
      await advance(4000);assert(frames[0].options.signal.aborted);assert.equal(activeFetch,0);assert(nodes.get('view-placeholder').textContent.includes('timed out'));
      await advance(249);assert.equal(frames.length,1);await advance(1);assert.equal(frames.length,2);
    }
    if(input.scenario==='hung_decode'){
      // Some browsers leave decode() pending after src removal. Cancellation
      // must release the frame loop independently of that decoder promise.
      await finishFetch(0);views[1].events.click();await settle();
      assert.equal(frames.length,2,'a canceled hung decoder cannot block the selected view');
      assert.equal(displayed.length,0);assert(revoked.includes(created[0]));
      await finishFetch(1);await finishDecode(1);assert.equal(displayed.length,1);assert(frames[1].url.endsWith('/camera'));
    }
    if(input.scenario==='decode_timeout'){
      await finishFetch(0);await advance(4000);
      assert.equal(activeDecode,0);assert(nodes.get('view-placeholder').textContent.includes('timed out'));
      await advance(250);assert.equal(frames.length,2,'a hung decode timeout must retry');
      await finishFetch(1);await finishDecode(1);assert.equal(displayed.length,1);
    }
  }
  windowEvents.pagehide();await settle();assert.equal(activeFetch,0);assert.equal(activeDecode,0);assert.equal(timers.size,0);assert.equal(intervals.size,0);
  assert(created.every(url=>revoked.includes(url)),'all frame URLs are released at cleanup');
  process.stdout.write(JSON.stringify({scenario:input.scenario,frames:frames.length}));
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", harness], input=json.dumps({"script": script, "scenario": scenario}),
        capture_output=True, text=True, check=True, timeout=5,
    )
    assert json.loads(result.stdout)["scenario"] == scenario


@pytest.mark.parametrize("api,path", [("chat", "/chat/completions"), ("responses", "/responses")])
def test_provider_http_payloads_and_tool_results(api, path):
    provider = OpenAICompatibleProvider(ProviderConfig("http://localhost:1234/v1", "test-model", api=api))
    sent = []

    def post(request_path, data, content_type, timeout):
        sent.append((request_path, json.loads(data)))
        if api == "chat":
            return {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call1", "function": {"name": "find_object", "arguments": '{"object":"mug","room":null}'}}]}}]}
        return {"output": [{"type": "function_call", "call_id": "call1", "name": "find_object", "arguments": '{"object":"mug","room":null}'}]}

    provider._post = post
    from go2_assistant.agent import tool_definitions
    result = provider.complete([{"role": "system", "content": "System"}, {"role": "user", "content": "Find mug"}], tool_definitions(), timeout=1)
    assert sent[0][0] == path
    assert sent[0][1]["parallel_tool_calls"] is False
    assert validate_call(result["calls"][0]["name"], result["calls"][0]["arguments"]) == {"object": "mug"}
    messages = [
        {"role": "system", "content": "System"}, {"role": "user", "content": "Find mug"},
        result["message"], {"role": "tool", "tool_call_id": "call1", "content": '{"reply":"Searching."}'},
    ]
    provider.complete(messages, tool_definitions(), timeout=1)
    if api == "responses":
        assert sent[1][1]["input"][-1] == {"type": "function_call_output", "call_id": "call1", "output": '{"reply":"Searching."}'}
        assert sent[1][1]["input"][-2]["type"] == "function_call"
    else:
        assert sent[1][1]["messages"][-1]["role"] == "tool"
