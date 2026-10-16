import time
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from showcase.models import Observation
from showcase.ai import Decision
from showcase.server import Runtime, handler

class FakeTransport:
    profile='test'
    def __init__(self):self.armed=False;self.commands=[]
    def read(self):return Observation(fresh=True,scan_fresh=True,front_clearance=8,sectors=tuple((a,8) for a in (-3,-2,-1,0,1,2,3)))
    def send(self,c):self.commands.append(c)
    def arm(self):self.armed=True
    def stop(self):self.armed=False
    def status(self):return {'armed':self.armed}
    def camera_jpeg(self):return b'\xff\xd8test\xff\xd9'
    def frame_jpeg(self):return self.camera_jpeg()
    def close(self):pass

class Agent:
    model='test-only'
    def decide(self,*args):return Decision('forward',.2,'test model action',('empty test frame',))

def wait_for(predicate, timeout=2):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        if predicate():return
        time.sleep(.01)
    raise AssertionError('state transition timed out')

def test_model_action_required_and_stop_disarms():
    r=Runtime(FakeTransport,hz=50,agent_factory=Agent)
    try:
        assert r.status()['mission']['state']=='disabled'
        r.start({'instruction':'Explore'})
        wait_for(lambda:any(c.vx>0 for c in r.transport.commands))
        assert r.last_decision['source']=='model'
        r.stop();assert not r.transport.armed
        assert r.status()['command']['vx']==0
    finally:r.close()

def test_stop_discards_inflight_model_response():
    released=threading.Event()
    class SlowAgent(Agent):
        def decide(self,*args):released.wait(2);return super().decide(*args)
    r=Runtime(FakeTransport,hz=50,agent_factory=SlowAgent)
    try:
        r.start({'instruction':'Explore'})
        wait_for(lambda:r.pending is not None)
        r.stop();released.set();time.sleep(.1)
        assert not r.transport.armed
        assert not any(c.vx>0 for c in r.transport.commands)
    finally:released.set();r.close()

def test_camera_cache_busting_query_returns_image():
    r=Runtime(FakeTransport,agent_factory=Agent)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(r))
    t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    try:
        response=urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/camera.jpg?t=123')
        assert response.headers['Content-Type']=='image/jpeg'
        assert response.read().startswith(b'\xff\xd8')
    finally:server.shutdown();server.server_close();r.close()


# --- model readiness, sensor faults, idle stops and access token ---------------------------

import json
import urllib.error
from showcase.ai import ModelUnavailable
from showcase.server import ModelNotReady


class CountingTransport(FakeTransport):
    """Fake robot whose sensors can be made stale and whose stops are counted."""
    def __init__(self):
        super().__init__()
        self.stops = 0
        self.healthy = True
    def read(self):
        return Observation(fresh=self.healthy, scan_fresh=self.healthy, front_clearance=8,
                           sectors=tuple((a, 8) for a in (-3, -2, -1, 0, 1, 2, 3)))
    def stop(self):
        self.stops += 1
        super().stop()
    def status(self):
        return {'armed': self.armed, 'sensor_ages': {'pose': 0.1}}


class LoadingAgent(Agent):
    """Model whose first load is slow, like a cold Ollama."""
    def __init__(self):
        self.loaded_now = threading.Event()
        self.fail = False
    def warm_up(self):
        self.loaded_now.wait(5)
        if self.fail:
            raise ModelUnavailable('Model service is unreachable or timed out')


def serve(runtime, token=None):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(runtime, token))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f'http://127.0.0.1:{server.server_port}'


def call(url, method='GET', body=None, token=None):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_start_waits_for_the_model_instead_of_faulting_a_run():
    agent = LoadingAgent()
    r = Runtime(CountingTransport, hz=50, agent_factory=lambda: agent)
    server, base = serve(r)
    try:
        status = r.status()
        assert status['ai']['model_state']['state'] == 'loading'
        code, body = call(base + '/start', 'POST', {'instruction': 'Turn left'})
        assert code == 503 and json.loads(body)['model']['state'] == 'loading'
        try:
            r.start({'instruction': 'Turn left'})
            raise AssertionError('start must be refused while the model loads')
        except ModelNotReady:
            pass
        time.sleep(.2)  # the control loop keeps running; nothing was armed or faulted
        assert r.status()['mission']['state'] == 'disabled' and not r.transport.armed
        assert r.status()['error'] is None
        agent.loaded_now.set()
        wait_for(lambda: r.status()['ai']['model_state']['state'] == 'ready')
        code, body = call(base + '/start', 'POST', {'instruction': 'Turn left'})
        assert code == 200 and json.loads(body)['mission']['state'] in ('waiting', 'running')
    finally:
        agent.loaded_now.set()
        server.shutdown(); server.server_close(); r.close()


def test_model_error_is_shown_and_a_failed_decision_requires_a_working_model_again():
    agent = LoadingAgent()
    agent.fail = True
    agent.loaded_now.set()
    r = Runtime(CountingTransport, hz=50, agent_factory=lambda: agent)
    try:
        wait_for(lambda: r.status()['ai']['model_state']['state'] == 'error')
        assert 'unreachable' in r.status()['ai']['model_state']['detail']
        try:
            r.start({'instruction': 'Explore'})
            raise AssertionError('start must be refused while the model is unavailable')
        except ModelNotReady:
            pass
    finally:
        r.close()


def test_slow_model_call_with_healthy_sensors_never_faults_the_run():
    release = threading.Event()
    class SlowAgent(Agent):
        def decide(self, *args):
            release.wait(5)
            return super().decide(*args)
    r = Runtime(CountingTransport, hz=50, agent_factory=SlowAgent)
    try:
        r.start({'instruction': 'Explore'})
        wait_for(lambda: r.pending is not None)
        for _ in range(30):  # 1.5 s of control cycles with the model still thinking
            assert r.status()['mission']['state'] == 'waiting'
            assert r.status()['ai']['thinking']
            time.sleep(.05)
        assert r.status()['error'] is None and r.transport.armed
        release.set()
        wait_for(lambda: any(c.vx > 0 for c in r.transport.commands))
    finally:
        release.set(); r.close()


def test_sensors_really_going_stale_during_a_model_call_still_stop_the_robot(capsys):
    release = threading.Event()
    class SlowAgent(Agent):
        def decide(self, *args):
            release.wait(5)
            return super().decide(*args)
    r = Runtime(CountingTransport, hz=50, agent_factory=SlowAgent)
    try:
        r.start({'instruction': 'Explore'})
        wait_for(lambda: r.pending is not None)
        r.transport.healthy = False
        wait_for(lambda: r.status()['mission']['state'] == 'fault')
        assert 'stale' in r.status()['mission']['reason'] and not r.transport.armed
        release.set(); time.sleep(.1)
        assert not any(c.vx > 0 for c in r.transport.commands)
        fault = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"fault"' in line]
        assert fault and fault[0]['sensor_ages'] == {'pose': 0.1} and fault[0]['model_request_seconds'] is not None
    finally:
        release.set(); r.close()


def test_idle_robot_gets_one_stop_not_one_per_control_cycle():
    r = Runtime(CountingTransport, hz=100, agent_factory=Agent)
    try:
        time.sleep(.4)  # about 40 control cycles while disabled
        assert r.transport.stops <= 2
        stops = r.transport.stops
        r.start({'instruction': 'Explore'})
        assert r.transport.stops == stops + 1  # stop-on-start is kept
        r.stop()
        time.sleep(.1)
        settled = r.transport.stops  # the operator stop plus one at the end-of-run transition
        assert settled <= stops + 3
        time.sleep(.4)  # about 40 more control cycles while stopped
        assert r.transport.stops == settled
        r.start({'instruction': 'Explore'})
        r.transport.healthy = False
        wait_for(lambda: r.status()['mission']['state'] == 'fault')
        time.sleep(.1)
        settled = r.transport.stops
        time.sleep(.4)
        assert r.transport.stops == settled
    finally:
        r.close()


def test_token_protects_status_camera_and_motion_but_not_the_page():
    r = Runtime(CountingTransport, hz=50, agent_factory=Agent)
    server, base = serve(r, 'correct horse battery')
    try:
        assert call(base + '/')[0] == 200
        assert json.loads(call(base + '/config')[1]) == {'token_required': True}
        for route in ('/status', '/health', '/camera.jpg'):
            assert call(base + route)[0] == 401
            assert call(base + route, token='wrong')[0] == 401
            assert call(base + route, token='correct horse battery')[0] in (200, 404)
        stops = r.transport.stops
        for route, body in (('/start', {'instruction': 'Explore'}), ('/stop', {})):
            assert call(base + route, 'POST', body)[0] == 401
            assert call(base + route, 'POST', body, token='wrong')[0] == 401
        assert r.transport.stops == stops and r.status()['mission']['state'] == 'disabled'
        code, _ = call(base + '/start', 'POST', {'instruction': 'Explore'}, token='correct horse battery')
        assert code == 200 and r.status()['mission']['state'] != 'disabled'
        assert call(base + '/stop', 'POST', {}, token='correct horse battery')[0] == 200
        assert call(base + '/nope', token='correct horse battery')[0] == 404
    finally:
        server.shutdown(); server.server_close(); r.close()


def test_without_a_token_everything_stays_open(monkeypatch):
    monkeypatch.delenv('GO2_HTTP_TOKEN', raising=False)
    r = Runtime(CountingTransport, hz=50, agent_factory=Agent)
    server, base = serve(r)
    try:
        assert json.loads(call(base + '/config')[1]) == {'token_required': False}
        assert call(base + '/status')[0] == 200
        assert call(base + '/stop', 'POST', {})[0] == 200
    finally:
        server.shutdown(); server.server_close(); r.close()


def test_token_from_the_environment_is_enforced(monkeypatch):
    monkeypatch.setenv('GO2_HTTP_TOKEN', ' from-env ')
    r = Runtime(CountingTransport, hz=50, agent_factory=Agent)
    server, base = serve(r)
    try:
        assert call(base + '/status')[0] == 401
        assert call(base + '/status', token='from-env')[0] == 200
    finally:
        server.shutdown(); server.server_close(); r.close()
