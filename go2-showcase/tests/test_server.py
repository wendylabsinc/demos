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
