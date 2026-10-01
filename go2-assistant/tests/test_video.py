"""Video requests share short-lived frames without bypassing access or freshness."""
import threading

from fastapi.testclient import TestClient
from PIL import Image

from go2_assistant.memory import Memory
from go2_assistant.server import create_app


class VideoAssistant:
    def __init__(self):
        self.lock = threading.RLock()
        self.memory = Memory(':memory:', 'video-test')
        self.backend = self
        self.views = []
        self.fail = False
        self.is_simulator = True
        self.person = {'x': 2., 'y': -1.3}

    def command(self, name, args):
        return {'reply': 'Stopped.'}

    def status(self):
        return {'person': dict(self.person), 'task': None}

    def set_person(self, x, y):
        self.person = {'x': x, 'y': y}

    def frame(self, view):
        self.views.append(view)
        if self.fail:
            raise RuntimeError('camera unavailable')
        return Image.new('RGB', (24, 16), 'green' if len(self.views) % 2 else 'red')


def test_video_reuses_recent_frames_then_produces_fresh_frames():
    a = VideoAssistant(); now = [1.]
    with TestClient(create_app(a, token='', tick_interval=0, frame_clock=lambda: now[0])) as client:
        assert client.get('/api/config').json()['frame_rate'] == 20
        first = client.get('/api/frame/overview')
        same = client.get('/api/frame/overview')
        assert first.content == same.content
        assert first.headers['x-frame-sequence'] == same.headers['x-frame-sequence']
        assert a.views == ['overview']
        camera = client.get('/api/frame/camera')
        assert camera.headers['x-frame-sequence'] != first.headers['x-frame-sequence']
        assert a.views == ['overview', 'robot']
        now[0] += .051
        fresh = client.get('/api/frame/overview')
        assert fresh.headers['x-frame-sequence'] != first.headers['x-frame-sequence']
        assert fresh.headers['cache-control'] == 'no-store'
        assert a.views == ['overview', 'robot', 'overview']


def test_cached_video_still_requires_token_and_same_origin():
    a = VideoAssistant()
    with TestClient(create_app(a, token='test-token', tick_interval=0)) as client:
        auth = {'Authorization': 'Bearer test-token'}
        assert client.get('/api/frame/overview', headers=auth).status_code == 200
        assert client.get('/api/frame/overview').status_code == 401
        assert client.get('/api/frame/overview', headers={**auth, 'Origin': 'https://elsewhere.example'}).status_code == 403
        assert a.views == ['overview']


def test_expired_camera_failure_is_reported_and_can_recover():
    a = VideoAssistant(); now = [1.]
    with TestClient(create_app(a, token='', tick_interval=0, frame_clock=lambda: now[0])) as client:
        assert client.get('/api/frame/camera').status_code == 200
        now[0] += .051; a.fail = True
        assert client.get('/api/frame/camera').status_code == 503
        a.fail = False
        assert client.get('/api/frame/camera').status_code == 200
        assert len(a.views) == 3


def test_person_move_invalidates_the_rendered_frame_immediately():
    a = VideoAssistant()
    with TestClient(create_app(a, token='', tick_interval=0, frame_clock=lambda: 1.)) as client:
        first = client.get('/api/frame/overview')
        assert client.post('/api/sim/person', json={'x': 2.35, 'y': -1.3}).status_code == 200
        second = client.get('/api/frame/overview')
        assert first.headers['x-frame-sequence'] != second.headers['x-frame-sequence']
        assert len(a.views) == 2


def test_new_physics_state_is_never_hidden_by_the_frame_interval():
    a = VideoAssistant(); a.elapsed = 1.
    with TestClient(create_app(a, token='', tick_interval=0, frame_clock=lambda: 1.)) as client:
        first = client.get('/api/frame/camera')
        a.elapsed += .05
        next_frame = client.get('/api/frame/camera')
        assert first.headers['x-frame-sequence'] != next_frame.headers['x-frame-sequence']
        assert len(a.views) == 2


def test_sensor_map_is_authenticated_and_drawn_without_a_scene_renderer():
    a = VideoAssistant()
    cells = [[-1] * 20 for _ in range(20)]
    for y in range(7, 13):
        for x in range(7, 13):
            cells[y][x] = 0
    cells[10][13] = 100
    a.map_snapshot = lambda: {'cells': cells, 'origin': [-1., -1.], 'resolution_m': .1,
                              'pose': {'x': 0., 'y': 0., 'yaw': 0.}}
    with TestClient(create_app(a, token='map-token', tick_interval=0)) as client:
        auth = {'Authorization': 'Bearer map-token'}
        assert client.get('/api/map').status_code == 401
        assert client.get('/api/map', headers=auth).json()['cells'][0][0] == -1
        assert client.get('/api/frame/map', headers=auth).status_code == 200
        assert a.views == []
        assert client.get('/api/frame/map').status_code == 401
