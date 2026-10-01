"""Navigation timestamps stay honest across slow camera work, without GL."""
import threading
from types import SimpleNamespace

import pytest

from go2_assistant.core import Assistant
from go2_assistant.memory import Memory
from go2_assistant import sim as sim_module
from go2_assistant.sim import SimBackend, TIMESTEP


def timed_backend(*, acquire=True):
    backend = SimBackend.__new__(SimBackend)
    now, events = [10.], []
    backend.lock = threading.RLock()
    backend.elapsed = .2
    backend.last_capture = backend._last_scan = 0.
    backend.sensor_fault = False
    backend.action = None
    backend.scan_remaining = 0.
    backend.sim = SimpleNamespace(command=(0., 0., 0.))
    backend._sensor_time = now[0]
    backend.capture_time = now[0]
    backend._velocity = lambda: (0., 0., 0.)
    backend._check_action = lambda: None

    def step(velocity):
        backend.elapsed += TIMESTEP

    def capture():
        events.append('camera')
        # A real exposure keeps its own timestamp even if downstream work is slow.
        backend.capture_time = now[0]
        now[0] += 3.
        backend.last_capture = backend.elapsed

    def lidar():
        events.append('lidar')
        if acquire:
            backend._sensor_time = now[0]
            backend._last_scan = backend.elapsed

    backend._step = step
    backend._capture = capture
    backend._sample_lidar = lidar
    backend.observe = lambda: {'map_id': 'timing', 'timestamp': backend._sensor_time,
                              'pose': {'x': 0., 'y': 0., 'yaw': 0.},
                              'detections': [], 'healthy': True}
    return backend, now, events


def test_slow_camera_work_precedes_actual_navigation_measurement():
    backend, now, events = timed_backend()
    backend.tick(.05)
    assert events == ['camera', 'lidar']
    assert backend._sensor_time == now[0] == 13.
    assert backend.capture_time == 10.
    memory = Memory(':memory:', 'timing')
    try:
        assistant = Assistant(backend, memory, clock=lambda: now[0])
        assert assistant._observe()['timestamp'] == 13.
    finally:
        memory.close()


def test_no_new_measurement_cannot_forge_a_fresh_navigation_timestamp():
    backend, now, events = timed_backend(acquire=False)
    backend.tick(.05)
    assert events == ['camera', 'lidar']
    assert backend._sensor_time == 10. and now[0] == 13.
    memory = Memory(':memory:', 'timing')
    try:
        assistant = Assistant(backend, memory, clock=lambda: now[0])
        with pytest.raises(ValueError, match='observations are stale'):
            assistant._observe()
    finally:
        memory.close()


def test_wall_age_triggers_real_scan_when_physics_scan_interval_is_not_due(monkeypatch):
    backend, now, events = timed_backend()
    backend._last_scan = backend.elapsed
    monkeypatch.setattr(sim_module.time, 'monotonic', lambda: now[0])
    backend.tick(.05)
    assert backend.elapsed - .2 < .1
    assert events == ['camera', 'lidar']
    assert backend._sensor_time == now[0] == 13.
