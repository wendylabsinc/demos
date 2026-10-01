import copy
import math
import time
from go2_assistant.core import Assistant
from go2_assistant.memory import Memory

class Backend:
    def __init__(self):
        self.pose = {'x': 0., 'y': 0., 'yaw': 0.}
        self.state, self.stops, self.goals, self.detections = 'idle', 0, [], []
        self.map_id, self.timestamp, self.healthy, self.reject = 'test-map', None, True, False
    def observe(self):
        return {'map_id': self.map_id, 'timestamp': self.timestamp or time.monotonic(),
                'pose': copy.copy(self.pose), 'detections': self.detections,
                'healthy': self.healthy, 'reason': 'sensor fault'}
    def stop(self): self.stops += 1; self.state = 'idle'
    def navigate(self, x, y):
        if self.reject: return {'accepted': False, 'reason': 'hardware is disarmed'}
        self.goals.append((x, y)); self.state = 'navigating'
    def status(self): return {'state': self.state}
    def tick(self, dt): pass

def assistant():
    b = Backend(); a = Assistant(b, Memory(':memory:', 'test-map'))
    a.memory.remember_room('kitchen', 2., 2.)
    return a, b

def detection(label='red cup', track='cup1', x=2., y=2., age=0):
    return {'label': label, 'track_id': track, 'x': x, 'y': y,
            'confidence': .95, 'timestamp': time.monotonic() - age}

def test_room_persists_only_on_matching_map(tmp_path):
    path = tmp_path / 'memory.sqlite3'; m = Memory(path, 'map1')
    m.remember_room('Kitchen', 2., 3.); m.add_message('user', 'remember kitchen'); m.close()
    m = Memory(path, 'map1')
    assert m.room('kitchen')['x'] == 2 and m.history()[0]['content'] == 'remember kitchen'
    m.close(); m = Memory(path, 'map2')
    assert m.rooms() == [] and m.history() == []

def test_teaching_uses_measured_pose():
    a, b = assistant(); b.pose = {'x': 3., 'y': 4., 'yaw': .5}
    assert 'remember' in a.command('remember_room', {'room': 'office'})['reply']
    assert a.memory.room('office')['x'] == 3

def test_unknown_room_never_moves():
    a, b = assistant()
    assert "don't know" in a.command('go_to_room', {'room': 'garage'})['reply'] and not b.goals

def test_requires_confirmed_arrival():
    a, b = assistant(); a.command('go_to_room', {'room': 'kitchen'})
    b.state = 'idle'; a.tick(); assert a.task is not None
    b.state = 'arrived'; a.tick(); assert a.last_result['state'] == 'succeeded' and a.task is None

def test_rejected_goal_fails():
    a, b = assistant(); b.reject = True
    assert 'disarmed' in a.command('go_to_room', {'room': 'kitchen'})['reply']
    assert a.task is None and a.last_result['state'] == 'failed'

def test_replacement_and_stop():
    a, b = assistant(); a.command('go_to_room', {'room': 'kitchen'})
    first_id, stops = a.task['id'], b.stops
    a.command('find_object', {'object': 'red cup'})
    assert b.stops > stops and a.task['id'] != first_id
    a.stop(); assert not a.task and b.state == 'idle'

def test_stale_map_changed_unhealthy_stop():
    for fault in ('stale', 'changed', 'unhealthy'):
        a, b = assistant(); a.command('go_to_room', {'room': 'kitchen'})
        if fault == 'stale': b.timestamp = time.monotonic() - 5
        elif fault == 'changed': b.map_id = 'new-map'
        else: b.healthy = False
        a.tick(); assert not a.task and a.last_result['state'] == 'failed' and b.state == 'idle'

def test_historical_object_cannot_confirm_find():
    a, b = assistant(); a.memory.observe_object(detection())
    a.command('find_object', {'object': 'red cup'}); assert a.task and not a.last_result
    b.detections = [detection(age=5)]; a.tick(); assert a.task
    b.detections = [detection()]; a.tick()
    assert a.last_result['state'] == 'succeeded' and 'kitchen' in a.last_result['reply']

def test_ambiguous_detection_fails():
    a, b = assistant(); b.detections = [detection(track='a'), detection(track='b')]
    a.command('find_object', {'object': 'red cup'})
    assert a.last_result['state'] == 'failed' and not b.goals

def test_follow_never_switches_identity():
    a, b = assistant(); b.detections = [detection('person', 'owner', 3., 0.)]
    a.command('follow_person'); assert b.goals and a.task
    b.detections = [detection('person', 'stranger', 3., 0.)]; a.tick()
    assert a.last_result['state'] == 'target_lost' and not a.task

def test_follow_stops_at_distance():
    a, b = assistant(); b.detections = [detection('person', 'owner', 1., 0.)]
    a.command('follow_person'); assert not b.goals and a.task
    a.tick(); assert b.state == 'idle' and a.task

def test_come_here_finishes_at_person_then_stays():
    a, b = assistant(); b.detections = [detection('person', 'owner', 3., 0.)]
    result = a.command('come_here')
    assert 'Coming over' in result['reply'] and b.goals == [(1.7, 0.)]
    b.pose['x'] = 1.7; a.tick()
    assert a.task is None and a.last_result['state'] == 'succeeded'
    assert 'stay here' in a.last_result['reply']
    b.detections = [detection('person', 'owner', 4., 0.)]; a.tick()
    assert b.goals == [(1.7, 0.)] and b.state == 'idle'

def test_come_here_already_close_never_navigates():
    a, b = assistant(); b.detections = [detection('person', 'owner', 1., 0.)]
    result = a.command('come_here')
    assert result['result']['state'] == 'succeeded' and not b.goals

def test_come_here_accepts_measured_navigation_arrival_tolerance():
    a, b = assistant(); b.detections = [detection('person', 'owner', 3., 0.)]
    a.command('come_here')
    b.pose['x'] = 1.5; b.state = 'arrived'; a.tick()
    assert a.task is None and a.last_result['state'] == 'succeeded'

def test_come_here_replans_after_person_moves_before_arrival():
    a, b = assistant(); b.detections = [detection('person', 'owner', 3., 0.)]
    a.command('come_here')
    b.pose['x'] = 1.5; b.state = 'arrived'
    b.detections = [detection('person', 'owner', 3.3, 0.)]
    a._last_follow_at -= 2.; a.tick()
    assert a.task and len(b.goals) == 2 and math.dist(b.goals[-1], (2., 0.)) < 1e-8

def test_come_here_requires_current_selected_person():
    for people in ([], [detection('person', age=5)],
                   [detection('person', 'a'), detection('person', 'b')]):
        a, b = assistant(); b.detections = people
        a.command('come_here'); assert not b.goals and not a.task
    a, b = assistant(); b.detections = [detection('person', 'owner', 3., 0.)]
    a.command('come_here'); b.detections = [detection('person', 'other', 3., 0.)]
    a.tick(); assert a.last_result['state'] == 'target_lost' and b.state == 'idle'

def test_playful_action_waits_for_measured_completion():
    a, b = assistant(); calls = []
    def perform(action):
        calls.append(action); b.state = 'navigating'; return {'accepted': True}
    b.perform_action = perform
    result = a.command('perform_action', {'action': 'spin'})
    assert calls == ['spin'] and result['task']['state'] == 'running'
    b.state = 'idle'; a.tick(); assert a.task and a.last_result is None
    b.state = 'arrived'; a.tick()
    assert a.task is None and a.last_result['state'] == 'succeeded'
    assert 'One spin' in a.last_result['reply']

def test_unavailable_or_invalid_play_does_not_replace_task():
    a, b = assistant(); a.command('go_to_room', {'room': 'kitchen'})
    original_id = a.task['id']; stops = b.stops
    assert a.command('perform_action', {'action': 'spin'})['error']
    b.perform_action = lambda action: {'accepted': True}
    for action in ('jump', ['spin'], None):
        assert a.command('perform_action', {'action': action})['error']
    assert a.task['id'] == original_id and b.stops == stops

def test_rejected_playful_action_stops_and_fails():
    a, b = assistant()
    b.perform_action = lambda action: {'accepted': False, 'reason': 'stand up first'}
    result = a.command('perform_action', {'action': 'spin'})
    assert a.task is None and result['result']['state'] == 'failed'
    assert b.state == 'idle' and 'stand up first' in result['reply']

def test_stop_cancels_playful_action():
    a, b = assistant()
    b.perform_action = lambda action: setattr(b, 'state', 'navigating')
    a.command('perform_action', {'action': 'lie_down'})
    a.stop(); b.state = 'arrived'; a.tick()
    assert a.task is None and a.last_result['state'] == 'cancelled'

def test_stop_explains_a_posture_transition_still_settling():
    a, b = assistant()
    b.stop = lambda: {'accepted': True, 'settling': True}
    result = a.stop()
    assert result['result']['state'] == 'cancelled'
    assert 'posture is still settling safely' in result['reply']

def test_timeout_and_blocked_stop():
    for fault in ('timeout', 'blocked'):
        a, b = assistant(); a.command('go_to_room', {'room': 'kitchen'})
        if fault == 'timeout': a.task['started_at'] -= 200
        else: b.state = 'blocked'
        a.tick(); assert not a.task and b.state == 'idle'
        assert a.last_result['state'] in {'timed_out', 'failed'}

def test_stop_failure_never_claims_success():
    a, b = assistant()
    b.stop = lambda: {'accepted': False, 'reason': 'controller unreachable'}
    result = a.stop()
    assert a.last_result['state'] == 'stop_failed'
    assert 'could not confirm' in result['reply'] and not result['reply'].startswith('Stopped')

def test_every_entrypoint_requires_token_for_public_bind(monkeypatch):
    import pytest
    from go2_assistant.cli import parser, validate_args
    monkeypatch.setenv('GO2_HOST', '0.0.0.0')
    monkeypatch.delenv('GO2_HTTP_TOKEN', raising=False)
    p = parser()
    with pytest.raises(SystemExit): validate_args(p.parse_args(['serve']), p)
    monkeypatch.setenv('GO2_HTTP_TOKEN', 'test-token')
    validate_args(p.parse_args(['serve']), p)


def test_learned_observation_is_usable_only_at_its_capture_viewpoint():
    a, b = assistant()
    item = detection(age=3)
    item.update(source='vision:test-model', capture_pose=dict(b.pose))
    b.detections = [item]
    a._observe()
    assert a._fresh_detection(item)
    assert not a._fresh_detection(item, following=True)
    b.pose['x'] = .3
    a._observe()
    assert not a._fresh_detection(item)
    item['timestamp'] = time.monotonic() - .1
    assert not a._fresh_detection(item)
    assert not a._fresh_detection(item, following=True)


def test_room_names_require_current_image_evidence():
    a, b = assistant()
    room = {'name': 'study', 'confidence': .9, 'evidence': ['desk and bookshelves'],
            'capture_pose': dict(b.pose), 'timestamp': time.monotonic(), 'model': 'test-model'}
    original = b.observe
    b.observe = lambda: {**original(), 'room': room}
    a._observe()
    assert a.memory.room('study')['source'] == 'vision:test-model'
    room.update(name='garage', evidence=[])
    a._observe()
    assert a.memory.room('garage') is None
    room.update(name='bedroom', evidence=['bed'], timestamp=time.monotonic() - 20)
    a._observe()
    assert a.memory.room('bedroom') is None


def test_unknown_room_explores_sensor_frontiers_without_given_coordinates():
    a, b = assistant()
    b.sensor_mapping = True
    b.search_points = lambda: [(1., 0.)]
    result = a.command('go_to_room', {'room': 'study'})
    assert a.task['kind'] == 'find_room'
    assert b.goals == [(1., 0.)]
    assert 'recognized' in result['reply']
    a.memory.remember_room('study', .5, 0., source='vision:test-model')
    a.tick()
    assert a.task['kind'] == 'go_to_room' and b.goals[-1] == (.5, 0.)


def test_dynamic_frontier_search_replenishes_after_new_sensors_reveal_space():
    a, b = assistant()
    b.sensor_mapping = True
    frontiers = [(1., 0.)]
    b.search_points = lambda: list(frontiers)
    a.command('explore')
    frontiers[:] = [(2., 0.)]
    b.state = 'arrived'
    a.tick()
    assert b.goals == [(1., 0.), (2., 0.)]
    b.state = 'arrived'
    a.tick()
    assert a.task is None and a.last_result['state'] == 'succeeded'


def test_camera_failure_cannot_be_replaced_by_a_scripted_find():
    a, b = assistant()
    b.sensor_mapping = True
    b.status = lambda: {'state': b.state, 'vision': {'pending': False, 'error': 'model offline'}}
    result = a.command('find_object', {'object': 'cup'})
    assert a.task is None and a.last_result['state'] == 'perception_unavailable'
    assert 'unavailable' in result['reply'] and not b.goals


def test_fixed_object_vocabulary_does_not_wander_to_invent_a_room_name():
    a, b = assistant()
    b.sensor_mapping = True
    b.status = lambda: {'state': b.state, 'vision': {'vocabulary': 'fixed learned classes'}}
    result = a.command('go_to_room', {'room': 'study'})
    assert not a.task and not b.goals
    assert 'teach room names' in result['reply']
