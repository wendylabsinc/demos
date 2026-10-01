"""Shared task logic. The language model chooses skills, never actuator commands."""
from __future__ import annotations

import math
import threading
import time
import uuid
from .memory import clean_name


class Assistant:
    def __init__(self, backend, memory, *, clock=time.monotonic, max_age=.75,
                 task_timeout=180., follow_distance=1.3, perception_max_age=10.):
        self.backend, self.memory = backend, memory
        self.lock = threading.RLock()
        self.clock, self.max_age = clock, max_age
        self.task_timeout, self.follow_distance = task_timeout, follow_distance
        self.perception_max_age = perception_max_age
        self.task = None
        self.last_result = None
        self.last_observation = None
        self._last_follow_goal = None
        self._last_follow_at = -math.inf
        self._search_queue = []
        self._search_phase = None
        self._scan = False
        self._target_track = None
        self._visited_search_points = []
        self._search_views = 0
        self._view_started = self.clock()
        self.events = []

    def _observe(self):
        obs = self.backend.observe()
        if not isinstance(obs, dict) or not obs.get('healthy'):
            raise ValueError(obs.get('reason', 'Robot observations are unavailable.') if isinstance(obs, dict)
                             else 'Robot observations are unavailable.')
        if obs.get('map_id') != self.memory.map_id:
            raise ValueError("Localization map changed. Use room memory for the current map.")
        age = self.clock() - float(obs['timestamp'])
        if not 0 <= age <= self.max_age:
            raise ValueError("Robot observations are stale.")
        pose = obs.get('pose', {})
        if not all(isinstance(pose.get(k), (int, float)) and not isinstance(pose[k], bool)
                   and math.isfinite(pose[k]) for k in ('x', 'y', 'yaw')):
            raise ValueError("Robot localization is invalid.")
        self.last_observation = obs
        for detection in obs.get('detections', []):
            if self._fresh_detection(detection):
                self.memory.observe_object(detection)
        room = obs.get('room')
        if isinstance(room, dict) and self._fresh_visual_observation(room):
            evidence = room.get('evidence')
            if float(room.get('confidence', 0)) >= .8 and evidence and isinstance(evidence, (str, list)):
                name = clean_name(room['name'])
                if self.memory.room(name) is None:
                    self.memory.remember_room(name, **room['capture_pose'], source='vision:' + str(room.get('model', 'model')))
        return obs

    def _fresh_visual_observation(self, observation):
        """Slow image inference is usable only while its viewpoint still matches."""
        try:
            age = self.clock() - float(observation['timestamp'])
            captured = observation['capture_pose']
            current = self.last_observation['pose']
            values = [float(captured[k]) for k in ('x', 'y', 'yaw')]
            if not all(math.isfinite(v) for v in values):
                return False
            yaw_error = (values[2] - current['yaw'] + math.pi) % (2 * math.pi) - math.pi
            return (0 <= age <= self.perception_max_age
                    and math.hypot(values[0] - current['x'], values[1] - current['y']) <= .2
                    and abs(yaw_error) <= .15)
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def _fresh_detection(self, detection, *, following=False):
        try:
            age = self.clock() - float(detection['timestamp'])
            fresh = 0 <= age <= self.max_age
            if str(detection.get('source', '')).startswith('vision:'):
                fresh = self._fresh_visual_observation(detection) and (fresh if following else True)
            return (fresh and float(detection['confidence']) >= .65
                    and all(math.isfinite(float(detection[k])) for k in ('x', 'y', 'confidence'))
                    and bool(detection['track_id']) and bool(detection['label']))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def _start(self, kind, args):
        # A replacement always cancels and stops the previous owner first.
        self._accept(self.backend.stop())
        self.task = {'id': str(uuid.uuid4()), 'kind': kind, 'args': args,
                     'state': 'running', 'started_at': self.clock()}
        self._search_queue = []
        self._search_phase = None
        self._last_follow_goal = None
        self._last_follow_at = -math.inf
        self._target_track = None
        self._scan = False
        self._visited_search_points = []
        self._search_views = 0
        self._view_started = self.clock()
        self._event('started', kind=kind, args=args)

    @staticmethod
    def _accept(result):
        if isinstance(result, dict) and result.get('accepted') is False:
            raise RuntimeError(result.get('reason', 'The robot rejected the request.'))

    def _navigate(self, x, y):
        self._accept(self.backend.navigate(x, y))

    def _event(self, state, **data):
        event = {'task_id': self.task['id'] if self.task else None, 'state': state, **data}
        self.events.append(event)
        self.events = self.events[-100:]
        self.memory.event(event)

    def _finish(self, state, reply):
        try:
            result = self.backend.stop()
            if isinstance(result, dict) and result.get('accepted') is False:
                state = 'stop_failed'
                reply = 'I could not confirm that the robot stopped. ' + result.get('reason', 'Robot unavailable.')
            elif isinstance(result, dict) and result.get('settling') is True:
                reply += ' My posture is still settling safely.'
        except (ValueError, PermissionError, RuntimeError, OSError) as error:
            state = 'stop_failed'
            reply = f'I could not confirm that the robot stopped. {error}'
        if self.task and self.task['kind'] == 'find_object' and hasattr(self.backend, 'set_search_target'):
            self.backend.set_search_target(None)
        if self.task:
            self.task['state'] = state
            self.last_result = {**self.task, 'reply': reply}
        else:
            self.last_result = {'state': state, 'reply': reply}
        self._event(state, reply=reply)
        self.task = None
        return {'reply': reply, 'result': self.last_result}

    def stop(self):
        with self.lock:
            return self._finish('cancelled', "Stopped.")

    def command(self, name, args=None):
        args = args or {}
        with self.lock:
            if name == 'stop':
                return self.stop()
            if name == 'status':
                s = self.status()
                if self.task:
                    reply = f"I'm working on {self.task['kind'].replace('_', ' ')}."
                elif self.last_result:
                    reply = self.last_result['reply']
                else:
                    reply = "Ready when you are. Ask me to follow you, find something, or do a little spin."
                return {'reply': reply, 'status': s}
            try:
                obs = self._observe()
                if name == 'remember_room':
                    room = clean_name(args['room'])
                    self.memory.remember_room(room, **obs['pose'])
                    return {'reply': f"I'll remember this location as {room}."}
                if name == 'go_to_room':
                    room_name = clean_name(args['room'])
                    room = self.memory.room(room_name)
                    if room is None:
                        vision = self._vision_status() or {}
                        if vision.get('vocabulary') == 'fixed learned classes':
                            return {'reply': f"I don't know {room_name} yet. Show me the room and say 'remember this as {room_name}'. My current vision models recognize objects; you teach room names."}
                        if getattr(self.backend, 'sensor_mapping', False):
                            self._start('find_room', {'room': room_name})
                            self._begin_search(obs)
                            return {'reply': f"I haven't recognized {room_name} yet. I'll explore the observed free space and look for it.", 'task': dict(self.task) if self.task else None}
                        return {'reply': f"I don't know {room_name} yet. Take me there and say 'remember this as {room_name}'."}
                    self._start(name, {'room': room_name})
                    self._navigate(room['x'], room['y'])
                    return {'reply': f"I'm going to {room_name}.", 'task': dict(self.task)}
                if name == 'find_object':
                    label = clean_name(args['object'])
                    room = clean_name(args['room']) if args.get('room') else None
                    if room and self.memory.room(room) is None:
                        return {'reply': f"I don't know {room} yet."}
                    if label in {'me', 'person', 'someone'}:
                        return {'reply': "Say 'follow me' to follow the person you're tracking."}
                    self._start(name, {'object': label, 'room': room})
                    if hasattr(self.backend, 'set_search_target'):
                        self.backend.set_search_target(label)
                    self._search_queue = self._search_points(label, room)
                    self._search_phase = 'observe' if self._vision_status() else 'choose'
                    self._tick_search(obs)
                    if self.task:
                        return {'reply': f"I'm looking for {label}" + (f" in {room}." if room else "."),
                                'task': dict(self.task)}
                    return {'reply': self.last_result['reply'], 'result': self.last_result}
                if name in {'follow_person', 'come_here'}:
                    people = [d for d in obs.get('detections', []) if d.get('label') == 'person'
                              and self._fresh_detection(d, following=True)]
                    requested = args.get('track_id')
                    if requested:
                        people = [d for d in people if d['track_id'] == requested]
                    if len(people) != 1:
                        reply = "I can't see you. Step into my camera view and try again."
                        if len(people) > 1:
                            reply = "I can see more than one person. Select a tracked person first."
                        return {'reply': reply}
                    self._start(name, {'track_id': people[0]['track_id']})
                    self._target_track = people[0]['track_id']
                    self._tick_follow(obs)
                    if not self.task:
                        return {'reply': self.last_result['reply'], 'result': self.last_result}
                    reply = ("Coming over! I'll stop a little way from you." if name == 'come_here'
                             else "On your heels! I'll give you some space. Say stop whenever you like.")
                    return {'reply': reply,
                            'task': dict(self.task) if self.task else None}
                if name == 'perform_action':
                    action = args.get('action')
                    replies = {'spin': 'One spin, coming up!',
                               'lie_down': "Getting comfy. I'm lowering into a rest pose.",
                               'stand_up': 'Up I go!'}
                    if not isinstance(action, str) or action not in replies:
                        return {'reply': 'Try spin, lie down, or stand up.', 'error': True}
                    if not hasattr(self.backend, 'perform_action'):
                        return {'reply': 'This robot connection does not support that move yet.', 'error': True}
                    self._start(name, {'action': action})
                    self._accept(self.backend.perform_action(action))
                    return {'reply': replies[action], 'task': dict(self.task)}
                if name == 'explore':
                    if not getattr(self.backend, 'sensor_mapping', False):
                        return {'reply': 'This connection does not expose sensor mapping.', 'error': True}
                    self._start(name, {})
                    self._begin_search(obs)
                    return {'reply': "I'll explore the space my lidar can reach and build a map. Say stop whenever you like.", 'task': dict(self.task) if self.task else None}
                return {'reply': f"Unknown skill: {name}.", 'error': True}
            except (ValueError, KeyError, PermissionError, RuntimeError, OSError) as error:
                return self._finish('failed', str(error))

    def _search_points(self, label, room):
        points = []
        # Last-seen positions are hints. A fresh camera observation must confirm a find.
        for obj in self.memory.objects():
            if obj['label'] == label and (room is None or obj['room'] == room):
                # Go to the associated safe room waypoint, never the object itself.
                remembered = self.memory.room(obj['room']) if obj.get('room') else None
                if remembered:
                    points.append((remembered['x'], remembered['y']))
        if room is not None and self.memory.room(room):
            remembered = self.memory.room(room)
            points.append((remembered['x'], remembered['y']))
        elif hasattr(self.backend, 'search_points'):
            points.extend(self.backend.search_points())
        else:
            points.extend((r['x'], r['y']) for r in self.memory.rooms()
                          if room is None or r['name'] == room)
        unique = []
        for point in points:
            point = tuple(float(v) for v in point)
            if len(point) == 2 and all(math.isfinite(v) for v in point) and point not in unique:
                unique.append(point)
        return unique

    def _vision_status(self):
        vision = self.backend.status().get('vision')
        return vision if isinstance(vision, dict) else None

    def _begin_search(self, obs):
        self._search_queue = self._search_points('', None)
        self._search_phase = 'observe' if self._vision_status() else 'choose'
        self._tick_search(obs)

    def _view_ready(self):
        vision = self._vision_status()
        if not vision:
            return True
        if vision.get('available') is not True:
            if vision.get('error') and not vision.get('pending'):
                self._finish('perception_unavailable', 'The vision model is unavailable, so I stopped. ' + str(vision['error']))
            elif self.clock() - self._view_started > 40:
                self._finish('perception_timeout', 'The camera model did not return a current observation, so I stopped.')
            return False
        if vision.get('error') and not vision.get('pending'):
            self._finish('perception_unavailable', 'The vision model is unavailable, so I stopped. ' + str(vision['error']))
            return False
        if self.clock() - self._view_started > 40:
            self._finish('perception_timeout', 'The camera model did not return a current observation, so I stopped.')
            return False
        capture = vision.get('capture_pose')
        captured_at = vision.get('captured_at')
        return bool(capture and captured_at is not None and self._fresh_visual_observation({'capture_pose': capture, 'timestamp': captured_at}))

    def _tick_search(self, obs):
        kind = self.task['kind']
        label = self.task['args'].get('object', '')
        room = self.task['args'].get('room')
        if kind == 'find_room' and self.memory.room(room):
            remembered = self.memory.room(room)
            self.task['kind'] = 'go_to_room'
            self._navigate(remembered['x'], remembered['y'])
            return
        matches = [d for d in obs.get('detections', []) if d.get('label') == label
                   and self._fresh_detection(d)
                   and (room is None or self.memory.nearest_room(d['x'], d['y']) == room)]
        if len(matches) > 1:
            self._finish('failed', f"I can see more than one {label}. Please specify which one.")
            return
        if matches:
            d = matches[0]
            location = self.memory.nearest_room(d['x'], d['y'])
            where = f" in {location}" if location else ""
            self._finish('succeeded', f"I found {label}{where}. Its map position is {d['x']:.1f}, {d['y']:.1f} metres.")
            self.last_result['object'] = dict(d)
            return
        state = self.backend.status()['state']
        if self._search_phase == 'navigate' and state == 'arrived':
            self.backend.stop()
            self._search_views = 0
            self._view_started = self.clock()
            self._search_phase = 'observe'
        elif self._search_phase == 'scan' and state in {'idle', 'arrived'}:
            self.backend.stop()
            self._view_started = self.clock()
            self._search_phase = 'observe'
        if self._search_phase == 'observe':
            if kind != 'explore' and not self._view_ready():
                return
            if self._search_views < 8 and hasattr(self.backend, 'scan'):
                self._search_views += 1
                self.backend.scan()
                self._search_phase = 'scan'
                return
            self._search_phase = 'choose'
        if self._search_phase == 'choose':
            if getattr(self.backend, 'sensor_mapping', False):
                # Scans can reveal or close routes. Never retain an old set of
                # frontier destinations across changes to the measured map.
                self._search_queue = [point for point in self._search_points(label, room)
                                      if not any(math.dist(point, old) < .6 for old in self._visited_search_points)]
            if not self._search_queue:
                points = self._search_points(label, room)
                self._search_queue = [point for point in points if not any(math.dist(point, old) < .6 for old in self._visited_search_points)]
            if not self._search_queue:
                if kind == 'explore':
                    self._finish('succeeded', 'I finished exploring the reachable frontiers in my sensor map.')
                elif kind == 'find_room':
                    self._finish('not_found', f"I haven't recognized {room} in the places I could safely inspect.")
                else:
                    self._finish('not_found', f"I didn't find {label} in the places I checked.")
                return
            x, y = self._search_queue.pop(0)
            self._visited_search_points.append((x, y))
            self._navigate(x, y)
            self._search_phase = 'navigate'

    def _tick_follow(self, obs):
        matches = [d for d in obs.get('detections', []) if d.get('label') == 'person'
                   and d.get('track_id') == self._target_track and self._fresh_detection(d, following=True)]
        if len(matches) != 1:
            self._finish('target_lost', "I've lost sight of you, so I stopped. Ask me to follow again when I can see you.")
            return
        person, pose = matches[0], obs['pose']
        dx, dy = person['x'] - pose['x'], person['y'] - pose['y']
        distance = math.hypot(dx, dy)
        now = self.clock()
        arrived = self.backend.status()['state'] == 'arrived'
        # Nav2/the simulator may finish a little short of the standoff goal.
        # A one-time approach must complete from that measured position, or
        # replan if the person moved while the previous goal was in flight.
        close = distance <= self.follow_distance + .15
        if self.task['kind'] == 'come_here' and arrived:
            close = distance <= self.follow_distance + .25
            if not close:
                self._last_follow_goal = None
        if close:
            if self.task['kind'] == 'come_here':
                self._finish('succeeded', "Here I am! I'll stay here.")
                return
            self._accept(self.backend.stop())
            self._last_follow_goal = None
            return
        goal = (person['x'] - dx / distance * self.follow_distance,
                person['y'] - dy / distance * self.follow_distance)
        # Do not churn Nav2 actions every frame. Every replan cancels ownership first.
        changed = self._last_follow_goal is None or math.dist(goal, self._last_follow_goal) > .45
        if changed and now - self._last_follow_at >= 1.5:
            self._navigate(*goal)
            self._last_follow_goal, self._last_follow_at = goal, now

    def tick(self, dt=.1):
        with self.lock:
            try:
                self.backend.tick(dt)
                obs = self._observe()
                if not self.task:
                    return
                if self.clock() - self.task['started_at'] > self.task_timeout:
                    self._finish('timed_out', "The task took too long, so I stopped.")
                    return
                state = self.backend.status()['state']
                if state in {'blocked', 'fault'}:
                    self._finish('failed', "I couldn't complete a safe route, so I stopped.")
                    return
                kind = self.task['kind']
                if kind == 'go_to_room' and state == 'arrived':
                    self._finish('succeeded', f"I've arrived in {self.task['args']['room']}.")
                elif kind in {'find_object', 'find_room', 'explore'}:
                    self._tick_search(obs)
                elif kind in {'follow_person', 'come_here'}:
                    self._tick_follow(obs)
                elif kind == 'perform_action' and state == 'arrived':
                    replies = {'spin': 'Ta-da! One spin.',
                               'lie_down': 'Settled in my rest pose.',
                               'stand_up': 'Standing and ready!'}
                    self._finish('succeeded', replies[self.task['args']['action']])
            except (ValueError, KeyError, PermissionError, RuntimeError, OSError) as error:
                if self.task:
                    self._finish('failed', str(error))
                else:
                    self.backend.stop()
                    self.last_result = {'state': 'observation_fault', 'reply': str(error)}

    def status(self):
        with self.lock:
            return {'map_id': self.memory.map_id, 'backend': self.backend.status(),
                    'task': dict(self.task) if self.task else None,
                    'last_result': self.last_result,
                    'rooms': self.memory.rooms(), 'objects': self.memory.objects(),
                    'pose': self.last_observation['pose'] if self.last_observation else None,
                    'events': self.events[-10:]}
