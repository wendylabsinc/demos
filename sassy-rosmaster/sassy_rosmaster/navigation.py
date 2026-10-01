"""Lidar-only roaming and an odometry-aligned occupancy map.

The planner samples bounded Ackermann arcs, checks the entire circular footprint,
and prefers visible places it has not visited. It never reads the simulator scene.
"""
import math
import random
import time
from collections import Counter

from .types import Command, Observation


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def validate_observation(obs, now=None):
    now = time.monotonic() if now is None else now
    if not all(math.isfinite(v) for v in (obs.pose.x, obs.pose.y, obs.pose.yaw,
                                         obs.angle_min, obs.angle_increment, obs.range_max, obs.captured_at)):
        raise ValueError("Invalid sensor geometry")
    if not 0 <= now - obs.captured_at <= .35:
        raise ValueError("Sensor data is stale")
    if len(obs.ranges) < 90 or obs.angle_increment <= 0 or obs.range_max < 1:
        raise ValueError("A full lidar scan is required")
    if not 6.15 <= len(obs.ranges) * obs.angle_increment <= 6.42:
        raise ValueError("Lidar must cover 360 degrees")
    # The adapters normalize positive infinity (valid no-return) to range_max.
    if any(not math.isfinite(v) or v <= 0 or v > obs.range_max + .01 for v in obs.ranges):
        raise ValueError("Lidar contains invalid ranges")


class OccupancyMap:
    def __init__(self, width=70, height=60, resolution=.2, origin=(-7., -6.)):
        self.width, self.height = width, height
        self.resolution, self.origin = resolution, origin
        self.odds = [0] * (width * height)
        self.seen = bytearray(width * height)
        self.last_update = -1.

    def index(self, x, y):
        cx = math.floor((x - self.origin[0]) / self.resolution)
        cy = math.floor((y - self.origin[1]) / self.resolution)
        if 0 <= cx < self.width and 0 <= cy < self.height:
            return cy * self.width + cx
        return None

    def update(self, obs):
        if obs.captured_at - self.last_update < .18:
            return
        self.last_update = obs.captured_at
        for i, distance in enumerate(obs.ranges):
            angle = obs.pose.yaw + obs.angle_min + i * obs.angle_increment
            c, s = math.cos(angle), math.sin(angle)
            previous = None
            for n in range(max(0, int((distance - .1) / (self.resolution / 2)))):
                reach = n * self.resolution / 2
                idx = self.index(obs.pose.x + reach*c, obs.pose.y + reach*s)
                if idx is not None and idx != previous:
                    self.seen[idx] = 1
                    self.odds[idx] = max(-12, self.odds[idx] - 1)
                    previous = idx
            if distance < obs.range_max - .03:
                px, py = obs.pose.x + distance*c, obs.pose.y + distance*s
                # Moving people stay out of the static map. All their rays still
                # participate in immediate collision avoidance.
                person_hit = any(math.hypot(px-p['x'], py-p['y']) < .5
                                 for p in obs.people if 'x' in p and 'y' in p)
                idx = self.index(px, py)
                if idx is not None and not person_hit:
                    self.seen[idx] = 1
                    self.odds[idx] = min(12, self.odds[idx] + 4)

    def export(self):
        return {"width": self.width, "height": self.height,
                "resolution": self.resolution, "origin": list(self.origin),
                "cells": [-1 if not seen else 100 if odds > 0 else 0
                          for seen, odds in zip(self.seen, self.odds)],
                "coverage": round(100 * sum(self.seen) / len(self.seen), 1)}


class Navigator:
    def __init__(self, max_speed=.45, wheelbase=.32, body_radius=.38, max_steering=.55, seed=None):
        self.max_speed, self.wheelbase, self.body_radius = max_speed, wheelbase, body_radius
        # The default .30m hard circle contains the .273m R2 footprint plus
        # clearance. Hardware adds .12m to its measured circle, so subtracting
        # .08m here still leaves .04m beyond its physical footprint.
        self.hard_body_radius = max(0., body_radius - .08)
        self.recovery_clearance = body_radius + .12
        self.recovering = False
        self.yielding = False
        self.yielded_for_person = False
        self.max_steering = max_steering
        self.rng = random.Random(seed)
        self.visits = Counter()
        self.goal = None
        self.goal_age = 0.
        self.last_cell = None
        self.last_steering = 0.
        self.previous_pose = None
        self.motion_speed = self.motion_steering = 0.
        self.points = []
        self.status = "Parked, judging everyone"
        self.map = OccupancyMap()

    def observe(self, obs, dt):
        self.map.update(obs)
        x, y = obs.pose.x, obs.pose.y
        if self.previous_pose is not None and dt > 0:
            previous = self.previous_pose
            self.motion_speed = ((x-previous.x)*math.cos(previous.yaw)
                                 + (y-previous.y)*math.sin(previous.yaw)) / dt
            yaw_rate = wrap(obs.pose.yaw-previous.yaw) / dt
            self.motion_steering = (math.atan(self.wheelbase*yaw_rate/self.motion_speed)
                                    if abs(self.motion_speed) > .015 else 0.)
            self.motion_steering = max(-self.max_steering, min(self.max_steering, self.motion_steering))
        self.previous_pose = obs.pose
        cell = math.floor(x / .4), math.floor(y / .4)
        if cell != self.last_cell:
            self.visits[cell] += 1
            self.last_cell = cell
        self.goal_age += dt
        self.points = []
        for i, distance in enumerate(obs.ranges):
            if distance >= obs.range_max - .03:
                continue
            angle = obs.pose.yaw + obs.angle_min + i * obs.angle_increment
            self.points.append((x+distance*math.cos(angle), y+distance*math.sin(angle)))

    def novelty(self, x, y):
        gx, gy = math.floor(x/.4), math.floor(y/.4)
        count = sum(self.visits[(gx+dx, gy+dy)] for dx in range(-1, 2) for dy in range(-1, 2))
        return 1. / (1. + count)

    def goal_candidates(self, obs):
        x, y, yaw = obs.pose.x, obs.pose.y, obs.pose.yaw
        choices = []
        for i in range(0, len(obs.ranges), max(1, len(obs.ranges)//24)):
            distance = obs.ranges[i]
            angle = yaw + obs.angle_min + i*obs.angle_increment
            for reach in (1.2, 2.2, 3.2):
                if reach > distance - self.body_radius:
                    continue
                gx, gy = x+reach*math.cos(angle), y+reach*math.sin(angle)
                clearance = min((math.hypot(gx-px, gy-py) for px, py in self.points), default=3.)
                if clearance < self.body_radius + .12:
                    continue
                # A clear endpoint on a neighboring beam can still put the
                # car's footprint through a person or the edge of furniture.
                blocked = False
                for px, py in self.points:
                    along = (px-x)*math.cos(angle) + (py-y)*math.sin(angle)
                    if 0 < along < reach:
                        lateral = abs(-(px-x)*math.sin(angle) + (py-y)*math.cos(angle))
                        if lateral < self.body_radius + .12:
                            blocked = True
                            break
                if blocked:
                    continue
                score = 2.8*self.novelty(gx, gy) + .4*min(clearance, 2.) + .12*reach
                score += .65*math.cos(wrap(angle-yaw)) + self.rng.uniform(0, .16)
                choices.append(dict(x=gx, y=gy, score=score, novelty=self.novelty(gx, gy),
                                    clearance=clearance, bearing=wrap(angle-yaw), distance=reach))
        return sorted(choices, key=lambda item: item['score'], reverse=True)

    def choose_goal(self, obs):
        choices = self.goal_candidates(obs)
        self.goal = (choices[0]['x'], choices[0]['y']) if choices else None
        self.goal_age = 0.

    def accept_goal(self, candidate, obs):
        """Recheck a model-selected sensor candidate against the current scan."""
        if self.yielding or self.recovering:
            return False
        gx, gy = candidate['x'], candidate['y']
        dx, dy = gx-obs.pose.x, gy-obs.pose.y
        reach = math.hypot(dx, dy)
        if not .6 <= reach <= min(3.8, obs.range_max-self.body_radius):
            return False
        angle = math.atan2(dy, dx)
        relative = (wrap(angle-obs.pose.yaw)-obs.angle_min) / obs.angle_increment
        beam = round(relative) % len(obs.ranges)
        if obs.ranges[beam] < reach+self.body_radius:
            return False
        c, s = math.cos(angle), math.sin(angle)
        for px, py in self.points:
            if math.hypot(gx-px, gy-py) < self.body_radius+.12:
                return False
            along = (px-obs.pose.x)*c + (py-obs.pose.y)*s
            lateral = abs(-(px-obs.pose.x)*s+(py-obs.pose.y)*c)
            if 0 < along < reach and lateral < self.body_radius+.12:
                return False
        self.goal, self.goal_age = (gx, gy), 0.
        return True

    def _arc(self, obs, speed, steering, steps, include_slew=True):
        """Predict steering slew and braking using odometry's current motion."""
        x, y, heading = obs.pose.x, obs.pose.y, obs.pose.yaw
        velocity, wheel_angle = self.motion_speed, self.motion_steering
        for _ in range(steps):
            if include_slew:
                target = 0. if velocity*speed < 0 else speed
                # Slow braking is conservative relative to the simulator's 1.6m/s².
                velocity += max(-.08, min(.08, target-velocity))
                wheel_angle += max(-.15, min(.15, steering-wheel_angle))
            else:
                velocity, wheel_angle = speed, steering
            heading += velocity*math.tan(wheel_angle)/self.wheelbase*.1
            x += velocity*math.cos(heading)*.1
            y += velocity*math.sin(heading)*.1
            yield x, y, heading

    def _retreat(self, obs, nearby, clearance):
        """Leave an inflated margin only along a reverse arc that opens space."""
        if clearance <= self.hard_body_radius:
            self.status = "Obstacle too close. Brakes on."
            return Command()
        speed = -min(.14, self.max_speed)
        horizon = 2.4
        rear_ranges = [distance for i, distance in enumerate(obs.ranges)
                       if abs(wrap(obs.angle_min + i*obs.angle_increment - math.pi)) <= math.pi/4]
        if not rear_ranges or min(rear_ranges) < self.body_radius + abs(speed)*horizon + .05:
            self.status = "No clear retreat. Brakes on."
            return Command()
        choices = []
        for ratio in (-.5, -.25, 0., .25, .5):
            steering = ratio * self.max_steering
            previous, safe = clearance, True
            for x, y, _ in self._arc(obs, speed, steering, 24):
                current = min((math.hypot(x-px, y-py) for px, py in nearby), default=3.)
                if current < previous - 1e-6 or current <= self.hard_body_radius:
                    safe = False
                    break
                previous = current
            if safe and previous >= self.recovery_clearance:
                choices.append((previous - .03*abs(steering), steering))
        if not choices:
            self.status = "No clear retreat. Brakes on."
            return Command()
        _, steering = max(choices)
        self.goal = None
        self.last_steering = steering
        self.status = "Backing away. Personal space, please."
        return Command(speed, steering)

    def _back_goal(self, obs):
        rear_ranges = [distance for i, distance in enumerate(obs.ranges)
                       if abs(wrap(obs.angle_min + i*obs.angle_increment - math.pi)) <= math.pi/4]
        available = min(rear_ranges, default=0.) - self.body_radius - .15
        if available < .6:
            return False
        reach = min(1.5, available)
        self.goal = (obs.pose.x-reach*math.cos(obs.pose.yaw),
                     obs.pose.y-reach*math.sin(obs.pose.yaw))
        self.goal_age = 0.
        self.yielding = True
        self.yielded_for_person = True
        return True

    def command(self, obs, dt=.05):
        x, y, yaw = obs.pose.x, obs.pose.y, obs.pose.yaw
        if obs.collision:
            self.status = "Bumper contact. Brakes on."
            return Command()
        nearby = [(px, py) for px, py in self.points if math.hypot(px-x, py-y) < 2.5]
        current_clearance = min((math.hypot(px-x, py-y) for px, py in nearby), default=3.)
        if current_clearance < self.body_radius:
            self.recovering = True
        if self.recovering and current_clearance < self.recovery_clearance:
            return self._retreat(obs, nearby, current_clearance)
        if self.recovering:
            if not self._back_goal(obs):
                self.status = "No clear retreat. Brakes on."
                return Command()
        self.recovering = False
        person_near = any((p.get('distance', 100) < .9 and abs(p.get('bearing', math.pi)) < .7)
                          or (not self.yielded_for_person and p.get('distance', 100) < 2.2
                              and abs(p.get('bearing', math.pi)) < .7)
                          for p in obs.people)
        if person_near and not self.yielding:
            if not self._back_goal(obs):
                self.status = "No clear retreat. Brakes on."
                return Command()
        if not any(p.get('distance', 100) < 1.5 for p in obs.people):
            self.yielding = False
        if not any(p.get('distance', 100) < 2.5 for p in obs.people):
            self.yielded_for_person = False
        goal_expired = self.goal is None or math.hypot(x-self.goal[0], y-self.goal[1]) < .55 or self.goal_age > 10
        if goal_expired:
            if self.yielding:
                if not self._back_goal(obs):
                    self.status = "No clear retreat. Brakes on."
                    return Command()
            else:
                self.choose_goal(obs)
        if self.goal is None:
            self.status = "No clear route. Waiting with dignity."
            return Command()
        gx, gy = self.goal
        initial_distance = math.hypot(gx-x, gy-y)
        choices = []
        forward_speed = min(self.max_speed, .18) if current_clearance < .9 else self.max_speed
        reverse_speed = min(.14 if current_clearance < .9 else .24, self.max_speed)
        rollout_steps = 40 if current_clearance < .9 else 16
        # A circumscribed circle accounts for both the front and rear overhang.
        for speed in (forward_speed, min(.12 if current_clearance < .9 else .28, forward_speed), -reverse_speed):
            for ratio in (-1., -.75, -.5, -.25, 0., .25, .5, .75, 1.):
                steering = ratio * self.max_steering
                px, py, heading, clearance, safe = x, y, yaw, 3., True
                for px, py, heading in self._arc(obs, speed, steering, rollout_steps):
                    clearance = min(clearance, min((math.hypot(px-qx, py-qy) for qx, qy in nearby), default=3.))
                    if clearance < self.body_radius:
                        safe = False
                        break
                if safe:
                    progress = initial_distance-math.hypot(gx-px, gy-py)
                    facing = math.cos(wrap(math.atan2(gy-py, gx-px)-heading))
                    score = 5*progress + .45*facing + .35*min(clearance, 1.2)
                    score -= .12*abs(steering-self.last_steering) + (.48 if speed < 0 else 0)
                    choices.append((score, speed, steering))
        if not choices:
            self.goal = None
            self.status = "Blocked. Brakes on. Ego intact."
            return Command()
        _, speed, steering = max(choices)
        self.last_steering = steering
        self.status = "Roaming, with no particular appointment" if speed > 0 else "Reversing, graciously"
        return Command(speed, steering)


ENCOUNTER_DISTANCE = 2.8


class Personality:
    """Rate-limit an observed encounter's horn independently of model speech."""

    def __init__(self, seed=None):
        self.level = 2
        self.cooldown = -100.

    def update(self, obs, now):
        blocked = any(0 < p.get('distance', 100) < ENCOUNTER_DISTANCE and abs(p.get('bearing', math.pi)) < .7
                      for p in obs.people)
        events = []
        if blocked and now - self.cooldown > 9:
            self.cooldown = now
            events = [("horn", "HONK HONK")]
        # The physical planner handles detours and braking. Model latency never
        # holds the rover in place or determines a motor command.
        return False, events
