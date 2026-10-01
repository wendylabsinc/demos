"""Execute short model-selected actions against fresh sensor interlocks.

This module makes no navigation or perception decisions. A completed task is
a model assertion and is recorded separately from observed physical evidence.
The transport must independently enforce its command watchdog and arm state.
"""
import math
import threading
from dataclasses import dataclass

from .models import Command


@dataclass(frozen=True)
class MotionAction:
    action: str
    duration: float = 0.5


class ActionExecutor:
    ACTIONS = {"forward", "turn_left", "turn_right", "hold", "stop", "complete"}

    def __init__(self, budget_s=120.0, geofence_m=5.0, travel_budget_m=25.0):
        if not all(math.isfinite(v) for v in (budget_s, geofence_m, travel_budget_m)):
            raise ValueError("executor limits must be finite")
        if not 1 <= budget_s <= 600 or not 1 <= geofence_m <= 20 or not 1 <= travel_budget_m <= 50:
            raise ValueError("executor limits exceed supported bounds")
        self.budget_s, self.geofence_m, self.travel_budget_m = budget_s, geofence_m, travel_budget_m
        self.lock = threading.RLock()
        self.enabled = False
        self.state, self.reason = "disabled", "explicit enable required"
        self.origin = self.previous_pose = (0.0, 0.0)
        self.started = self.last_tick = None
        self.action = None
        self.deadline = None
        self.distance_m = 0.0
        self.model_completion = None
        self.model_reason = ""

    def status(self):
        with self.lock:
            return {"state": self.state, "enabled": self.enabled, "reason": self.reason,
                    "action": self.action, "deadline": self.deadline,
                    "distance_m": round(self.distance_m, 3), "budget_s": self.budget_s,
                    "model_reason": self.model_reason,
                    "model_completion": dict(self.model_completion) if self.model_completion else None}

    def stop(self, reason="operator stop"):
        with self.lock:
            self.enabled, self.action, self.deadline = False, None, None
            self.state, self.reason = "disabled", reason
            return Command(reason=reason)

    def _fault(self, reason):
        self.enabled, self.action, self.deadline = False, None, None
        self.state, self.reason = "fault", reason
        return Command(reason=reason)

    def _valid(self, obs):
        if not obs.fresh or not all(math.isfinite(v) for v in (obs.x, obs.y, obs.yaw)):
            return "pose or camera missing, stale or invalid"
        if not obs.scan_fresh or obs.front_clearance is None:
            return "fresh scan required"
        if not math.isfinite(obs.front_clearance) or obs.front_clearance < 0:
            return "invalid scan clearance"
        for ray in obs.sectors:
            if len(ray) != 2 or not all(math.isfinite(v) for v in ray) or ray[1] < 0:
                return "invalid scan sector"
        return None

    def enable(self, obs, now):
        with self.lock:
            if not math.isfinite(now):
                return self._fault("invalid control clock")
            invalid = self._valid(obs)
            if invalid:
                return self._fault(invalid)
            self.enabled = True
            self.origin = self.previous_pose = (obs.x, obs.y)
            self.started = self.last_tick = now
            self.distance_m = 0.0
            self.model_completion = None
            self.model_reason = ""
            self.action = self.deadline = None
            self.state, self.reason = "waiting", "waiting for model decision"
            return Command(reason=self.reason)

    def _check(self, obs, now):
        if not math.isfinite(now) or now < self.last_tick or now - self.last_tick > 1.0:
            return "control clock gap or reversal"
        invalid = self._valid(obs)
        if invalid:
            return invalid
        if now - self.started >= self.budget_s:
            return "session time budget reached"
        pose = (obs.x, obs.y)
        if math.dist(pose, self.origin) > self.geofence_m - 0.4:
            return "geofence reached"
        self.distance_m += math.dist(pose, self.previous_pose)
        self.previous_pose = pose
        if self.distance_m > self.travel_budget_m:
            return "session travel budget reached"
        self.last_tick = now
        return None

    @staticmethod
    def _field(decision, name, default=None):
        return decision.get(name, default) if isinstance(decision, dict) else getattr(decision, name, default)

    def begin(self, decision, obs, now):
        """Accept one model decision, or return zero when disabled/faulted."""
        with self.lock:
            if not self.enabled:
                return Command(reason=self.reason)
            failure = self._check(obs, now)
            if failure:
                return self._fault(failure)
            action = self._field(decision, "action")
            duration = self._field(decision, "duration", self._field(decision, "duration_s", 0.5))
            if not isinstance(action, str) or action not in self.ACTIONS:
                return self._fault("unsupported model action")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration):
                return self._fault("invalid model action duration")
            if not 0 <= duration <= 1.0 or (action in {"forward", "turn_left", "turn_right"} and duration == 0):
                return self._fault("model action duration outside 0 to 1 second")
            self.model_reason = str(self._field(decision, "reason", self._field(decision, "reasoning", "")))[:2000]
            if action == "complete":
                self.model_completion = {"reported": True, "verified": False,
                                         "reason": self.model_reason,
                                         "observations": str(self._field(decision, "observations", ""))[:4000]}
                self.enabled = False
                self.action = self.deadline = None
                self.state, self.reason = "complete", "model reports task complete; physical result unverified"
                return Command(reason=self.reason)
            if action == "stop":
                return self.stop("model requested stop")
            self.action, self.deadline = action, now + duration
            self.state, self.reason = "running", "executing model-selected " + action
            return self._command(obs)

    def step(self, obs, now):
        with self.lock:
            if not self.enabled:
                return Command(reason=self.reason)
            failure = self._check(obs, now)
            if failure:
                return self._fault(failure)
            if self.action is None or now >= self.deadline:
                self.action = self.deadline = None
                self.state, self.reason = "waiting", "waiting for model decision"
                return Command(reason=self.reason)
            return self._command(obs)

    def _command(self, obs):
        if self.action == "hold":
            return Command(reason="model selected hold")
        if self.action == "forward":
            if obs.front_clearance < 1.0:
                return self._fault("model forward action blocked by scan")
            predicted = (obs.x + 0.7 * math.cos(obs.yaw), obs.y + 0.7 * math.sin(obs.yaw))
            if math.dist(predicted, self.origin) > self.geofence_m - 0.4:
                return self._fault("model forward action reaches geofence")
            return Command(vx=0.55, reason=self.reason)
        if self.action in {"turn_left", "turn_right"}:
            # Full swept-footprint clearance needs measured front, sides and rear.
            bearings = [math.atan2(math.sin(a), math.cos(a)) for a, _ in obs.sectors]
            if not all(any(abs(math.atan2(math.sin(a-required), math.cos(a-required))) < 0.4
                           for a in bearings) for required in (0, math.pi/2, -math.pi/2, math.pi)):
                return self._fault("rotation requires front, side and rear scan coverage")
            if min(r for _, r in obs.sectors) < 0.55:
                return self._fault("model rotation blocked by swept footprint scan")
            return Command(yaw_rate=0.45 if self.action == "turn_left" else -0.45, reason=self.reason)
        return Command(reason="waiting for model decision")
