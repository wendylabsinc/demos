"""Adapter for a commissioned Go2 navigation and perception installation.

This module does not implement SLAM, object detection, camera calibration, or
person identification. Supply their measured output at ``observation_url``.
GET that URL must return this JSON object, with coordinates in metres::

    {
      "map_id": "home-map-2026-10-v1",
      "frame_id": "map",
      "timestamp": 1790812800.25,
      "healthy": true,
      "reason": "",
      "localization": {"valid": true, "calibrated": true},
      "pose": {"x": 1.2, "y": 2.3, "yaw": 0.1},
      "detections": [
        {"label": "person", "track_id": "owner-1", "x": 2.0, "y": 2.5,
         "confidence": 0.95, "timestamp": 1790812800.20}
      ]
    }

Input timestamps are Unix epoch seconds from synchronized wall clocks.
Output timestamps use this process's monotonic clock. Each detection needs
its own capture timestamp; old detections are discarded rather than refreshed
when an HTTP response is received. ``map_id`` identifies a persistent, loaded
map revision, not an odometry origin or a new map created on each restart.
``calibrated`` attests that camera/depth transforms and map localization were
commissioned on this installation. An empty detections list is valid.

Configure an observation URL and the expected map ID, then explicitly call
``arm()`` through the operator's hardware CLI flag. A later task command must
call ``navigate()``. Arming never sends a navigation goal. The existing motion
and navigator services must be commissioned, and the fruit-hunter brain and
other motion controllers must be stopped before this adapter takes ownership.
The navigator keeps its own geofence; this module invents no room geometry.
No camera endpoint is assumed, so ``frame()`` returns None.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable


def _http_json(method: str, url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # Bound externally supplied observations and error responses.
        body = response.read(1_048_577)
    if len(body) > 1_048_576:
        raise ValueError("hardware response exceeds 1 MiB")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("hardware response must be a JSON object")
    return payload


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


class HardwareBackend:
    """Serial HTTP navigation with local arming and fresh measured inputs."""

    def __init__(
        self,
        observation_url: str,
        map_id: str,
        navigator_url: str = "http://127.0.0.1:3202",
        motion_url: str = "http://127.0.0.1:3201",
        *,
        max_observation_age_s: float = 0.75,
        arrival_tolerance_m: float = 0.25,
        request_timeout_s: float = 0.4,
        transport: Callable = _http_json,
        clock: Callable = time.monotonic,
        wall_clock: Callable = time.time,
    ):
        if not isinstance(map_id, str) or not map_id.strip():
            raise ValueError("a commissioned map_id is required")
        for name, url in (("observation_url", observation_url),
                          ("navigator_url", navigator_url), ("motion_url", motion_url)):
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an HTTP or HTTPS URL")
        self.map_id = map_id
        self.observation_url = observation_url
        self.navigator_url = navigator_url.rstrip("/")
        self.motion_url = motion_url.rstrip("/")
        self.max_observation_age_s = _number(max_observation_age_s, "max_observation_age_s")
        self.arrival_tolerance_m = _number(arrival_tolerance_m, "arrival_tolerance_m")
        self.request_timeout_s = _number(request_timeout_s, "request_timeout_s")
        if min(self.max_observation_age_s, self.arrival_tolerance_m, self.request_timeout_s) <= 0:
            raise ValueError("freshness, arrival tolerance, and HTTP timeout must be positive")
        self._transport = transport
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.RLock()
        self._armed = False
        self._state = "idle"
        self._reason = "hardware is disarmed"
        self._goal = None
        self._observation = None
        self._outcome_baseline = {}

    def _request(self, method: str, url: str) -> dict:
        payload = self._transport(method, url, self.request_timeout_s)
        if not isinstance(payload, dict):
            raise ValueError("hardware response must be a JSON object")
        return payload

    def _capture_time(self, value, field: str) -> float:
        age = self._wall_clock() - _number(value, field)
        if age < -0.05:
            raise ValueError(f"{field} is in the future; synchronize source clocks")
        if age > self.max_observation_age_s:
            raise ValueError(f"{field} is stale ({age:.3f}s)")
        return self._clock() - max(0.0, age)

    def _normalize(self, source: dict) -> dict:
        if source.get("map_id") != self.map_id:
            raise ValueError("observation map_id does not match the commissioned map")
        if source.get("frame_id") != "map":
            raise ValueError("observations must use the commissioned map frame")
        localization = source.get("localization")
        if not isinstance(localization, dict) or localization.get("valid") is not True:
            raise ValueError("map localization is not valid")
        if localization.get("calibrated") is not True:
            raise ValueError("perception and map transforms are not commissioned")
        if source.get("healthy") is not True:
            raise ValueError(str(source.get("reason") or "observation source is unhealthy"))
        timestamp = self._capture_time(source.get("timestamp"), "observation timestamp")
        pose_source = source.get("pose")
        if not isinstance(pose_source, dict):
            raise ValueError("a measured map-frame pose is required")
        pose = {key: _number(pose_source.get(key), f"pose.{key}") for key in ("x", "y", "yaw")}
        source_detections = source.get("detections")
        if not isinstance(source_detections, list):
            raise ValueError("detections must be a list of measured map-frame observations")
        detections = []
        for item in source_detections:
            if not isinstance(item, dict):
                raise ValueError("each detection must be an object")
            if not isinstance(item.get("label"), str) or not item["label"].strip():
                raise ValueError("each detection needs a label")
            if not isinstance(item.get("track_id"), str) or not item["track_id"].strip():
                raise ValueError("each detection needs a stable track_id")
            confidence = _number(item.get("confidence"), "detection confidence")
            if not 0 <= confidence <= 1:
                raise ValueError("detection confidence must be between zero and one")
            # Reject malformed/future timestamps, but discard genuinely old captures.
            captured = _number(item.get("timestamp"), "detection timestamp")
            if self._wall_clock() - captured > self.max_observation_age_s:
                continue
            detections.append({
                "label": item["label"], "track_id": item["track_id"],
                "x": _number(item.get("x"), "detection.x"),
                "y": _number(item.get("y"), "detection.y"),
                "confidence": confidence,
                "timestamp": self._capture_time(captured, "detection timestamp"),
            })
        return {"map_id": self.map_id, "timestamp": timestamp, "pose": pose,
                "detections": detections, "healthy": True, "reason": ""}

    def _halt(self) -> list[str]:
        """Attempt both stop paths even when one service is unreachable."""
        errors = []
        try:
            pause = self._request("POST", self.navigator_url + "/pause")
            if pause.get("mode") != "paused":
                raise RuntimeError("navigator did not acknowledge pause")
            state = self._request("GET", self.navigator_url + "/state")
            if state.get("mode") != "paused":
                raise RuntimeError("navigator pause did not persist on readback")
        except Exception as exc:
            errors.append(f"navigator stop failed: {exc}")
        try:
            stopped = self._request("POST", self.motion_url + "/stop")
            if stopped.get("stopped") is not True:
                raise RuntimeError("motion service did not acknowledge stop")
        except Exception as exc:
            errors.append(f"motion stop failed: {exc}")
        return errors

    def _fault(self, reason: str, *, halt: bool = True) -> None:
        errors = self._halt() if halt else []
        self._armed = False
        self._goal = None
        self._state = "fault"
        self._reason = "; ".join([reason] + errors)

    def observe(self) -> dict:
        with self._lock:
            try:
                observation = self._normalize(self._request("GET", self.observation_url))
            except Exception as exc:
                reason = f"measured observation unavailable: {exc}"
                if self._armed or self._state == "navigating":
                    self._fault(reason)
                observation = {"map_id": self.map_id, "timestamp": self._clock(),
                               "pose": None, "detections": [], "healthy": False, "reason": reason}
            self._observation = observation
            return observation

    def _ready(self) -> dict:
        navigator = self._request("GET", self.navigator_url + "/health")
        if navigator.get("ready") is not True:
            raise RuntimeError(str(navigator.get("reason") or "navigator is not ready"))
        mount = navigator.get("lidar_mount")
        if not isinstance(mount, dict) or mount.get("measured_on_this_robot") is not True:
            raise RuntimeError("navigator lidar mount has not been commissioned")
        fence = navigator.get("geofence")
        if not isinstance(fence, dict) or fence.get("enabled") is not True:
            raise RuntimeError("navigator requires a commissioned geofence")
        pose = navigator.get("pose")
        if not isinstance(pose, dict) or pose.get("usable") is not True or pose.get("frame") != "map":
            raise RuntimeError("navigator lacks a fresh map-frame localization")
        age = _number(pose.get("age_s"), "navigator pose age")
        if not 0 <= age <= self.max_observation_age_s:
            raise RuntimeError("navigator pose is stale")
        observed_pose = self._observation["pose"]
        mismatch = math.hypot(_number(pose.get("x"), "navigator pose.x") - observed_pose["x"],
                              _number(pose.get("y"), "navigator pose.y") - observed_pose["y"])
        if mismatch > max(0.5, self.arrival_tolerance_m * 2):
            raise RuntimeError("navigator and observation map poses disagree")
        motion = self._request("GET", self.motion_url + "/health")
        guard = motion.get("forward_guard")
        if motion.get("status") != "ok" or not isinstance(guard, dict) or guard.get("installed") is not True:
            raise RuntimeError("motion service or forward obstacle guard is not ready")
        if guard.get("have_scan") is not True:
            raise RuntimeError("motion service lacks a current obstacle scan")
        guard_age = _number(guard.get("age_s"), "obstacle scan age")
        stale_after = _number(guard.get("stale_after_s"), "obstacle scan freshness limit")
        if not 0 <= guard_age <= min(stale_after, self.max_observation_age_s):
            raise RuntimeError("motion obstacle scan is stale")
        guard_mount = guard.get("mount")
        if not isinstance(guard_mount, dict) or guard_mount.get("measured_on_this_robot") is not True:
            raise RuntimeError("motion lidar mount has not been commissioned")
        return navigator

    def arm(self) -> dict:
        """Operator action only. Validate commissioning without sending a goal."""
        with self._lock:
            if self._state == "navigating":
                return self.status() | {"accepted": False, "reason": "stop the current task before arming"}
            self._armed = False
            observation = self.observe()
            if not observation["healthy"]:
                self._fault(observation["reason"], halt=False)
                return self.status() | {"accepted": False}
            try:
                self._ready()
            except Exception as exc:
                self._fault(f"hardware cannot arm: {exc}", halt=False)
                return self.status() | {"accepted": False}
            self._armed = True
            self._state = "idle"
            self._reason = "hardware armed; awaiting a task command"
            return self.status() | {"accepted": True}

    def disarm(self) -> dict:
        with self._lock:
            result = self.stop()
            self._armed = False
            if self._state != "fault":
                self._reason = "hardware is disarmed"
            return self.status() | {"accepted": result.get("accepted", False)}

    def navigate(self, x: float, y: float) -> dict:
        with self._lock:
            if not self._armed:
                return self.status() | {"accepted": False, "reason": "hardware must be explicitly armed"}
            try:
                target = {"x": _number(x, "goal.x"), "y": _number(y, "goal.y")}
            except ValueError as exc:
                return self.status() | {"accepted": False, "reason": str(exc)}
            observation = self.observe()
            if not observation["healthy"]:
                return self.status() | {"accepted": False}
            try:
                self._ready()
                errors = self._halt()
                if errors:
                    raise RuntimeError("; ".join(errors))
                baseline = self._request("GET", self.navigator_url + "/state")
                if baseline.get("mode") != "paused":
                    raise RuntimeError("navigator resumed before goal replacement")
                self._outcome_baseline = dict(baseline.get("explore") or {})
                query = urllib.parse.urlencode(target)
                result = self._request("POST", self.navigator_url + "/goal?" + query)
                if result.get("accepted") is not True:
                    raise RuntimeError(str(result.get("reason") or "navigation goal was rejected"))
                self._goal = target
                self._state = "navigating"
                self._reason = "goal accepted; waiting for measured arrival"
                return self.status() | {"accepted": True}
            except Exception as exc:
                self._fault(f"navigation could not start: {exc}")
                return self.status() | {"accepted": False}

    def stop(self) -> dict:
        with self._lock:
            errors = self._halt()
            self._goal = None
            if errors:
                self._armed = False
                self._state = "fault"
                self._reason = "; ".join(errors)
            else:
                self._state = "idle"
                self._reason = "navigator paused and motion stop acknowledged"
            return self.status() | {"accepted": not errors}

    def tick(self, dt: float = 0.1) -> dict:
        """Poll measured state. ``dt`` never advances or fabricates robot pose."""
        with self._lock:
            if not self._armed:
                return self.status()
            observation = self.observe()
            if not observation["healthy"] or self._state != "navigating":
                return self.status()
            try:
                self._ready()
                navigator = self._request("GET", self.navigator_url + "/state")
                outcome = navigator.get("explore") or {}
                failed = outcome.get("goals_failed", 0) > self._outcome_baseline.get("goals_failed", 0)
                succeeded = outcome.get("goals_succeeded", 0) > self._outcome_baseline.get("goals_succeeded", 0)
                distance = math.hypot(observation["pose"]["x"] - self._goal["x"],
                                      observation["pose"]["y"] - self._goal["y"])
                mode = navigator.get("mode")
                if failed or mode in {"paused", "idle"}:
                    errors = self._halt()
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    # Existing navigator reports result counters. Where those are
                    # unavailable, measured arrival and idle are both required.
                    has_result = "goals_succeeded" in outcome or "goals_failed" in outcome
                    arrived = not failed and mode == "idle" and distance <= self.arrival_tolerance_m
                    if arrived and has_result:
                        arrived = succeeded
                    self._state = "arrived" if arrived else "blocked"
                    self._reason = ("arrival confirmed by measured map pose" if arrived else
                                    "navigation ended without a confirmed arrival")
                    self._goal = None
                elif mode != "goal":
                    raise RuntimeError("navigator ownership changed during this task")
            except Exception as exc:
                self._fault(f"navigation monitoring failed: {exc}")
            return self.status()

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "reason": self._reason, "armed": self._armed,
                    "map_id": self.map_id, "goal": dict(self._goal) if self._goal else None,
                    "backend": "hardware", "observation_url": self.observation_url}

    def frame(self, view: str = "robot"):
        return None
