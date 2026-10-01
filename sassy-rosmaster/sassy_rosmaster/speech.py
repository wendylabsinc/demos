"""Online Ollama dialogue and goal deliberation, outside the motion loop.

Model replies can provide text and choose a proposed goal. They never provide
motor commands. The server must revalidate each chosen goal against live lidar.
"""
from concurrent.futures import Future
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class ModelError(RuntimeError):
    pass


class InferenceLane:
    """One daemon worker with a bounded queue and speech ahead of exploration."""

    def __init__(self):
        self._condition = threading.Condition()
        self._queue = []
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="rover-local-model", daemon=True)
        self._thread.start()

    def submit(self, kind, function):
        future = Future()
        with self._condition:
            if self._closed:
                raise ModelError("Model worker is closed")
            if len(self._queue) >= 2 or any(k == kind for k, _, _ in self._queue):
                raise ModelError("Model request queue is full")
            self._queue.append((kind, function, future))
            self._condition.notify()
        return future

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._queue)
                if self._closed:
                    return
                index = next((i for i, (kind, _, _) in enumerate(self._queue)
                              if kind == "speech"), 0)
                _, function, future = self._queue.pop(index)
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function())
                except Exception as exc:
                    future.set_exception(exc)

    def close(self):
        with self._condition:
            self._closed = True
            for _, _, future in self._queue:
                future.cancel()
            self._queue.clear()
            self._condition.notify_all()


class OllamaClient:
    def __init__(self, endpoint=None, model=None, *, timeout=15,
                 opener=urllib.request.urlopen):
        self.endpoint = (endpoint or os.getenv("ROVER_LLM_URL", "http://127.0.0.1:11434")).rstrip("/")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
            raise ValueError("ROVER_LLM_URL must be a plain http(s) Ollama URL")
        self.model = model or os.getenv("ROVER_LLM_MODEL", "qwen3:1.7b")
        if not isinstance(self.model, str) or not self.model or len(self.model) > 100:
            raise ValueError("Set a valid installed ROVER_LLM_MODEL")
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or not .1 <= self.timeout <= 30:
            raise ValueError("Model timeout must be between 0.1 and 30 seconds")
        self._opener = opener

    def generate(self, system, context, schema):
        request = urllib.request.Request(self.endpoint+"/api/chat", data=json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(context, allow_nan=False, separators=(",", ":"))}],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"temperature": .85, "num_predict": 100, "num_ctx": 2048},
        }, allow_nan=False).encode(), headers={"Content-Type": "application/json"})
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise ModelError("Ollama response exceeded the size limit")
            reply = json.loads(raw)
        except urllib.error.HTTPError as exc:
            raise ModelError(f"Ollama HTTP {exc.code}; check the installed model and server") from exc
        except (OSError, ValueError, TypeError) as exc:
            raise ModelError(f"Ollama request failed: {type(exc).__name__}") from exc
        if not isinstance(reply, dict) or reply.get("done") is not True:
            raise ModelError("Ollama returned an incomplete response")
        model = reply.get("model")
        tokens = reply.get("eval_count")
        if not isinstance(model, str) or not model or isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ModelError("Ollama response has no model identity or generated token evidence")
        try:
            message = reply["message"]
            if message.get("role") != "assistant":
                raise ValueError("not assistant")
            content = json.loads(message["content"])
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ModelError("Ollama did not return the required JSON response") from exc
        if not isinstance(content, dict):
            raise ModelError("Ollama response JSON must be an object")
        return dict(content=content, model=model, requested_model=self.model,
                    eval_count=tokens, prompt_eval_count=reply.get("prompt_eval_count"),
                    created_at=reply.get("created_at"), total_duration_ns=reply.get("total_duration"))


def _finite(value, name, low=-1e6, high=1e6):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number between {low} and {high}")
    return value


def _text(value, name, limit):
    if not isinstance(value, str):
        raise ModelError(f"Model {name} is not text")
    value = " ".join(value.split()).strip('"')
    if not value or len(value) > limit or "<think>" in value or "</think>" in value:
        raise ModelError(f"Model {name} is empty, too long or contains thinking markup")
    return value


class _OnlineJob:
    kind = "model"

    def __init__(self, *, client=None, lane=None, clock=time.monotonic,
                 min_interval=9, max_age=12):
        self.client = client or OllamaClient()
        self.lane = lane or InferenceLane()
        self._owns_lane = lane is None
        self._clock = clock
        self._interval = float(min_interval)
        self._max_age = float(max_age)
        self._lock = threading.Lock()
        self._pending = None
        self._counter = 0
        self._last_request = -math.inf
        self._history = []
        self._state = "unverified"
        self._error = None
        self._actual_model = None
        self._last_result = None
        self._completed = 0
        self._cancelled = set()

    def request(self, context):
        """Schedule one bounded inference. Never wait for a model in this call."""
        prepared = self._context(context)
        with self._lock:
            now = self._clock()
            if self._pending is not None or now-self._last_request < self._interval:
                return False
            self._counter += 1
            if self.kind == "speech":
                prepared["recent_dialogue"] = list(self._history[-5:])
            future = self.lane.submit(self.kind, lambda: self.client.generate(
                self.system, prepared, self._schema(prepared)))
            self._pending = (future, self._counter, now, prepared)
            self._last_request = now
            self._state, self._error = "thinking", None
            return True

    def poll(self, *, allow_result=True):
        """Return completed results for the control loop to validate and emit."""
        with self._lock:
            if self._pending is None or not self._pending[0].done():
                return []
            future, request_id, started, context = self._pending
            self._pending = None
            try:
                reply = future.result()
                self._actual_model = reply["model"]
                if request_id in self._cancelled or not allow_result:
                    self._cancelled.discard(request_id)
                    self._state = "ready"
                    return []
                if self._clock()-started > self._max_age:
                    raise ModelError("Model reply expired; the scene may have changed")
                result = self._validate(reply["content"], context)
                result.update({k: v for k, v in reply.items() if k != "content"})
                result.update(kind=self.kind, source="ollama", request_id=request_id,
                              latency_ms=round((self._clock()-started)*1000))
                self._history.append(result.get("text", result.get("reason", "")))
                self._history = self._history[-5:]
                self._last_result = result
                self._completed += 1
                self._state, self._error = "ready", None
                return [result]
            except Exception as exc:
                self._state, self._error = "unavailable", str(exc)
                return [dict(kind=self.kind+"_error", source="ollama", request_id=request_id,
                             requested_model=self.client.model, error=self._error)]

    def cancel_pending(self):
        """Invalidate pending dialogue after Stop, Reset or a sensor fault."""
        with self._lock:
            if self._pending:
                self._cancelled.add(self._pending[1])

    def status(self):
        with self._lock:
            return dict(state=self._state, backend="ollama", endpoint=self.client.endpoint,
                        model=self._actual_model or self.client.model,
                        requested_model=self.client.model, actual_model=self._actual_model,
                        pending=self._pending is not None, error=self._error,
                        completed=self._completed, last_result=self._last_result)

    def close(self):
        self.cancel_pending()
        if self._owns_lane:
            self.lane.close()


class Speech(_OnlineJob):
    kind = "speech"
    system = (
        "You are a harmless, cheeky British rover. Ask the person in your way to "
        "move aside and add ONE short joke about your own oversized ego. "
        "Sass 1 polite, 2 teasing, 3 irritable with mild swearing. "
        "Speak directly as I/me to you/your. Never offer to move aside yourself. "
        "Aim for 18 words. No threats, retaliation, slurs, personal appearance, "
        "narration, brand names, stage directions or markdown. "
        "Avoid recent dialogue. Return JSON with line."
    )

    def _context(self, context):
        if not isinstance(context, dict):
            raise ValueError("Speech context must be a dictionary")
        level = int(_finite(context.get("sass", 2), "sass", 1, 3))
        result = dict(reason=str(context.get("reason", "person_in_way"))[:100], sass=level,
                      tone={1: "polite", 2: "playfully sarcastic", 3: "comically grumpy; mild swearing allowed"}[level],
                      speaking_goal="Ask the person blocking the route to move aside")
        for key in ("coverage", "distance", "speed", "nearest_person_distance"):
            if key in context:
                result[key] = _finite(context[key], key)
        for key in ("distance_m", "bearing_deg", "speed_m_s", "coverage_percent", "travelled_m"):
            if key in context:
                result[key] = _finite(context[key], key)
        people = context.get("people", [])
        if not isinstance(people, list):
            raise ValueError("Speech people must be a list")
        result["people"] = [{key: _finite(p[key], key) for key in ("distance", "bearing") if key in p}
                            for p in people[:5]]
        return result

    def _schema(self, context):
        return {"type": "object", "properties": {"line": {"type": "string", "maxLength": 220}},
                "required": ["line"], "additionalProperties": False}

    def _validate(self, output, context):
        text = _text(output.get("line"), "line", 220)
        direct = re.search(r"\b(?:you|your|you're|you’re)\b", text, re.IGNORECASE)
        imperative = re.match(r"^(?:(?:oi|hey|darling|love)[,!]?\s+)?(?:please\s+)?"
                              r"(?:move|step|make|clear|shift|get)\b", text, re.IGNORECASE)
        if not direct and not imperative:
            raise ModelError("Model line did not address the person directly")
        if any(marker in text for marker in ("(", ")", "[", "]", "*", "`")):
            raise ModelError("Model line contained stage directions or markdown")
        if re.search(r"\b(?:robot|rosmaster)\b", text, re.IGNORECASE):
            raise ModelError("Model line described the robot instead of speaking as it")
        if re.search(r"\b(?:regret|retaliate|kill|hurt|crush|attack|ram)\b|"
                     r"\brun\s+(?:you|them|someone)\s+over\b|\bmake\s+you\s+pay\b", text, re.IGNORECASE):
            raise ModelError("Model line included a threat instead of a comic remark")
        if any(text.casefold() == previous.casefold() for previous in self._history):
            raise ModelError("Model repeated a recent line")
        return dict(text=text, reason=context["reason"])

    def explorer(self):
        """Share the same bounded worker so exploration cannot flood inference."""
        return Explorer(client=self.client, lane=self.lane, clock=self._clock)


class Explorer(_OnlineJob):
    kind = "exploration"
    system = (
        "You select the next exploration goal for a small indoor ROSMASTER robot. "
        "The candidate locations come from fresh lidar and already pass the host's "
        "collision checks. Select one listed goal id. Prefer unfamiliar space and "
        "comfortable clearance while avoiding repetitive backtracking. Consider the "
        "current map coverage and the previous goal. You cannot control speed, "
        "steering or brakes. The host will revalidate your choice before moving. "
        "Return only JSON with integer goal_id and a concise reason under 20 words."
    )

    def __init__(self, **kwargs):
        kwargs.setdefault("min_interval", 6)
        kwargs.setdefault("max_age", 8)
        super().__init__(**kwargs)

    def _context(self, context):
        if not isinstance(context, dict):
            raise ValueError("Exploration context must be a dictionary")
        candidates = context.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
            raise ValueError("Exploration needs between 1 and 12 visible candidates")
        prepared, ids = [], set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("Exploration candidates must be dictionaries")
            identity = candidate.get("id")
            if isinstance(identity, bool) or not isinstance(identity, int) or identity in ids:
                raise ValueError("Each candidate needs a unique integer id")
            ids.add(identity)
            prepared.append(dict(id=identity, **{
                key: round(_finite(candidate[key], key), 3 if key in ("bearing", "novelty") else 2) for key in
                ("x", "y", "novelty", "clearance", "bearing", "distance")}))
        result = dict(candidates=prepared, coverage=round(_finite(context.get("coverage", 0), "coverage", 0, 100), 1))
        previous = context.get("recent_goal")
        if previous is not None:
            if not isinstance(previous, (list, tuple)) or len(previous) != 2:
                raise ValueError("recent_goal needs two coordinates")
            result["recent_goal"] = [round(_finite(v, "recent_goal"), 2) for v in previous]
        return result

    def _schema(self, context):
        return {"type": "object", "properties": {
            "goal_id": {"type": "integer", "enum": [c["id"] for c in context["candidates"]]},
            "reason": {"type": "string", "maxLength": 160}}, "required": ["goal_id", "reason"],
            "additionalProperties": False}

    def _validate(self, output, context):
        identity = output.get("goal_id")
        if isinstance(identity, bool) or not isinstance(identity, int):
            raise ModelError("Model goal_id is not an integer")
        candidate = next((c for c in context["candidates"] if c["id"] == identity), None)
        if candidate is None:
            raise ModelError("Model selected a goal that was not proposed")
        return dict(goal_id=identity, candidate=dict(candidate),
                    reason=_text(output.get("reason"), "reason", 160))
