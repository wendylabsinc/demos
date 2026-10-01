from concurrent.futures import Future
import json
import threading
import unittest

from sassy_rosmaster.speech import Explorer, InferenceLane, ModelError, OllamaClient, Speech


class FakeClock:
    def __init__(self):
        self.now = 10.

    def __call__(self):
        return self.now


class ManualLane:
    def __init__(self):
        self.jobs = []

    def submit(self, kind, function):
        future = Future()
        self.jobs.append((kind, function, future))
        return future

    def complete(self):
        _, function, future = self.jobs[-1]
        try:
            future.set_result(function())
        except Exception as exc:
            future.set_exception(exc)


class FakeClient:
    endpoint = "http://local-model:11434"
    model = "test-installed-model"

    def __init__(self):
        self.content = {"line": "Could you make room? My map has better manners than this queue."}
        self.contexts = []
        self.error = None

    def generate(self, system, context, schema):
        self.contexts.append(context)
        if self.error:
            raise self.error
        return dict(content=self.content, model=self.model, requested_model=self.model,
                    eval_count=20, prompt_eval_count=180, created_at="2026-10-01T00:00:00Z")


def candidates():
    return [{"id": 7, "x": 2., "y": 1., "novelty": .8, "clearance": 1.1,
             "bearing": .2, "distance": 2.2},
            {"id": 8, "x": -1., "y": 0., "novelty": .2, "clearance": .7,
             "bearing": 3., "distance": 1.}]


class OnlineSpeechTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.client, self.lane = FakeClock(), FakeClient(), ManualLane()
        self.voice = Speech(client=self.client, lane=self.lane, clock=self.clock)

    def test_request_and_poll_never_wait_for_inference(self):
        self.assertTrue(self.voice.request({"sass": 3, "distance_m": 1.2, "bearing_deg": 5,
                          "coverage_percent": 37, "activity": "yielding", "intention": "explore"}))
        self.assertEqual(self.voice.status()["state"], "thinking")
        self.assertEqual(self.voice.poll(), [])
        self.assertFalse(self.voice.request({}))
        self.lane.complete()
        result = self.voice.poll()[0]
        self.assertEqual(result["kind"], "speech")
        self.assertEqual(result["source"], "ollama")
        self.assertEqual(result["model"], self.client.model)
        self.assertEqual(self.client.contexts[0]["distance_m"], 1.2)
        self.assertEqual(self.voice.status()["state"], "ready")

    def test_failures_are_visible_and_do_not_create_canned_speech(self):
        self.client.error = ModelError("model service unavailable")
        self.voice.request({})
        self.lane.complete()
        result = self.voice.poll()[0]
        self.assertEqual(result["kind"], "speech_error")
        self.assertNotIn("text", result)
        self.assertEqual(self.voice.status()["state"], "unavailable")
        self.assertEqual(self.voice.status()["error"], "model service unavailable")

    def test_stop_and_scene_change_discard_completed_speech(self):
        self.voice.request({})
        self.voice.cancel_pending()
        self.lane.complete()
        self.assertEqual(self.voice.poll(), [])
        self.clock.now += 10
        self.voice.request({})
        self.lane.complete()
        self.assertEqual(self.voice.poll(allow_result=False), [])

    def test_expired_reply_is_not_spoken(self):
        self.voice.request({})
        self.clock.now += 13
        self.lane.complete()
        result = self.voice.poll()[0]
        self.assertEqual(result["kind"], "speech_error")
        self.assertIn("expired", result["error"])

    def test_rate_limit_and_recent_dialogue_prevent_repeated_lines(self):
        self.voice.request({})
        self.lane.complete()
        self.voice.poll()
        self.assertFalse(self.voice.request({}))
        self.clock.now += 9
        self.assertTrue(self.voice.request({}))
        self.lane.complete()
        result = self.voice.poll()[0]
        self.assertEqual(result["kind"], "speech_error")
        self.assertIn("repeated", result["error"])
        self.assertEqual(len(self.client.contexts[-1]["recent_dialogue"]), 1)

    def test_scene_description_and_stage_directions_are_not_spoken(self):
        for line in ("A tiny robot yields and continues mapping.",
                     "Can you move aside? (Wheezing)",
                     "The ROSMASTER robot wants you to move aside.",
                     "Move aside, or I'll make you regret it."):
            with self.subTest(line=line):
                self.clock.now += 10
                self.client.content = {"line": line}
                self.voice.request({})
                self.lane.complete()
                self.assertEqual(self.voice.poll()[0]["kind"], "speech_error")

    def test_explorer_only_returns_a_proposed_goal_snapshot(self):
        explorer = self.voice.explorer()
        self.client.content = {"goal_id": 7, "reason": "More unseen space with comfortable clearance."}
        source = candidates()
        explorer.request(dict(candidates=source, coverage=37, recent_goal=[0., 1.]))
        source[0]["x"] = 99
        self.lane.complete()
        result = explorer.poll()[0]
        self.assertEqual(result["goal_id"], 7)
        self.assertEqual(result["candidate"]["x"], 2.)
        self.assertNotIn("speed", result)
        self.assertNotIn("steering", result)

    def test_explorer_rejects_unproposed_goal_and_stale_result(self):
        explorer = self.voice.explorer()
        self.client.content = {"goal_id": 999, "reason": "invented goal"}
        explorer.request(dict(candidates=candidates(), coverage=37))
        self.lane.complete()
        self.assertEqual(explorer.poll()[0]["kind"], "exploration_error")
        self.clock.now += 9
        self.client.content = {"goal_id": 7, "reason": "Fresh space."}
        explorer.request(dict(candidates=candidates(), coverage=37))
        self.clock.now += 9
        self.lane.complete()
        self.assertIn("expired", explorer.poll()[0]["error"])


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return json.dumps(self.payload).encode()


class ModelTransportTests(unittest.TestCase):
    def test_actual_response_requires_model_identity_and_generated_tokens(self):
        reply = dict(model="qwen3:0.6b", done=True, eval_count=19,
                     message=dict(role="assistant", content='{"line":"A generated remark."}'))
        sent = []

        def opener(request, timeout):
            sent.append(json.loads(request.data))
            return Response(reply)

        client = OllamaClient(opener=opener)
        result = client.generate("system", {"distance": 1.2}, {"type": "object"})
        self.assertEqual(result["model"], "qwen3:0.6b")
        self.assertEqual(sent[0]["think"], False)
        self.assertEqual(sent[0]["stream"], False)
        reply["eval_count"] = 0
        with self.assertRaisesRegex(ModelError, "token evidence"):
            client.generate("system", {}, {})

    def test_single_worker_prioritizes_queued_speech(self):
        lane = InferenceLane()
        active, release, spoken = threading.Event(), threading.Event(), []

        def first():
            active.set()
            release.wait(timeout=2)

        running = lane.submit("exploration", first)
        self.assertTrue(active.wait(timeout=2))
        queued_goal = lane.submit("exploration", lambda: spoken.append("exploration"))
        queued_speech = lane.submit("speech", lambda: spoken.append("speech"))
        release.set()
        running.result(timeout=2)
        queued_goal.result(timeout=2)
        queued_speech.result(timeout=2)
        self.assertEqual(spoken, ["speech", "exploration"])
        lane.close()


if __name__ == "__main__":
    unittest.main()
