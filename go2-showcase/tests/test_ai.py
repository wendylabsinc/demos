import base64
import json
import math
import unittest
from urllib.error import URLError
from showcase.ai import Decision, VisionAgent, ModelUnavailable, NUM_CTX


class Reply:
    def __init__(self, data):
        self.data = data
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, count):
        return json.dumps(self.data).encode()


class AITests(unittest.TestCase):
    # Protocol fixture, not a model-generated decision or a live AI smoke test.
    jpeg = b'\xff\xd8image bytes\xff\xd9'
    valid = {'action': 'forward', 'duration_s': .5, 'reason': 'Visible corridor is clear',
             'observations': ['A corridor ahead'], 'evidence_bbox': None}

    def test_openai_payload_contains_actual_frame_and_goal(self):
        requests = []
        def opener(request, timeout):
            requests.append((request, timeout))
            return Reply({'choices': [{'message': {'content': json.dumps(self.valid)}}]})
        agent = VisionAgent(url='https://model.test/v1', model='vision-test', opener=opener)
        decision = agent.decide('Find the notebook', self.jpeg,
                                {'front_clearance': 3, 'fixture_position': [1, 2], 'detections': ['fake']}, [])
        self.assertEqual(decision.action, 'forward')
        self.assertEqual(decision.source, 'model')
        payload = json.loads(requests[0][0].data)
        user = payload['messages'][1]['content']
        self.assertIn('Find the notebook', user[0]['text'])
        self.assertNotIn('fixture_position', user[0]['text'])
        self.assertNotIn('detections', user[0]['text'])
        self.assertEqual(user[1]['image_url']['url'], 'data:image/jpeg;base64,'+base64.b64encode(self.jpeg).decode())
        self.assertEqual(requests[0][0].full_url, 'https://model.test/v1/chat/completions')
        self.assertEqual(requests[0][1], 120)

    def test_ollama_image_and_strict_format(self):
        captured = []
        def opener(request, timeout):
            captured.append(json.loads(request.data))
            return Reply({'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(self.valid)}})
        agent = VisionAgent(url='http://model.test:11434/api/chat', model='local-vision', opener=opener)
        agent.decide('Find a person', self.jpeg, {'fresh': True}, [{'action': 'hold'}])
        self.assertEqual(captured[0]['messages'][1]['images'], [base64.b64encode(self.jpeg).decode()])
        self.assertFalse(captured[0]['stream'])
        self.assertFalse(captured[0]['format']['additionalProperties'])

    def test_ollama_schema_json_in_thinking_channel_still_validated(self):
        agent = VisionAgent(url='http://model.test/api/chat', model='vision',
                            opener=lambda *args, **kwargs: Reply({'done': True, 'done_reason': 'stop',
                            'message': {'content': '', 'thinking': json.dumps(self.valid)}}))
        decision = agent.decide('Find keys', self.jpeg, {}, [])
        self.assertEqual(decision.response_channel, 'thinking')
        self.assertEqual(decision.action, 'forward')
        for content in ('I think I should go forward', json.dumps(self.valid | {'duration_s': 2})):
            agent.opener = lambda *args, **kwargs: Reply({'done': True, 'done_reason': 'stop',
                            'message': {'content': '', 'thinking': content}})
            with self.assertRaises(ModelUnavailable):
                agent.decide('Find keys', self.jpeg, {}, [])

    def test_incomplete_generation_fails_even_if_json_is_valid(self):
        agent = VisionAgent(url='http://model.test/api/chat', model='vision',
                            opener=lambda *args, **kwargs: Reply({'done': True, 'done_reason': 'length',
                            'message': {'content': json.dumps(self.valid)}}))
        with self.assertRaises(ModelUnavailable):
            agent.decide('Find keys', self.jpeg, {}, [])

    def test_full_sensor_and_long_history_context_stays_bounded(self):
        payloads = []
        def opener(request, timeout):
            payloads.append(json.loads(request.data))
            return Reply({'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(self.valid)}})
        sectors = [[-math.pi+i*2*math.pi/72, 4.123456789] for i in range(72)]
        sectors[1][1] = .234567891
        history = [{'action': 'hold', 'reason': 'observed actual image '*1000} for _ in range(40)]
        agent = VisionAgent(url='http://model.test/api/chat', model='vision', opener=opener)
        agent.decide('Find keys', self.jpeg, {'scan_sectors': sectors, 'pose': {'x': .123456789}}, history)
        context = json.loads(payloads[0]['messages'][1]['content'])
        state = context['measured_sensors']
        self.assertEqual(len(state['scan_sectors']), 12)
        self.assertEqual(state['scan_sectors'][0][1], .235)
        self.assertEqual(state['pose']['x'], .123)
        self.assertEqual(len(context['recent_action_results']), 4)
        self.assertTrue(all(len(h['reason']) <= 600 for h in context['recent_action_results']))
        self.assertLess(len(payloads[0]['messages'][1]['content']), 4000)
        self.assertEqual(payloads[0]['options']['num_ctx'], 8192)

    def test_rejects_unbounded_or_unknown_actions(self):
        for patch in ({'duration_s': 1.1}, {'duration_s': float('nan')}, {'duration_s': True},
                      {'action': 'reverse'}, {'action': 'forward', 'duration_s': 0},
                      {'action': 'complete', 'duration_s': .5}, {'extra_command': 'run'},
                      {'evidence_bbox': [0, 0, 1.1, 1]}, {'evidence_bbox': [1, 0, 0, 1]}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                Decision.parse(self.valid | patch)

    def test_provider_failure_never_returns_fallback_motion(self):
        def opener(request, timeout):
            raise TimeoutError('secret must not appear')
        agent = VisionAgent(url='http://model.test/api/chat', model='vision', opener=opener)
        with self.assertRaises(ModelUnavailable) as error:
            agent.decide('Find keys', self.jpeg, {}, [])
        self.assertNotIn('secret', str(error.exception))
        self.assertEqual(agent.last_status['state'], 'error')

    def test_invalid_json_and_absent_config_fail_closed(self):
        agent = VisionAgent(url='', model='')
        with self.assertRaises(ModelUnavailable):
            agent.decide('Find keys', self.jpeg, {}, [])
        agent = VisionAgent(url='http://model.test/api/chat', model='vision',
                            opener=lambda *args, **kwargs: Reply({'done': True, 'message': {'content': 'not JSON'}}))
        with self.assertRaises(ModelUnavailable):
            agent.decide('Find keys', self.jpeg, {}, [])


class WarmUpTests(unittest.TestCase):
    def test_ollama_warm_up_loads_with_the_decision_context_and_no_image(self):
        seen = []
        def opener(request, timeout):
            seen.append((request.full_url, request.get_method(), json.loads(request.data)))
            return Reply({'done': True, 'done_reason': 'load'})
        VisionAgent(url='http://127.0.0.1:11434', provider='ollama', model='qwen3-vl:4b', opener=opener).warm_up()
        url, method, payload = seen[0]
        self.assertEqual((url, method), ('http://127.0.0.1:11434/api/chat', 'POST'))
        self.assertEqual(payload['messages'], [])
        self.assertEqual(payload['keep_alive'], -1)
        self.assertEqual(payload['options']['num_ctx'], NUM_CTX)

    def test_decisions_use_the_warm_up_context_size(self):
        seen = []
        def opener(request, timeout):
            seen.append(json.loads(request.data))
            return Reply({'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(AITests.valid)}})
        VisionAgent(url='http://m.test/api/chat', model='v', opener=opener).decide('Find keys', AITests.jpeg, {}, [])
        self.assertEqual(seen[0]['options']['num_ctx'], NUM_CTX)

    def test_openai_compatible_warm_up_is_a_one_token_text_request(self):
        seen = []
        def opener(request, timeout):
            seen.append(json.loads(request.data))
            return Reply({'choices': []})
        agent = VisionAgent(url='https://model.test/v1', model='vision', opener=opener)
        agent.warm_up()
        self.assertEqual(seen[0]['max_tokens'], 1)
        self.assertNotIn('image', json.dumps(seen[0]))
        self.assertIsNone(agent.loaded())

    def test_warm_up_failures_are_model_unavailable_without_details(self):
        def refuse(request, timeout):
            raise URLError('connection refused to http://secret-host')
        agent = VisionAgent(url='http://127.0.0.1:11434', provider='ollama', model='m', opener=refuse)
        for call in (agent.warm_up, agent.loaded):
            with self.assertRaisesRegex(ModelUnavailable, 'unreachable or timed out') as caught:
                call()
            self.assertNotIn('secret-host', str(caught.exception))
        unfinished = VisionAgent(url='http://m.test', provider='ollama', model='m',
                                 opener=lambda *a, **k: Reply({'done': False}))
        with self.assertRaises(ModelUnavailable):
            unfinished.warm_up()

    def test_loaded_reads_ollama_resident_models(self):
        seen = []
        def opener(request, timeout):
            seen.append((request.full_url, request.get_method()))
            return Reply({'models': [{'name': 'qwen3-vl:4b', 'model': 'qwen3-vl:4b'}]})
        agent = VisionAgent(url='http://127.0.0.1:11434/api/chat', provider='ollama', model='qwen3-vl:4b', opener=opener)
        self.assertTrue(agent.loaded())
        self.assertEqual(seen[0], ('http://127.0.0.1:11434/api/ps', 'GET'))
        self.assertFalse(VisionAgent(url='http://127.0.0.1:11434', provider='ollama', model='other', opener=opener).loaded())


if __name__ == '__main__':
    unittest.main()
