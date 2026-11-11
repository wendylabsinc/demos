import asyncio
import base64
import hashlib
import io
import json
import socket
import unittest
from unittest.mock import patch
import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from zeroconf import ServiceInfo
from app import app, Event, Fleet, MODEL, VISION_MODEL, Station, public_job, lifespan
from PIL import Image
from discovery import resolved_peer, SERVICE


def event(identity='test-1'):
    return Event(event_id=identity, source='test', zone='gate', motion=True)


def result(e, node='remote'):
    return {'event_id': e.event_id, 'served_by': node, 'text': 'Check the gate.',
            'model': MODEL, 'mode': 'inference'}


class DispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuous_vision_recycles_after_four_generation_attempts(self):
        calls = []
        async def request(req):
            if req.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': VISION_MODEL, 'size_vram': 1000}]})
            payload = json.loads(req.content)
            calls.append(payload['keep_alive'])
            return httpx.Response(200, json={'done': True, 'response': 'Observed.'})
        buffer = io.BytesIO()
        Image.new('RGB', (16, 16)).save(buffer, format='JPEG')
        frame = Event(event_id='frame', source='webcam', zone='desk',
                      image_b64=base64.b64encode(buffer.getvalue()).decode())
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('a', client)
            for i in range(4):
                await station.infer(frame.model_copy(update={'event_id': f'frame-{i}'}))
            await station.infer(frame.model_copy(update={'event_id': 'frame-3'}))
            self.assertEqual(calls, ['30s'] * 4, 'Cached results must not reload the engine')
            await station.infer(frame.model_copy(update={'event_id': 'frame-4'}))
            self.assertEqual(calls, ['30s'] * 4 + [0, '30s'])
            self.assertEqual(station.runner_requests, 1)

    async def test_failed_vision_generations_also_consume_runner_budget(self):
        calls = []
        async def request(req):
            payload = json.loads(req.content)
            calls.append(payload['keep_alive'])
            if payload['keep_alive'] == 0:
                return httpx.Response(200, json={'done': True})
            return httpx.Response(500, json={'error': 'generation failed'})
        buffer = io.BytesIO()
        Image.new('RGB', (16, 16)).save(buffer, format='JPEG')
        frame = Event(event_id='frame', source='webcam', zone='desk',
                      image_b64=base64.b64encode(buffer.getvalue()).decode())
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('a', client)
            for _ in range(5):
                with self.assertRaises(httpx.HTTPStatusError):
                    await station.infer(frame)
            self.assertEqual(calls, ['30s'] * 4 + [0, '30s'])
            self.assertFalse(station.busy)

    async def test_model_switch_unloads_before_loading_and_keeps_idle_residency_bounded(self):
        resident, calls = set(), []
        async def request(req):
            if req.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [
                    {'name': name, 'size_vram': 1000} for name in resident]})
            payload = json.loads(req.content)
            calls.append(payload)
            if payload['keep_alive'] == 0:
                resident.discard(payload['model'])
                return httpx.Response(200, json={'done': True})
            self.assertFalse(resident - {payload['model']}, 'Old model still resident')
            resident.add(payload['model'])
            self.assertEqual(payload['keep_alive'], '30s')
            self.assertEqual(payload['options']['num_ctx'], 2048)
            return httpx.Response(200, json={'done': True, 'response': 'Observed.'})
        buffer = io.BytesIO()
        Image.new('RGB', (16, 16)).save(buffer, format='JPEG')
        frame = Event(event_id='frame', source='webcam', zone='desk',
                      image_b64=base64.b64encode(buffer.getvalue()).decode())
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('a', client)
            await station.infer(event())
            await station.infer(frame)
            await station.infer(event('text-again'))
        self.assertEqual([(p['model'], p['keep_alive']) for p in calls],
                         [(MODEL, '30s'), (MODEL, 0), (VISION_MODEL, '30s'),
                          (VISION_MODEL, 0), (MODEL, '30s')])

    async def test_failed_unload_does_not_start_another_model(self):
        calls = []
        async def request(req):
            calls.append(json.loads(req.content))
            return httpx.Response(503, json={'error': 'unload unavailable'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('a', client)
            station.loaded_model = VISION_MODEL
            with self.assertRaises(httpx.HTTPStatusError):
                await station.infer(event())
            self.assertFalse(station.busy)
            self.assertEqual(station.loaded_model, VISION_MODEL)
        self.assertEqual(calls, [{'model': VISION_MODEL, 'stream': False, 'keep_alive': 0}])

    async def test_startup_releases_text_model_before_vision_warmup(self):
        import tempfile
        calls, resident = [], set()
        async def request(req):
            if req.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [
                    {'name': name, 'size_vram': 1000} for name in resident]})
            payload = json.loads(req.content)
            calls.append(payload)
            if payload['keep_alive'] == 0:
                resident.discard(payload['model'])
            else:
                self.assertFalse(resident - {payload['model']})
                self.assertEqual(payload['options']['num_ctx'], 2048)
                self.assertEqual(payload['keep_alive'], '30s')
                resident.add(payload['model'])
            return httpx.Response(200, json={'done': True, 'response': 'Ready.'})
        client = httpx.AsyncClient(transport=httpx.MockTransport(request))
        with tempfile.TemporaryDirectory() as data:
            with patch.dict('os.environ', {'SIMULATE': '0', 'DISCOVERY': '0', 'DATA_DIR': data,
                                           'OLLAMA_URL': 'http://engine', 'PEERS': ''}):
                with patch('app.httpx.AsyncClient', return_value=client):
                    async with lifespan(app):
                        self.assertEqual(resident, {VISION_MODEL})
                        self.assertEqual(app.state.station.runner_requests, 1)
        self.assertEqual([(p['model'], p['keep_alive']) for p in calls],
                         [(MODEL, '30s'), (MODEL, '30s'), (MODEL, 0), (VISION_MODEL, '30s')])

    async def test_concurrent_dispatch_skips_reserved_worker(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def local(e):
            entered.set()
            await release.wait()
            return result(e, 'a')
        async def remote(req):
            e = Event(**json.loads(req.content))
            return httpx.Response(200, json=result(e, 'b'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(remote)) as client:
            fleet = Fleet('a', client, local)
            fleet.peers['b'] = 'http://10.99.0.2:18090'
            first = asyncio.create_task(fleet.dispatch(event()))
            await entered.wait()
            fleet.cursor = 0
            second = await fleet.dispatch(event('second'))
            self.assertEqual(second['served_by'], 'b')
            self.assertEqual(second['attempts'][0]['error'], 'already in flight')
            release.set()
            await first
            self.assertEqual(fleet.inflight, set())

    async def test_live_route_then_retry_and_local_return(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def request(req):
            entered.set()
            await release.wait()
            raise httpx.ConnectError('unreachable', request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('a', client, simulate=True)
            endpoint = 'http://10.99.0.2:18090'
            station.discovered['b'] = {'node': 'b', 'endpoint': endpoint}
            station.fleet.peers['b'] = endpoint
            station.fleet.cursor = 1
            job = station.submit(event())
            task = asyncio.create_task(station.consume())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                live = public_job(job)
                self.assertEqual(live['status'], 'running')
                self.assertEqual(live['flow']['node'], 'b')
                self.assertEqual(live['flow']['endpoint'], endpoint)
                self.assertEqual(live['flow']['phase'], 'dispatched')
                release.set()
                await asyncio.wait_for(station.queue.join(), 2)
                self.assertEqual(job['status'], 'done')
                self.assertEqual([f['phase'] for f in job['flow_history']],
                                 ['dispatched', 'retrying', 'dispatched', 'returned'])
                self.assertEqual(job['flow']['node'], 'a')
                self.assertEqual(job['flow']['endpoint'], 'local')
                self.assertGreaterEqual(job['flow']['at'], job['received_at'])
            finally:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

    async def test_real_engine_receives_image_and_returns_frame_proof(self):
        buffer = io.BytesIO()
        Image.new('RGB', (64, 64), (10, 20, 30)).save(buffer, format='JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        calls = []
        async def request(req):
            if req.url.path == '/api/generate':
                payload = json.loads(req.content)
                calls.append(payload)
                return httpx.Response(200, json={'done': True, 'response': 'A dark image.'})
            return httpx.Response(200, json={'models': [{'name': VISION_MODEL, 'size_vram': 1000}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            station = Station('vision-worker', client)
            frame = Event(event_id='frame-1', source='webcam', zone='desk', image_b64=encoded)
            result = await station.infer(frame)
            self.assertEqual(calls[0]['images'], [encoded])
            self.assertEqual(calls[0]['model'], VISION_MODEL)
            self.assertEqual(result['image_sha256'], hashlib.sha256(buffer.getvalue()).hexdigest())
            self.assertEqual(result['mode'], 'inference')
            job = station.submit(frame)
            self.assertTrue(public_job(job)['has_image'])
            self.assertNotIn('image_b64', public_job(job)['event'])

    async def test_vision_result_cannot_skip_the_image(self):
        buffer = io.BytesIO()
        Image.new('RGB', (16, 16)).save(buffer, format='JPEG')
        frame = Event(event_id='frame', source='webcam', zone='desk',
                      image_b64=base64.b64encode(buffer.getvalue()).decode())
        async def local(e):
            return {**result(e), 'model': VISION_MODEL}  # Missing frame hash.
        async with httpx.AsyncClient() as client:
            with self.assertRaises(RuntimeError):
                await Fleet('a', client, local).dispatch(frame)

    async def test_failed_remote_falls_back_locally(self):
        async def local(e):
            return result(e, 'local')
        async def request(req):
            raise httpx.ConnectError('worker unplugged', request=req)
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            fleet = Fleet('local', client, local)
            fleet.peers['peer'] = 'http://10.99.0.2:18090'
            fleet.cursor = 1
            answer = await fleet.dispatch(event())
            self.assertEqual(answer['served_by'], 'local')
            self.assertEqual(len(answer['attempts']), 1)
            self.assertIn('10.99.0.2', answer['attempts'][0]['endpoint'])

    async def test_round_robin_uses_resolved_endpoint(self):
        calls = []
        async def local(e):
            return result(e, 'local')
        async def request(req):
            calls.append(str(req.url))
            return httpx.Response(200, json=result(event('second')))
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            fleet = Fleet('local', client, local)
            fleet.peers['name'] = 'http://10.99.4.2:18090'
            self.assertEqual((await fleet.dispatch(event()))['served_by'], 'local')
            self.assertEqual((await fleet.dispatch(event('second')))['served_by'], 'remote')
            self.assertEqual(calls, ['http://10.99.4.2:18090/api/infer'])

    async def test_gpu_slot_and_idempotence(self):
        async with httpx.AsyncClient() as client:
            station = Station('a', client, simulate=True)
            first = asyncio.create_task(station.infer(event()))
            await asyncio.sleep(0.02)
            with self.assertRaises(HTTPException) as error:
                await station.infer(event('other'))
            self.assertEqual(error.exception.status_code, 503)
            answer = await first
            self.assertEqual(await station.infer(event()), answer)
            with self.assertRaises(HTTPException):
                await station.infer(event().model_copy(update={'motion': False}))

    async def test_queue_backpressure_and_duplicate(self):
        async with httpx.AsyncClient() as client:
            station = Station('a', client, simulate=True)
            first = station.submit(event())
            self.assertIs(first, station.submit(event()))
            for i in range(31):
                station.submit(event(f'e-{i}'))
            with self.assertRaises(HTTPException) as error:
                station.submit(event('full'))
            self.assertEqual(error.exception.status_code, 429)
            self.assertNotIn('full', station.jobs)

    async def test_all_failed_is_not_a_success(self):
        async def local(e):
            raise HTTPException(503, 'busy')
        async with httpx.AsyncClient() as client:
            fleet = Fleet('a', client, local)
            with self.assertRaises(RuntimeError):
                await fleet.dispatch(event())


class DiscoveryTests(unittest.TestCase):
    def test_txt_private_address_never_overrides_mesh_resolution(self):
        info = ServiceInfo(SERVICE, 'Remote.' + SERVICE,
                           addresses=[socket.inet_aton('10.99.1.8')], port=18090,
                           properties={'demo': 'mesh-ranger-v2', 'node': 'B', 'mode': 'inference',
                                       'origin': '172.18.0.2:8090'}, server='remote.local.')
        peer = resolved_peer(info, 'A', False)
        self.assertEqual(peer['endpoint'], 'http://10.99.1.8:18090')
        self.assertEqual(peer['origin'], '172.18.0.2:8090')
        self.assertTrue(peer['mesh_address'])
        self.assertIsNone(resolved_peer(info, 'B', False))
        self.assertIsNone(resolved_peer(info, 'A', True))

    def test_invalid_measurements(self):
        for values in ({'temperature_c': float('nan')}, {'temperature_c': 150}, {'motion': 'yes'}):
            with self.assertRaises(ValidationError):
                Event(event_id='a', source='test', zone='gate', **values)

    def test_invalid_frames(self):
        for encoded in ('not-base64!', base64.b64encode(b'not a JPEG').decode(), 'a'*32001):
            with self.assertRaises(ValidationError):
                Event(event_id='a', source='webcam', zone='desk', image_b64=encoded)


class APITests(unittest.TestCase):
    def test_requests_and_body_limit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as data:
            with patch.dict('os.environ', {'SIMULATE': '1', 'DISCOVERY': '0', 'DATA_DIR': data}):
                with TestClient(app) as client:
                    self.assertEqual(client.get('/health').status_code, 200)
                    self.assertEqual(client.post('/api/events', json=event().model_dump()).status_code, 202)
                    self.assertEqual(client.post('/api/events', json={'motion': True}).status_code, 422)
                    self.assertEqual(client.post('/api/events', content=b'x'*36001).status_code, 413)
                    self.assertEqual(client.post('/api/advertisement', json={'enabled': False}).status_code, 409)
                    self.assertEqual(client.get('/api/state').json()['mode'], 'simulation')


if __name__ == '__main__':
    unittest.main()
