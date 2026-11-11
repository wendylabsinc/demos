"""Loopback dashboard through authenticated Wendy LAN management.

Browser traffic uses a persistent authenticated Wendy app forward.
Peer inference still uses Wendy Net.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import subprocess
import socket
import threading
import signal
from collections import deque
import urllib.request
import urllib.error
from urllib.parse import urlsplit, parse_qs
from sensorlink_camera import SensorLinkCamera

class WendyForward:
    """One authenticated CLI connection per preview, shared by HTTP requests."""
    def __init__(self, device):
        self.device = device
        self.process = None
        self.lock = threading.Lock()
        self.port = None
        self.messages = deque(maxlen=12)
        self.ready = threading.Event()
        self.opened = False

    def _read_output(self, process, ready):
        with process.stdout:
            for line in process.stdout:
                self.messages.append(line.strip())
                if line.startswith('Forwarding '):
                    self.opened = True
                    ready.set()
        ready.set()

    def ensure(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.opened:
                return self.port
            self._stop()
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                self.port = reservation.getsockname()[1]
            self.messages.clear()
            self.ready = threading.Event()
            self.opened = False
            self.process = subprocess.Popen(
                ['wendy', 'net', 'forward', f'{self.port}:18090', '--device', self.device],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=self._read_output,
                             args=(self.process, self.ready), daemon=True).start()
            if not self.ready.wait(10) or not self.opened:
                self._stop()
                raise ConnectionError('The authenticated app forward could not start.')
            return self.port

    def _stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        self.opened = False

    def close(self):
        with self.lock:
            self._stop()

    def request(self, method, path, body):
        port = self.ensure()
        request = urllib.request.Request(f'http://127.0.0.1:{port}'+path,
            data=body if method == 'POST' else None,
            headers={'Content-Type': 'application/json'}, method=method)
        try:
            response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=12)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, urllib.error.URLError):
            self.close()
            raise ConnectionError('The authenticated app forward was interrupted.') from None
        with response:
            return response.status, response.read()


def connection_failure(stderr='', timed_out=False):
    """Keep CLI diagnostics out of the dashboard's primary message."""
    code = 'station_timeout' if timed_out else 'station_unavailable'
    diagnostic = stderr.strip()
    for line in reversed(stderr.splitlines()):
        try:
            failure = json.loads(line).get('error', {})
        except (ValueError, AttributeError):
            continue
        if isinstance(failure, dict) and failure.get('code'):
            code = failure['code']
            diagnostic = failure.get('message', code)
            break
    return {'code': code, 'detail': 'The station is temporarily unavailable.',
            'diagnostic': diagnostic[-4000:] if diagnostic else
            'The authenticated device request timed out.' if timed_out else
            'The authenticated device request did not complete.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--port', type=int, default=19090)
    parser.add_argument('--sensorlink-device', help='Explicit Wendy Lite selector for a SensorLink MJPEG camera')
    args = parser.parse_args()
    forward = WendyForward(args.device)
    camera = SensorLinkCamera(args.sensorlink_device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                return self.reply(200, Path(__file__).with_name('index.html').read_bytes(), 'text/html; charset=utf-8')
            if self.path.startswith('/api/camera/'):
                return self.camera_request()
            self.forward()

        def do_POST(self):
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, b'{"detail":"Use application/json"}')
            if self.path.startswith('/api/camera/'):
                return self.camera_request()
            self.forward()

        def camera_request(self):
            # The preview controls a physical camera. Accept only this loopback
            # dashboard's origin, not a web page issuing cross-origin requests.
            origin = self.headers.get('Origin')
            if origin and origin not in {f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'}:
                return self.reply(403, b'{"detail":"Use the local dashboard."}')
            parsed = urlsplit(self.path)
            try:
                if self.command == 'GET' and parsed.path == '/api/camera/sources':
                    return self.reply(200, json.dumps(camera.sources()).encode())
                if self.command == 'GET' and parsed.path == '/api/camera/frame':
                    after = int(parse_qs(parsed.query).get('after', ['0'])[0])
                    frame, sequence = camera.frame(after)
                    return self.reply(200 if frame else 204, frame or b'', 'image/jpeg', {'X-Frame-Sequence': sequence})
                if self.command == 'POST' and parsed.path in {'/api/camera/start', '/api/camera/stop'}:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 4096 or self.headers.get('Transfer-Encoding'):
                        return self.reply(413, b'{}')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError('Expected a JSON object.')
                    if parsed.path.endswith('/start'):
                        result = camera.start(body.get('channel'))
                    else:
                        camera.close()
                        result = {'stopped': True}
                    return self.reply(200, json.dumps(result).encode())
                return self.reply(404, b'{}')
            except (ConnectionError, subprocess.TimeoutExpired, OSError) as error:
                return self.reply(409, json.dumps({'detail': str(error)[:240]}).encode())
            except (ValueError, KeyError, TypeError) as error:
                return self.reply(400, json.dumps({'detail': str(error)[:240]}).encode())

        def forward(self):
            allowed = {'GET': {'/api/state', '/health'}, 'POST': {'/api/events', '/api/advertisement'}}
            frame_path = self.command == 'GET' and re.fullmatch(r'/api/frames/[a-zA-Z0-9_.-]{1,64}', self.path)
            if self.path not in allowed[self.command] and not frame_path:
                return self.reply(404, b'{}')
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 <= length <= 36000 or self.headers.get('Transfer-Encoding'):
                    return self.reply(413, b'{}')
                body = self.rfile.read(length)
                status, result = forward.request(self.command, self.path, body)
                self.reply(status, result)
            except (ConnectionError, urllib.error.URLError):
                self.reply(502, json.dumps(connection_failure('\n'.join(forward.messages))).encode())
            except (ValueError, KeyError, OSError):
                self.reply(502, json.dumps(connection_failure()).encode())

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f'Open http://127.0.0.1:{args.port} for {args.device}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        forward.close()
        camera.close()


if __name__ == '__main__':
    main()
