#!/usr/bin/env python3
"""Serve local viewer assets and relay its API to the WendyOS VM.

QEMU user networking can refuse simultaneous TCP connects. Serving assets on the
host and serialising API connections avoids that limit. No physics runs here.
"""
import argparse
import json
import logging
import mimetypes
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--upstream", required=True)
parser.add_argument("--port", type=int, default=18884)
parser.add_argument("--state-file", type=Path, default=Path(".runtime/viewer.json"))
args = parser.parse_args()
upstream = args.upstream.rstrip("/")
parsed = urlparse(upstream)
if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path:
    parser.error("The upstream must be the VM's loopback HTTP URL.")
static = (Path(__file__).resolve().parent.parent / "static").resolve()
gate = threading.Lock()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *items):
        pass

    def send(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def relay(self, body=None):
        request = urllib.request.Request(upstream + self.path, data=body,
                                         headers={"Content-Type": "application/json"} if body is not None else {},
                                         method="POST" if body is not None else "GET")
        try:
            with gate:
                with opener.open(request, timeout=10) as response:
                    result = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            self.send(status, result, content_type)
        except urllib.error.HTTPError as error:
            self.send(error.code, error.read())
        except (OSError, urllib.error.URLError):
            self.send(502, b'{"error":"The WendyOS VM is unavailable. Start it with scripts/run-vm.sh."}')

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/viewer-health":
                self.send(200, json.dumps({"app": "robot-vacuum-viewer", "upstream": upstream}).encode())
            elif path.startswith("/api/") or path == "/health":
                self.relay()
            else:
                relative = "index.html" if path == "/" else path.removeprefix("/static/")
                file = (static / relative).resolve()
                if not file.is_relative_to(static) or not file.is_file():
                    self.send(404, b'{"error":"File not found."}')
                else:
                    self.send(200, file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            self.send(404, b'{"error":"Endpoint not found."}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError()
            self.relay(self.rfile.read(length))
        except ValueError:
            self.send(400, b'{"error":"Invalid request size."}')
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(ThreadingHTTPServer):
    request_queue_size = 64


try:
    server = Server(("127.0.0.1", args.port), Handler)
except OSError:
    if args.port == 0:
        raise
    server = Server(("127.0.0.1", 0), Handler)
url = f"http://127.0.0.1:{server.server_port}"
args.state_file.parent.mkdir(parents=True, exist_ok=True)
args.state_file.write_text(json.dumps({"pid": os.getpid(), "url": url, "upstream": upstream}) + "\n")
print(f"VM viewer ready at {url}. Physics runs in {upstream}.", flush=True)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
