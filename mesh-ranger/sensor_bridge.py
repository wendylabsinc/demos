"""Forward newline-delimited sensor JSON from stdin to a local Ranger station.

Run inside a mesh-entitled gateway or use a verified local port forward.
Input example: {"source":"lite-gate","zone":"north gate","motion":true}
"""
import argparse
import json
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8090')
    args = parser.parse_args()
    client = build_opener(ProxyHandler({}))
    for line in sys.stdin:
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError('Expected a JSON object')
            event.setdefault('event_id', str(uuid.uuid4()))
            body = json.dumps(event).encode()
            if len(body) > 4096:
                raise ValueError('Event exceeds 4096 bytes')
        except (ValueError, TypeError) as error:
            print(f'Skipping invalid sensor line: {error}', file=sys.stderr)
            continue
        for attempt in range(3):
            try:
                request = Request(args.url.rstrip('/') + '/api/events', data=body,
                                  headers={'Content-Type': 'application/json'})
                with client.open(request, timeout=10) as response:
                    print(response.read().decode(), flush=True)
                break
            except (HTTPError, URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and 400 <= error.code < 500 and error.code != 429:
                    raise SystemExit(f'Sensor event rejected: {error}') from error
                if attempt == 2:
                    raise SystemExit(f'Sensor event was not accepted: {error}') from error
                time.sleep(2)


if __name__ == '__main__':
    main()
