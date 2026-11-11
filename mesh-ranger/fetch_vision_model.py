"""Build-time-only download of a pinned Ollama vision model and verified blobs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

REGISTRY = 'https://registry.ollama.ai/v2/library/moondream'
MANIFEST_SHA = '55fc3abd386771e5b5d1bbcc732f3c3f4df6e9f9f08f1131f9cc27ba2d1eec5b'
ROOT = Path('/opt/vision-models')


def fetch_blob(descriptor):
    digest = descriptor['digest']
    target = ROOT / 'blobs' / digest.replace(':', '-')
    hasher = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(REGISTRY + '/blobs/' + digest, timeout=120) as response, target.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    if hasher.hexdigest() != digest.split(':')[1] or size != descriptor['size']:
        raise RuntimeError('Model blob integrity check failed: ' + digest)
    print('Verified', digest, size, flush=True)


def main():
    raw = urllib.request.urlopen(REGISTRY + '/manifests/1.8b', timeout=60).read()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise RuntimeError('Vision manifest changed; review the new revision before updating the pin')
    manifest = json.loads(raw)
    (ROOT / 'blobs').mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch_blob, [manifest['config'], *manifest['layers']]))
    target = ROOT / 'manifests/registry.ollama.ai/library/moondream/1.8b'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


if __name__ == '__main__':
    main()
