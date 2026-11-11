"""Run three explicitly simulated workers on loopback; no model or mDNS claims."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).parent
processes = []
try:
    for index, port in enumerate((8090, 8091, 8092)):
        peers = ','.join(f'http://127.0.0.1:{other}' for other in (8090, 8091, 8092) if other != port)
        env = dict(os.environ, SIMULATE='1', DISCOVERY='0', NODE_NAME=f'rehearsal-{index+1}',
                   DATA_DIR=str(ROOT / '.local' / str(port)), PEERS=peers, PORT=str(port))
        processes.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app:app',
                                           '--host', '127.0.0.1', '--port', str(port)], cwd=ROOT, env=env))
    print('Open http://127.0.0.1:8090 . All responses are SIMULATED; mDNS is disabled.', flush=True)
    while all(process.poll() is None for process in processes):
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
