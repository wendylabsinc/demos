"""The offline runner verifies evidence and blocks accidental device access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from offline import run

class OfflinePackTests(unittest.TestCase):
    def test_missing_or_modified_fixture_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);offline=root/'offline';offline.mkdir()
            artifact=root/'frame.bin';artifact.write_bytes(b'original capture')
            (offline/'manifest.json').write_text(json.dumps({'files':{'frame.bin':hashlib.sha256(artifact.read_bytes()).hexdigest()}}))
            with patch.object(run,'HERE',offline),patch.object(run,'PHYSICAL',root):
                run.integrity()
                artifact.write_bytes(b'changed capture')
                with self.assertRaisesRegex(RuntimeError,'changed captured evidence'):run.integrity()
                artifact.unlink()
                with self.assertRaisesRegex(RuntimeError,'Missing'):run.integrity()

    def test_spawned_child_rejects_robot_and_existing_loopback_connections_and_real_dds(self):
        code='''import socket
for address in [('10.10.20.238',8102),('127.0.0.1',8102)]:
    with socket.socket() as sock:
        try:sock.connect(address)
        except RuntimeError as error:assert 'Offline regression' in str(error)
        else:raise AssertionError('Live endpoint was reachable')
try:import unitree_sdk2py
except ImportError as error:assert 'disabled in offline' in str(error)
else:raise AssertionError('Real DDS was imported')
'''
        env=dict(os.environ,HANDOFF_OFFLINE='1',PYTHONPATH=str(run.HERE))
        result=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
