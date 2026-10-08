"""A failed physical trial must remain replayable after the robot is gone."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from native_trial_export import create_export
from trial_evidence import export_trial, result_payload


class EvidenceTests(unittest.TestCase):
    def fixture(self, root):
        events=[]
        for i in range(7):
            path=root/f'camera-{i}-motion-tracking.npz'
            np.savez(path,rays=np.ones((2,2,3)),metadata_json=json.dumps({'frame_id':i}))
            path.with_suffix('.source.zip').write_bytes(b'exact source '+bytes([i]))
            np.savez(path.with_suffix('.timing.npz'),color_time=i)
            events.append(dict(unix_ns=i+10,phase='phase_camera_evidence',path=str(path)))
            events.append(dict(unix_ns=i+10,phase='target',q=[i]*7))
        events.append(dict(unix_ns=17,phase='primary_failure',error='exposure stale (401.7 ms)'))
        events.append(dict(unix_ns=18,phase='error',error='cancelled'))
        log=root/'events.jsonl';log.write_text(''.join(json.dumps(e)+'\n' for e in events))
        return log,events

    def test_archive_keeps_every_event_and_exact_boundary_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log,events=self.fixture(root)
            info=create_export(log,10,17,root/'trial.tar.gz',root)
            with tarfile.open(info['path']) as tar:
                manifest=json.load(tar.extractfile('manifest.json'))
                saved=[json.loads(e) for e in gzip.decompress(tar.extractfile('events.jsonl.gz').read()).splitlines()]
                self.assertEqual(saved,events[:-1])
                self.assertEqual(manifest['primary_failure']['error'],'exposure stale (401.7 ms)')
                self.assertEqual(manifest['selected_frames'],[str(root/f'camera-{i}-motion-tracking.npz') for i in [0,1,4,5,6]])
                self.assertEqual(tar.extractfile('rgbd/camera-6-motion-tracking.source.zip').read(),b'exact source '+bytes([6]))
                for name,row in manifest['files'].items():
                    raw=tar.extractfile(name).read()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])

    def test_export_rejects_camera_evidence_outside_native_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log=root/'events.jsonl'
            log.write_text(json.dumps(dict(unix_ns=1,phase='phase_camera_evidence',path='/etc/test.npz'))+'\n')
            with self.assertRaisesRegex(RuntimeError,'outside controller'):create_export(log,1,2,root/'trial.tar.gz')

    def transfer_client(self, metadata, raw):
        def tool(name,arguments):
            argv=arguments['command']
            if len(argv)==7:out=json.dumps(metadata)
            else:out=base64.b64encode(raw[int(argv[-2]):int(argv[-2])+int(argv[-1])]).decode()
            return {'structuredContent':dict(exit_code=0,stdout=out,stderr='',truncated=False)}
        return Mock(tool=Mock(side_effect=tool))

    def test_native_transfer_is_verified_before_publishing_local_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);log,_=self.fixture(root)
            metadata=create_export(log,10,17,root/'native.tar.gz',root)
            raw=(root/'native.tar.gz').read_bytes();destination=root/'local.tar.gz'
            result=export_trial(self.transfer_client(metadata,raw),{'native_event_file':'/data/events.jsonl'},10,17,destination)
            self.assertTrue(result['verified']);self.assertEqual(destination.read_bytes(),raw)
            self.assertFalse(root.joinpath('local.tar.gz.pending').exists())

    def test_short_chunk_never_leaves_an_apparently_complete_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);destination=root/'local.tar.gz'
            metadata=dict(bytes=10,sha256='0'*64)
            with self.assertRaisesRegex(RuntimeError,'Short trial artifact'):
                export_trial(self.transfer_client(metadata,b'bad'),{'native_event_file':'/data/events.jsonl'},1,2,destination)
            self.assertFalse(destination.exists());self.assertFalse(root.joinpath('local.tar.gz.pending').exists())

    def test_checksum_mismatch_rejects_even_a_complete_sized_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'local.tar.gz'
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                export_trial(self.transfer_client(dict(bytes=3,sha256='0'*64),b'bad'),{'native_event_file':'/data/events.jsonl'},1,2,destination)
            self.assertFalse(destination.exists())

    def test_native_truncation_and_failed_exec_are_not_valid_evidence(self):
        for row in [dict(exit_code=0,truncated=True),dict(exit_code=1,stderr='disk full')]:
            with self.assertRaises(RuntimeError):result_payload({'structuredContent':row})


if __name__=='__main__':unittest.main()
