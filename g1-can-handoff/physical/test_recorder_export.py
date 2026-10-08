import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from native_recorder_export import create_recording_export


class RecorderExportTests(unittest.TestCase):
    def fixture(self,root):
        sealed=[];rows=[]
        for i in range(4):
            path=root/f'samples-{i:06d}.jsonl.gz'
            segment=[dict(unix_ns=i*10+j,topic=topic,receive_sequence=i*4+j,storage_codec='dds-cdr-base64/1',cdr_base64='AAE=')
                     for j,topic in enumerate(['rt/lowstate','rt/arm_sdk','rt/dex3/left/cmd','rt/dex3/right/cmd'])]
            rows.extend(segment);data=(''.join(json.dumps(e)+'\n' for e in segment)).encode()
            path.write_bytes(gzip.compress(data,mtime=0))
            if i<3:sealed.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
        return dict(session='fixture',current_file=str(path),closed_segments=sealed),rows

    def test_exports_every_recorded_topic_in_interval_and_retains_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);status,rows=self.fixture(root)
            info=create_recording_export(status,10,23,root/'trial.tar.gz')
            with tarfile.open(info['path']) as tar:
                saved=[json.loads(line) for line in gzip.decompress(tar.extractfile('samples.jsonl.gz').read()).splitlines()]
                manifest=json.load(tar.extractfile('manifest.json'))
            self.assertEqual(saved,[e for e in rows if 10<=e['unix_ns']<=23])
            self.assertEqual(info['records'],8)
            self.assertEqual(manifest['topics'],{'rt/lowstate':2,'rt/arm_sdk':2,'rt/dex3/left/cmd':2,'rt/dex3/right/cmd':2})
            self.assertIn('upstream DDS loss is unknown',manifest['coverage'])

    def test_unsealed_tail_is_rejected_rather_than_silently_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);status,_=self.fixture(root)
            with self.assertRaisesRegex(RuntimeError,'not fully sealed'):
                create_recording_export(status,10,31,root/'trial.tar.gz')
            self.assertFalse((root/'trial.tar.gz').exists())

    def test_missing_start_and_corrupted_sealed_segment_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);status,_=self.fixture(root)
            with self.assertRaisesRegex(RuntimeError,'not fully sealed'):
                create_recording_export(status,-1,23,root/'trial.tar.gz')
            status['closed_segments'][1]['sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                create_recording_export(status,10,23,root/'trial.tar.gz')

    def test_out_of_order_callbacks_across_rotation_are_kept_in_original_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);status,_=self.fixture(root)
            path=root/'samples-000000.jsonl.gz'
            raw=gzip.decompress(path.read_bytes())+b'{"unix_ns":11,"topic":"late-callback","cdr_base64":"AAE="}\n'
            path.write_bytes(gzip.compress(raw,mtime=0));status['closed_segments'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            info=create_recording_export(status,10,23,root/'trial.tar.gz')
            with tarfile.open(info['path']) as tar:
                saved=[json.loads(line) for line in gzip.decompress(tar.extractfile('samples.jsonl.gz').read()).splitlines()]
            self.assertEqual(saved[0]['topic'],'late-callback');self.assertEqual(info['records'],9)


if __name__=='__main__':unittest.main()
