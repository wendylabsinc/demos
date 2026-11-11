"""Verify and preserve locally downloaded demo evidence; no robot connection."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def preserve(directory, destination):
    saved=[]
    for summary in sorted(directory.glob('trial-??.json')):
        row=json.loads(summary.read_text())
        archive=directory/(summary.stem+'.tar.gz')
        expected=row['artifact']
        raw=archive.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('Trial archive checksum mismatch: '+str(archive))
        target=destination/summary.stem
        with tarfile.open(archive) as tar:
            manifest=json.load(tar.extractfile('manifest.json'))
            payloads={}
            for name,metadata in manifest['files'].items():
                path=PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe artifact member: '+name)
                content=tar.extractfile(name).read()
                if len(content)!=metadata['bytes'] or hashlib.sha256(content).hexdigest()!=metadata['sha256']:
                    raise ValueError('Artifact member checksum mismatch: '+name)
                payloads[name]=content
            payloads['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
            payloads['trial.json']=summary.read_bytes()
            for name,content in payloads.items():
                output=target/name
                if output.exists() and output.read_bytes()!=content:
                    raise ValueError('Refusing to overwrite different evidence: '+str(output))
                output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(content)
        saved.append(dict(trial=row['index'],archive_sha256=expected['sha256'],
                          files=len(payloads),events=manifest['events']))
    return saved


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(preserve(args.directory,args.destination),indent=2))
