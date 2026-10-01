"""Build-time, hash-checked speech/language/runtime provisioning. No runtime downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen


def fetch(url,path,sha256):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==sha256:return
    temporary=path.with_suffix(path.suffix+'.part')
    digest=hashlib.sha256()
    with urlopen(url,timeout=120) as response,temporary.open('wb') as output:
        while chunk:=response.read(1024*1024):digest.update(chunk);output.write(chunk)
    if digest.hexdigest()!=sha256:
        temporary.unlink();raise ValueError(f'Checksum mismatch for {path.name}')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/opt/models')
    parser.add_argument('--runtime',default='/opt/llama');args=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parent.parent/'models.lock.json').read_text())
    for asset in manifest['assets']:
        fetch(asset['url'],Path(args.root)/asset['path'],asset['sha256'])
    runtime=manifest['runtime'];archive=Path('/tmp/yard-llama.tar.gz')
    fetch(runtime['url'],archive,runtime['sha256'])
    dest=Path(args.runtime);dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            relative=Path(*Path(member.name).parts[1:])
            if not relative.parts:continue
            if '..' in relative.parts or relative.is_absolute():raise ValueError('Invalid runtime archive path')
            member.name=str(relative);tar.extract(member,dest,filter='data')
    archive.unlink()
if __name__=='__main__':main()
