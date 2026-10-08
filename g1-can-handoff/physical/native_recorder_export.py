"""Preserve all callback-delivered DDS records in a sealed trial time interval."""
import collections
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def create_recording_export(status,start_ns,end_ns,output):
    root=Path(status['current_file']).parent
    if not start_ns<=end_ns:raise ValueError('Invalid recording interval')
    segments=[]
    # Only sealed files are read in full. The live file is used solely as a
    # right boundary proving no part of the requested interval is still open.
    for entry in status['closed_segments']+[dict(file=Path(status['current_file']).name)]:
        name=entry['file']
        if Path(name).name!=name:raise ValueError('Invalid recorder segment name')
        path=root/name
        with gzip.open(path,'rt') as src:first=json.loads(next(src))['unix_ns']
        segments.append((first,path,entry))
    if not segments or segments[0][0]>start_ns or segments[-1][0]<=end_ns:
        raise RuntimeError('Trial recording interval is not fully sealed yet')
    if any(a[0]>b[0] for a,b in zip(segments,segments[1:])):
        raise RuntimeError('Recorder segment timestamps did not advance')
    # Include one neighboring sealed segment on both sides to preserve tiny
    # callback reorderings around a rotation boundary.
    overlaps=[i for i,(first,_,_) in enumerate(segments[:-1]) if first<=end_ns and segments[i+1][0]>=start_ns]
    selected=segments[max(0,min(overlaps)-1):min(len(segments)-1,max(overlaps)+2)]
    counts=collections.Counter();sealed=[];raw=io.BytesIO();first_record=None;last_record=None
    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=3) as saved:
        for _,path,entry in selected:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                raise RuntimeError('Sealed recorder checksum mismatch: '+path.name)
            sealed.append(entry)
            with gzip.open(path,'rb') as src:
                for line in src:
                    row=json.loads(line)
                    if start_ns<=row['unix_ns']<=end_ns:
                        saved.write(line);counts[row['topic']]+=1
                        first_record=row['unix_ns'] if first_record is None else min(first_record,row['unix_ns'])
                        last_record=row['unix_ns'] if last_record is None else max(last_record,row['unix_ns'])
    if not counts:raise RuntimeError('No recorded DDS samples in trial interval')
    data=raw.getvalue()
    if len(data)>64000000:raise RuntimeError('Trial recording exceeds 64 MB transfer budget')
    manifest=dict(schema='wendy.g1.trial-dds-export.v1',session=status['session'],
        start_unix_ns=start_ns,end_unix_ns=end_ns,first_record_unix_ns=first_record,last_record_unix_ns=last_record,
        records=sum(counts.values()),topics=dict(counts),sealed_segments=sealed,
        coverage='All callback-delivered records in the requested interval; upstream DDS loss is unknown',
        files={'samples.jsonl.gz':dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())})
    members=[('samples.jsonl.gz',data)]
    # Decoder is provenance only; no SDK is imported here.
    decoder=Path('/app/export_cdr.py')
    if decoder.is_file():
        source=decoder.read_bytes();members.append(('export_cdr.py',source))
        manifest['files']['export_cdr.py']=dict(bytes=len(source),sha256=hashlib.sha256(source).hexdigest())
    members.append(('manifest.json',json.dumps(manifest,indent=2).encode()))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix(output.suffix+'.pending')
    with tarfile.open(pending,'w:gz') as tar:
        for name,data in members:
            member=tarfile.TarInfo(name);member.size=len(data);member.mtime=0
            tar.addfile(member,io.BytesIO(data))
    pending.replace(output)
    return dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        records=manifest['records'],topics=manifest['topics'],session=status['session'])
