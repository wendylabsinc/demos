"""Package one terminal trial's complete events and bounded exact RGB-D evidence.

No actuator, SDK, camera lease, or network client is constructed here. The caller
supplies a native event file and explicit time interval. Images are selected for
bounded local transfer; every archived frame's source remains listed in events.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np


def create_export(events_path,start_ns,end_ns,output,app_root=Path('/app')):
    events_path=Path(events_path);root=events_path.parent.resolve();output=Path(output)
    if start_ns>end_ns:raise ValueError('Invalid trial interval')
    events=[]
    with events_path.open() as src:
        for line in src:
            row=json.loads(line)
            if start_ns<=row['unix_ns']<=end_ns:events.append(row)
    cameras=[e for e in events if e['phase']=='phase_camera_evidence']
    # Scene/planning baseline and the final three exposures bracket failures.
    selected=[]
    for row in cameras[:2]+cameras[-3:]:
        if row['path'] not in selected:selected.append(row['path'])
    files={};members=[]
    def member(name,raw):
        if len(raw)>16000000:raise RuntimeError('Trial evidence member exceeds 16 MB')
        members.append((name,raw));files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    member('events.jsonl.gz',gzip.compress((''.join(json.dumps(e,separators=(',',':'))+'\n' for e in events)).encode(),mtime=0))
    for index,name in enumerate(selected):
        path=Path(name).resolve()
        if path.parent!=root:raise RuntimeError('Camera evidence outside controller data directory')
        for suffix in ('.source.zip','.timing.npz'):
            source=path.with_suffix(suffix)
            member('rgbd/'+source.name,source.read_bytes())
        if index==0:
            with np.load(path,allow_pickle=False) as data:
                rays=data['rays'].copy();metadata=data['metadata_json'].copy()
            bundle=io.BytesIO();np.savez_compressed(bundle,rays=rays,metadata_json=metadata)
            member('rgbd/rays.npz',bundle.getvalue())
    sources={}
    for name in ('runtime.py','camera_input.py','physical_core.py','table_plane.py','pickup.py','target_tracker.py'):
        path=Path(app_root)/name
        if path.exists():
            raw=path.read_bytes();sources[name]=hashlib.sha256(raw).hexdigest()
            member('source/'+name,raw)
    manifest=dict(schema='wendy.g1.trial-export.v1',start_unix_ns=start_ns,end_unix_ns=end_ns,
        native_event_file=str(events_path),events=len(events),camera_events=len(cameras),
        selected_frames=selected,files=files,source_hashes=sources,
        primary_failure=next((e for e in events if e['phase']=='primary_failure'),None))
    member('manifest.json',json.dumps(manifest,indent=2).encode())
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix(output.suffix+'.pending')
    with tarfile.open(pending,'w:gz') as tar:
        for name,raw in members:
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
            tar.addfile(info,io.BytesIO(raw))
    if pending.stat().st_size>32000000:raise RuntimeError('Trial export exceeds 32 MB transfer budget')
    pending.replace(output)
    return dict(path=str(output),bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        events=len(events),selected_frames=len(selected),primary_failure=manifest['primary_failure'])
