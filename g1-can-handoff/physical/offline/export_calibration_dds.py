"""Export a completed calibration trial's sealed DDS interval without motion."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from archive_trial_recording import archive
from operator_panel import Wendy


def export(path):
    path=Path(path);result=json.loads(path.read_text())
    if result.get('state')!='terminal' or not result.get('control_released_verified'):
        raise ValueError('Export requires terminal recorded release proof')
    if result.get('dds_export'):return result['dds_export']
    before=result['before'];after=result['after']
    if before['recorder']['session']!=after['recorder']['session']:
        raise ValueError('Recorder session changed during trial')
    evidence=dict(recorder_after=after['recorder'],started_unix_ns=before['unix_ns'],finished_unix_ns=after['unix_ns'])
    w=Wendy()
    try:
        parts=[]
        start=evidence['started_unix_ns'];end=evidence['finished_unix_ns'];width=45_000_000_000
        for index,begin in enumerate(range(start,end+1,width)):
            interval={**evidence,'started_unix_ns':begin,'finished_unix_ns':min(end,begin+width-1)}
            destination=path.with_name(f'{path.stem}.dds-{index:02d}.tar.gz')
            metadata=archive(w,interval,destination)
            parts.append({**metadata,'start_unix_ns':begin,'end_unix_ns':interval['finished_unix_ns']})
            result['dds_export_parts_pending']=parts
            path.write_text(json.dumps(result,indent=2)+'\n')
        result['dds_export']={'parts':parts,'coverage':'Contiguous requested timestamp intervals, all callback-delivered DDS records; upstream loss unknown'}
        result.pop('dds_export_parts_pending',None)
        result.pop('dds_export_error',None)
    except Exception as e:
        result['dds_export_error']=str(e)
        raise
    finally:
        path.write_text(json.dumps(result,indent=2)+'\n');w.process.terminate()
    return result['dds_export']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trial',nargs='+');a=p.parse_args()
    for path in a.trial:print(json.dumps(export(path),indent=2),flush=True)
