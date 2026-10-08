"""Replay nine native stale stops at the real command-loop freshness seam.

Actuator sinks record calls only. Camera/geometry and native execution speed
are not simulated; source exposures, successful observations and availability
times come from immutable captured events. Availability is conservatively
bounded by the successful parent-relayed perception result and recorded child
geometry completion, never by an unvalidated frame-received event.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np

PHYSICAL=Path(__file__).resolve().parents[1]
_data=tempfile.TemporaryDirectory(prefix='stale-replay-')
with patch.dict(os.environ,{'HANDOFF_DATA':_data.name}):
    from runtime import Runtime
from physical_core import PhysicalPlant


def walk(value):
    if isinstance(value,dict):
        if 'phase' in value and 'monotonic_ns' in value:yield value
        else:
            for child in value.values():yield from walk(child)
    elif isinstance(value,list):
        for child in value:yield from walk(child)


def read_events(path):
    content=path.read_bytes()
    if path.suffix=='.gz':content=gzip.decompress(content)
    if path.name.endswith('.jsonl.gz'):return [json.loads(line) for line in content.splitlines()]
    return list(walk(json.loads(content)))


def cases():
    ranking=json.loads((PHYSICAL/'offline/validation/exposure-500ms-20261008/error-ranking.json').read_text())
    combined=json.loads(gzip.decompress((PHYSICAL/'offline/trial-events.json.gz').read_bytes()))
    result=[]
    for fault in ranking['faults']:
        if fault['category']!='Stale RGB-D observation':continue
        paths=[PHYSICAL/source for source in fault['sources']]
        if any('exposure-restart-20261008' in str(path) for path in paths):
            paths.append(PHYSICAL/'offline/evidence/exposure-restart-20261008-preserved/trial-01/events.jsonl.gz')
        rows=[]
        for path in paths:
            if path.name=='trial-events.json.gz':
                for source in fault['sources']:
                    key=Path(source).stem
                    if key in combined:rows.extend(walk(combined[key]))
            elif path.is_file():rows.extend(read_events(path))
        # Restarted devices can reuse monotonic times. Only source files
        # associated with this primary failure are combined, never all trials.
        rows=sorted({json.dumps(row,sort_keys=True):row for row in rows}.values(),key=lambda row:row['monotonic_ns'])
        stop=fault['monotonic_ns']/1e9
        rows=[row for row in rows if abs(row['monotonic_ns']/1e9-stop)<20]
        metadata={row['metadata']['frame_id']:row['metadata'] for row in rows if row['phase']=='phase_camera_evidence'}
        packets=[];pending={}
        for row in rows:
            if row['phase']=='perception_result' and row.get('geometry') and not row.get('error'):
                if row['frame_id'] in metadata:pending[row.get('phase_name')]=row
            elif row['phase']=='perception_worker_result':
                observed=pending.pop(row.get('phase_name'),None)
                if observed is None or row.get('error'):continue
                frame_id=observed['frame_id'];meta=metadata[frame_id]
                geometry_complete_at=meta['depth_time']+observed['exposure_age_ms']/1000
                validated_result_at=observed['monotonic_ns']/1e9
                packets.append(dict(frame_id=frame_id,exposure=meta['depth_time'],
                    phase=observed.get('phase_name'),legacy_at=row['monotonic_ns']/1e9,
                    receipt_at=max(validated_result_at,geometry_complete_at),metadata=meta,
                    geometry_complete_at=geometry_complete_at,validated_result_at=validated_result_at,
                    processed_age_ms=observed['exposure_age_ms'],processing_ms=observed['processing_ms']))
        measured=float(re.search(r'\(([\d.]+) ms\)',fault['error']).group(1))/1000
        old=[packet for packet in packets if packet['legacy_at']<=stop and packet['phase']=='motion-tracking']
        if not old:raise RuntimeError('Missing accepted background exposure: '+fault['error'])
        original=min(old,key=lambda packet:abs(stop-packet['exposure']-measured))
        if abs(stop-original['exposure']-measured)>.003:
            raise RuntimeError('Original stop does not match recorded exposure within3ms: '+fault['error'])
        later=[packet for packet in packets if packet['legacy_at']>stop]
        if not later:raise RuntimeError('Missing next validated packet: '+fault['error'])
        next_packet=min(later,key=lambda packet:packet['legacy_at'])
        raw_inputs={}
        for packet in (original,next_packet):
            for path in paths:
                directory=path.parent/'rgbd'
                for source in directory.glob('camera-'+str(packet['frame_id'])+'-*.source.zip'):
                    with zipfile.ZipFile(source) as archive:
                        saved=json.loads(archive.read('metadata.json'))
                    if (saved['source_frame_id']==packet['frame_id'] and
                            saved['captured_at_unix_ns']==packet['metadata']['captured_at_unix_ns']):
                        timing=source.with_name(source.name.replace('.source.zip','.timing.npz'))
                        if timing.is_file():raw_inputs[str(packet['frame_id'])]=[
                            str(source.relative_to(PHYSICAL)),str(timing.relative_to(PHYSICAL))]
        result.append(dict(fault=fault,stop=stop,original=original,packets=packets,
                           next=next_packet,raw_inputs=raw_inputs,
                           input_sha256={str(path.relative_to(PHYSICAL)):hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in paths if path.is_file()}))
    if len(result)!=9:raise RuntimeError('Expected all nine captured stale stops')
    return result


def command_tick(timestamp,at,max_age_ms,*,uncertain=None,cancelled=False):
    runtime=Runtime.__new__(Runtime);runtime.lock=threading.RLock()
    runtime.tracking_error=uncertain;runtime.tracking_observation=SimpleNamespace(timestamp=timestamp)
    now=lambda:at
    plant=PhysicalPlant.__new__(PhysicalPlant)
    plant.cancelled=cancelled;plant.cancel=threading.Event();plant.command_lock=threading.RLock()
    plant.perception_guard=runtime.tracking_guard;plant.clock=SimpleNamespace(monotonic=now,sleep=lambda seconds:None)
    plant.base_joints=np.zeros(15);plant.base_rpy=np.zeros(3);plant.right_arm=np.zeros(7)
    plant.left_hand=np.zeros(7);plant.right_hand=np.zeros(7);plant.arm_tau_command=np.zeros(14)
    plant.event=Mock();plant.sync=Mock();plant.command_arm=Mock()
    plant.sender=SimpleNamespace(ctrl_dual_hand=Mock())
    def snapshot():
        return {name:dict(time=at,q=[0.]*n,dq=[0.]*n,tau_est=[0.]*n,
                          imu_rpy=[0.,0.,0.],faults=[0]*n,temperature_max=35.)
                for name,n in [('body',29),('left',7),('right',7)]}
    plant.read=Mock(side_effect=snapshot)
    error=None
    with patch.dict(os.environ,{'HANDOFF_CAMERA_MAX_AGE_MS':str(max_age_ms)}),patch('runtime.time.monotonic',side_effect=now):
        try:plant._execute_poses(np.zeros(7),[np.zeros(7)])
        except RuntimeError as exc:error=str(exc)
    writes=plant.command_arm.call_count+plant.sender.ctrl_dual_hand.call_count
    if error and writes:raise AssertionError('A rejected command tick wrote an actuator sink')
    return dict(accepted=error is None,error=error,recording_sink_calls=writes,
                exposure_age_ms=(at-timestamp)*1000)


def replay(max_age_ms=500,publication='receipt',repeats=1):
    reports=[]
    for case in cases():
        original=case['original'];stop=case['stop'];next_packet=case['next']
        legacy=command_tick(original['exposure'],stop,400)
        if legacy['accepted']:raise AssertionError('Historical400ms stop failed to reproduce')
        available_key='receipt_at' if publication=='receipt' else 'legacy_at'
        phases=({'motion-tracking','approach','align','reach','pre-close'}
                if publication=='receipt' else {'motion-tracking'})
        ticks=[stop]
        end=next_packet['legacy_at']
        ticks.extend(np.arange(stop+.02,end,.02).tolist());ticks.append(end)
        # Age increases monotonically between publications. Checking the
        # instant immediately before each publication proves the whole
        # interval's age envelope, in addition to actual20ms motor cadence.
        ticks.extend(float(np.nextafter(packet[available_key],-np.inf))
                     for packet in case['packets'] if stop<packet[available_key]<=end and packet['phase'] in phases)
        ticks=sorted(set(ticks))
        outcomes=[]
        for _ in range(repeats):
            run=[]
            for at in ticks:
                available=[packet for packet in case['packets'] if packet[available_key]<=at and packet['phase'] in phases]
                packet=max(available,key=lambda packet:packet['exposure']) if available else original
                outcome=command_tick(packet['exposure'],at,max_age_ms)
                outcome.update(frame_id=packet['frame_id'],at=at,
                    available_at=packet[available_key],geometry_complete_at=packet['geometry_complete_at'],
                    validated_result_at=packet['validated_result_at']);run.append(outcome)
            outcomes.append(run)
        if any(run!=outcomes[0] for run in outcomes[1:]):raise AssertionError('Same-input replay changed verdict')
        reports.append(dict(error=case['fault']['error'],source=case['fault']['sources'],
            original_frame_id=original['frame_id'],original_age_ms=legacy['exposure_age_ms'],
            original400ms_rejected=True,next_frame_id=next_packet['frame_id'],
            next_packet_delay_ms=(end-stop)*1000,raw_frame_inputs=case['raw_inputs'],
            input_sha256=case['input_sha256'],
            ticks=outcomes[0],worst_exposure_age_ms=max(tick['exposure_age_ms'] for tick in outcomes[0]),
            accepted=all(tick['accepted'] for tick in outcomes[0])))
    return dict(scope='Saved successful observation timelines at real command-loop gate; no native latency or physical pickup claim',
                max_age_ms=max_age_ms,publication=publication,repeats=repeats,
                scenarios=len(reports),accepted=sum(case['accepted'] for case in reports),cases=reports)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-age-ms',type=int,default=500)
    parser.add_argument('--publication',choices=['receipt','legacy'],default='receipt')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();report=replay(args.max_age_ms,args.publication,args.repeats)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('max_age_ms','publication','repeats','scenarios','accepted')}))
    return 0 if report['accepted']==report['scenarios'] else 1


if __name__=='__main__':raise SystemExit(main())
