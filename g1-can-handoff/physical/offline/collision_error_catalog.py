"""Index collision failures in saved evidence without contacting a robot."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def catalog(root):
    entries=[];sources={};seen=set()
    for path in sorted(root.rglob('*.gz')):
        if not path.name.endswith(('.jsonl.gz','.json.gz')):continue
        raw=path.read_bytes();text=gzip.decompress(raw).decode()
        if path.name.endswith('.jsonl.gz'):
            events=[json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload=json.loads(text)
            events=payload.get('events',[]) if isinstance(payload,dict) else payload
        if not isinstance(events,list):raise ValueError('Expected saved event list: '+str(path))
        relative=str(path.relative_to(root))
        sources[relative]=hashlib.sha256(raw).hexdigest()
        for index,event in enumerate(events):
            if not isinstance(event,dict):continue
            collision=event.get('collision')
            message=event.get('error') or event.get('reason') or ''
            if not isinstance(message,str):message=json.dumps(message,sort_keys=True)
            combined=json.dumps(collision,sort_keys=True)+' '+message
            if not collision and not any(term in message.lower() for term in
                    ('collision','trajectory intersects','grasp orientation search time limit')):continue
            if 'torso_link' in combined and 'left_elbow_link' in combined:
                category='elbow_torso_intersection'
            elif collision and 'table' in collision.get('geoms',[]):
                category='table_clearance_or_intersection'
            elif 'table' in combined and ('left_hand_' in combined or 'left_wrist_' in combined):
                category='table_clearance_or_intersection'
            elif 'Staging endpoint collides' in message or 'Target pose collision' in message:
                category='endpoint_collision'
            elif 'finger' in message.lower() or event.get('phase')=='finger_trajectory_collision':
                category='finger_trajectory_collision'
            elif 'orientation search time limit' in message:
                category='grasp_search_deadline'
            elif collision:
                category='other_robot_intersection'
            else:
                category='trajectory_or_preflight_rejection'
            # Captured traces can overlap. Deduplicate the same native event,
            # retaining all source references rather than inflating the count.
            key=(event.get('unix_ns',relative+':'+str(index)),event.get('phase'),combined)
            if key in seen:
                next(e for e in entries if e['_key']==key)['captures'].append(relative)
                continue
            seen.add(key)
            entries.append(dict(_key=key,category=category,captures=[relative],
                event_index=index,unix_ns=event.get('unix_ns'),phase=event.get('phase'),
                phase_name=event.get('phase_name'),collision=collision,message=message,
                joint_q=event.get('joint_q'),path_fraction=event.get('path_fraction')))
    for entry in entries:entry.pop('_key')
    return dict(hardware_contacted=False,sources=sources,unique_events=len(entries),
                categories=dict(sorted(Counter(e['category'] for e in entries).items())),entries=entries)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).parent/'evidence')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=catalog(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'unique_events':result['unique_events'],'categories':result['categories']}))
