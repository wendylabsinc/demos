"""Join concurrent spans by immutable frame/group, not by worker request.

Usage: python3 analyze_perception_profile.py profile.json > derived-frames.json
The exposure-to-return partition is additive. Stage durations are inclusive.
"""
import json
from pathlib import Path
import sys


def analyze(report):
    spans = [span for sample in report['samples'] for event in sample['events']
             if event['phase'] == 'timing_spans' for span in event['spans']]
    rows = []
    for sample in report['samples']:
        if sample['error'] or sample['vision'] is None:
            rows.append({'i': sample['index'], 'error': sample['error']})
            continue
        frame_id = sample['vision']['frame_id']
        frame = next(event for event in reversed(sample['events'])
                     if event['phase'] == 'timing_frame' and event['frame_id'] == frame_id)
        camera = next(s for s in spans if s['name'] == 'camera_frame' and s['group'] == frame['group'])
        geometry = next(s for s in spans if s['name'] == 'geometry_including_detection' and s['frame_id'] == frame_id)
        archive = next(s for s in spans if s['name'] == 'archive_total' and s['frame_id'] == frame_id)
        inside_geometry = [s for s in spans if s['start_s'] >= geometry['start_s'] and s['end_s'] <= geometry['end_s']]
        inside_camera = [s for s in spans if s['start_s'] >= camera['start_s'] and s['end_s'] <= camera['end_s']
                         and s['thread'] not in ('MainThread', 'perception-archive_0')]
        def duration(collection, name=None, path=None):
            return next((s['ms'] for s in collection if (name is None or s['name'] == name)
                         and (path is None or s.get('path', '').startswith(path))), None)
        result = next(e for e in sample['events'] if e['phase'] == 'perception_result')
        # Partition the same frame's exposure age, including waiting before a fetch
        # and after prefetch. Never sum medians or concurrent decode/archive spans.
        partition = {
            'exposure_to_sdk_receive_ms': (camera['raw_received_s'] - frame['exposure_s']) * 1000,
            'sdk_receive_to_aligned_ms': (camera['received_s'] - camera['raw_received_s']) * 1000,
            'aligned_to_fetch_start_ms': (camera['start_s'] - camera['received_s']) * 1000,
            'fetch_decode_ms': camera['ms'],
            'prefetch_wait_ms': (frame['used_s'] - camera['end_s']) * 1000,
            'pose_and_dispatch_ms': (geometry['start_s'] - frame['used_s']) * 1000,
            'geometry_ms': geometry['ms'],
            'remaining_archive_wait_ms': max(0, archive['end_s'] - geometry['end_s']) * 1000,
            'receipt_tail_ms': (sample['end_s'] - max(archive['end_s'], geometry['end_s'])) * 1000,
        }
        age = sample['exposure_age_ms']
        if abs(sum(partition.values()) - age) > .001:
            raise ValueError(f'Exposure partition does not reconcile for frame {frame_id}')
        rows.append({
            'i': sample['index'], 'frame': frame_id, 'mode': sample['vision']['tracking']['mode'],
            'exposure_age_ms': age, 'headroom_ms': report['deadline_ms'] - age,
            'round_trip_ms': sample['round_trip_ms'], 'exposure_partition_ms': partition,
            'inclusive_stage_ms': {
                'policy_http': duration(inside_camera, path='/frame.policy-rgbd'),
                'recorder_http': duration(inside_camera, path='/frame.recorder-rgbd'),
                'depth_decode': duration(inside_camera, 'depth_decode'),
                'color_decode': duration(inside_camera, 'color_decode'),
                'yolo_client': duration(inside_geometry, 'yolo_client'),
                'yolo_service_predict': result['inference_ms'],
                'mask_or_visual_tracker': duration(inside_geometry, 'mask_or_visual_tracker'),
                'cylinder_fit': duration(inside_geometry, 'cylinder_fit'),
                'archive_total': archive['ms'],
            },
        })
    return rows


if __name__ == '__main__':
    print(json.dumps(analyze(json.loads(Path(sys.argv[1]).read_text())), indent=2, allow_nan=False))
