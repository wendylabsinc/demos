"""Offline table-clearance sweep using captured posture and synthetic can positions."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import scipy
from offline.planner_replay import fixture, clearance_auditor
from pickup import prepare
from physical_core import calibrated_scene, PhysicalPlant

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'offline/evidence/clearance-demos-20261008-preserved'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--exact', action='store_true')
parser.add_argument('--positions', nargs='+', help='Can XY pairs, in meters, formatted x,y')
parser.add_argument('--height', type=float, help='Synthetic table height in meters')
args = parser.parse_args()
cases = [(None, None)] if args.exact else [(x, y) for x in (.28, .35, .44) for y in (.10, .18, .28)]
if args.positions:
    cases = [tuple(float(v) for v in xy.split(',')) for xy in args.positions]
result = dict(parent_commit='b75956609a5a9e0003ad9b6485c08036d9b06055',
              physical_core_sha256=hashlib.sha256((ROOT/'physical_core.py').read_bytes()).hexdigest(),
              python=sys.version, mujoco=mujoco.__version__, numpy=np.__version__, scipy=scipy.__version__,
              source_trial=3, hardware_used=False, actuation=False, physical_success_verified=False,
              scenario='Captured base/arms; synthetic can XY and optional table height; recorded terminal state replay', cases=[])
for index, (x, y) in enumerate(cases):
    plant, observation, _, logged = fixture(3, EVIDENCE)
    if x is not None:
        observation.center[:2] = [x, y]
    if args.height is not None:
        config = copy.deepcopy(plant.config)
        config['table_height_m'] = args.height
        observation.table_height = args.height
        observation.center[2] = args.height + config['can_height_m']/2
        scene = calibrated_scene(config, x=observation.center[0], y=observation.center[1])
        plant = PhysicalPlant(scene, None, None, plant.read, config, plant.event, clock=plant.clock)
    plant.observe_can(observation)
    before = plant.ik.data.qpos.copy()
    clearance = clearance_auditor(plant)
    endpoint_errors = []
    solve = plant.ik.solve
    def tracked_solve(*params, **kwargs):
        try:
            return solve(*params, **kwargs)
        except ValueError as exc:
            endpoint_errors.append(dict(phase=plant.ik.contact_phase, error=str(exc)))
            raise
    plant.ik.solve = tracked_solve
    def observe(*params):
        observation.timestamp = time.monotonic()
        return observation, dict(source='captured_scene_with_synthetic_can_XY')
    started = time.monotonic()
    record = dict(center_m=observation.center.tolist(), table_height_m=observation.table_height)
    try:
        _, _, points, rotation = prepare(plant, observe, plant.event)
        record.update(passed=True, waypoints=[p.tolist() for p in points], rotation=rotation.tolist(), clearance=clearance)
    except (ValueError, RuntimeError) as exc:
        record.update(passed=False, error=str(exc))
    record.update(elapsed_s=time.monotonic()-started,
                  endpoint_errors=endpoint_errors,
                  endpoint_error_counts=dict(Counter(e['error'].split(':', 1)[0] for e in endpoint_errors)),
                  collision_events=[e for e in logged if e['phase'] in ('trajectory_collision', 'finger_trajectory_collision')],
                  rejected=[e for e in logged if e['phase']=='grasp_candidate_rejected'],
                  path_preflight_ok=plant.path_preflight_ok,
                  arm_state_restored=bool(np.array_equal(before[plant.ik.qadr[plant.ik.arm]], plant.ik.data.qpos[plant.ik.qadr[plant.ik.arm]])),
                  adapter_absent=plant.adapter is None, sender_absent=plant.sender is None)
    result['cases'].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(case=index, xy_m=observation.center[:2].tolist(), passed=record['passed'],
                         elapsed_s=record['elapsed_s'], error=record.get('error'),
                         collision_pairs=dict(Counter('/'.join(e['collision']['bodies']) for e in record['collision_events'])),
                         endpoint_errors=record['endpoint_error_counts'])), flush=True)
raise SystemExit(0 if all(case['passed'] for case in result['cases']) else 1)
