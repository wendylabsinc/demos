"""Offline matched MuJoCo comparison. No device access or hardware commands."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sim_runtime import run, PhysicalPlant
from physical_core import sim_hand_to_bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_path = ROOT / 'offline/validation/single-plan-deploy-20261009T164140Z/project/physical_core.py'
    source = source_path.read_text()
    results = []
    baseline_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PhysicalPlant')
    originals = {name: getattr(PhysicalPlant, name) for name in ['hand', 'contacts', 'maintain_hand']}
    try:
        result = run(output=args.output_dir / 'gentle.json', position_noise='none', height=.75)
        results.append({'mode': 'gentle', 'lift_verified': result['lift_verified'], 'error': result.get('error'), 'truth_can_position_m': result['truth_can_position_m']})
        for name in originals:
            method = next(n for n in baseline_class.body if isinstance(n, ast.FunctionDef) and n.name == name)
            scope = {'np': np, 'sim_hand_to_bridge': sim_hand_to_bridge}
            exec(ast.get_source_segment(source, method), scope)
            setattr(PhysicalPlant, name, scope[name])
        result = run(output=args.output_dir / 'old-closing-same-P2.json', position_noise='none', height=.75)
        results.append({'mode': 'old-closing-same-P2', 'lift_verified': result['lift_verified'], 'error': result.get('error'), 'truth_can_position_m': result['truth_can_position_m']})
    finally:
        for name, method in originals.items():
            setattr(PhysicalPlant, name, method)
    summary = {'baseline_source': str(source_path), 'baseline_source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'results': results, 'scope': 'Same arm P75, finger P2/D0.2, model, planner, and scene. Only closing/contact behavior differs. No physical validation.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
