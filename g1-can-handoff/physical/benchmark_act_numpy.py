"""Read-only ACT timing/parity comparison; imports no controller or robot IO.

Example: python benchmark_act_numpy.py --baseline-source /tmp/act_numpy_before.py
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import warnings

import numpy as np

from act_numpy import NumpyACT, verify_parity


def benchmark(directory, baseline_source=None, rounds=5):
    models = {'optimized': NumpyACT(directory)}
    if baseline_source:
        spec = importlib.util.spec_from_file_location('act_numpy_baseline', baseline_source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        models['baseline'] = module.NumpyACT(directory)
    with np.load(Path(directory) / 'parity-fixtures.npz', allow_pickle=False) as fixture:
        observations = fixture['observations'].copy()
    report = dict(hardware_commands=0, machine=platform.machine(), system=platform.system(),
                  numpy_version=np.__version__, samples_per_round=len(observations), rounds=rounds,
                  parity=verify_parity(directory))
    if baseline_source:
        report['baseline_source_sha256'] = hashlib.sha256(Path(baseline_source).read_bytes()).hexdigest()
        before = np.stack([models['baseline'].predict(obs) for obs in observations])
        after = np.stack([models['optimized'].predict(obs) for obs in observations])
        np.testing.assert_allclose(after, before, atol=2e-5, rtol=2e-5)
        report['baseline_maximum_action_error'] = float(np.abs(after - before).max())
        report['baseline_bitwise_equal'] = bool(np.array_equal(after, before))
    timings = {name: [] for name in models}
    round_reports = []
    for model in models.values():
        model.reset()
        for obs in observations[:100]:
            model.predict(obs)
    for index in range(rounds):
        names = list(models)
        if index % 2:
            names.reverse()
        current = {}
        for name in names:
            model = models[name]
            model.reset()
            samples = []
            for obs in observations:
                start = time.perf_counter_ns()
                model.predict(obs)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            timings[name].extend(samples)
            current[name] = dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))
        round_reports.append(current)
    report['timings'] = {
        name: dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))
        for name, samples in timings.items()}
    report['timing_rounds'] = round_reports
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', default=str(Path(__file__).parent / 'act_policy'))
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20:
        parser.error('--rounds must be between 1 and 20')
    # Record backend warnings as evidence; do not change NumPy error handling.
    # Default warning filtering records each warning location once, so logging
    # does not dominate the steady-state timing measurement.
    with warnings.catch_warnings(record=True) as backend_warnings:
        warnings.simplefilter('default')
        result = benchmark(args.directory, args.baseline_source, args.rounds)
    result['backend_warnings'] = sorted(set(str(item.message) for item in backend_warnings))
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + '\n')
    print(output)
