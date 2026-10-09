"""Offline comparison of contact signals on two sealed real failure traces."""
import gzip
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads(gzip.decompress((ROOT / 'trace.json.gz').read_bytes()))
summaries = []
for case in data['cases']:
    rows = case['rows']
    baseline = statistics.median(r['temperatures'][2][1] for r in rows if -8 <= r['t'] < -1)
    starts = {}
    first = {}
    decisions = []
    for index, row in enumerate(rows):
        t = row['t']
        # The commanded phase is CLOSE. Do not arm this signal before closing.
        if t < 0:
            continue
        tactile = row['tactile_channels'] >= 2
        history = [r for r in rows[:index + 1] if t - 0.25 <= r['t'] <= t]
        blocked = []
        cmd = row['cmd']
        if cmd and len(history) >= 3 and t - history[0]['t'] >= 0.18:
            for joint in range(7):
                span = max(r['q'][joint] for r in history) - min(r['q'][joint] for r in history)
                gap = cmd['q'][joint] - row['q'][joint]
                if span <= 0.02 and abs(gap) >= 0.12 and abs(gap * cmd['kp'][joint]) > 0.5:
                    blocked.append(joint)
        conditions = {
            'tactile_only': tactile,
            'encoder_plateau_only': bool(blocked),
            'tactile_and_encoder_plateau': tactile and bool(blocked),
            'temperature_rise_5C': row['temperatures'][2][1] >= baseline + 5,
        }
        for name, active in conditions.items():
            if not active:
                starts.pop(name, None)
                continue
            starts.setdefault(name, t)
            if name not in first and t - starts[name] >= 0.15:
                first[name] = {
                    'trigger_s': t,
                    'lead_before_stop_s': case['stop_s'] - t,
                    'thumb2_winding_C': row['temperatures'][2][1],
                    'thumb2_q': row['q'][2],
                    'blocked_joint_slots': blocked,
                    'tactile_channels': row['tactile_channels'],
                    'observed_command_age_s': t - cmd['time'] if cmd else None,
                }
        decisions.append({'t': t, 'conditions': conditions, 'blocked': blocked})
    # This is a candidate-signal replay, not a simulated thermal outcome.
    chosen = first.get('tactile_and_encoder_plateau')
    assert chosen is not None, f"No combined contact signal in {case['name']}"
    assert chosen['trigger_s'] < case['stop_s'] - 2, f"Signal too late in {case['name']}"
    assert chosen['thumb2_winding_C'] < 50, f"Heating already high in {case['name']}"
    preclose_max = max(r['tactile_channels'] for r in rows if r['t'] < 0)
    assert preclose_max == 0, f"Tactile false trigger before close: {case['name']}"
    summaries.append({
        'name': case['name'], 'stop_s': case['stop_s'],
        'baseline_thumb2_winding_C': baseline,
        'candidate_signals': first,
        'preclose_max_tactile_channels': preclose_max,
        'lost_counters_unchanged': rows[0]['lost'] == rows[-1]['lost'],
        'source_file': case['file'], 'source_sha256': case['sha'],
    })
result = {
    'cases': summaries,
    'candidate_rule': 'During close: >=2 tactile channels rise >=max(2000 raw,6 baseline MAD), AND a loaded joint spans <=0.02 rad over >=0.18 s with >=0.12 rad target gap and nominal proportional effort >0.5 Nm; require combined condition for >=0.15 s.',
    'scope': 'Candidate calibrated and evaluated on two known failures. No independent held-out success/failure validation, no grip controller change, no deployment, no motion.',
    'limits': ['Raw pressure units and anatomical mapping unverified.', 'SDK temperature and motor torque/current semantics are not calibrated here.', 'Pressure samples may repeat; callback receipt age is not sensor exposure age.', 'This predicts trigger timing only; cannot prove reduced heating or grip retention without a physical test.'],
}
(ROOT / 'signal-comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
