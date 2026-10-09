"""Read-only complete sealed-archive audit; run in the pinned SDK runtime."""

import base64
import collections
import gzip
import json
import math
from unitree_sdk2py.idl.unitree_hg.msg import dds_
from recorder import TOPICS


def audit(path):
    counts = collections.Counter()
    faults = collections.Counter()
    ranges = {}
    ticks = []
    pressure = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            topic = row.get("topic")
            if not topic:
                continue
            msg = getattr(dds_, TOPICS[topic]).deserialize(
                base64.b64decode(row["cdr_base64"])
            )
            counts[topic] += 1
            ms = getattr(msg, "motor_state", [])
            if any(m.motorstate for m in ms) or any(getattr(msg, "error", [])):
                faults[topic] += 1
            if topic == "rt/lowstate":
                ticks.append(msg.tick)
            if ms:
                if topic not in ranges:
                    ranges[topic] = {
                        "q_min": [float(m.q) for m in ms],
                        "q_max": [float(m.q) for m in ms],
                        "temperature_max": 0,
                        "nonfinite_motor_fields": 0,
                        "tau_est_abs_max_raw": 0,
                    }
                r = ranges[topic]
                for i, m in enumerate(ms):
                    if math.isfinite(m.q):
                        r["q_min"][i] = min(r["q_min"][i], m.q)
                        r["q_max"][i] = max(r["q_max"][i], m.q)
                    for value in (m.q, m.dq, m.ddq, m.tau_est):
                        r["nonfinite_motor_fields"] += not math.isfinite(value)
                    r["temperature_max"] = max(r["temperature_max"], *m.temperature)
                    r["tau_est_abs_max_raw"] = max(
                        r["tau_est_abs_max_raw"], abs(m.tau_est)
                    )
            sensors = getattr(msg, "press_sensor_state", [])
            if sensors:
                values = [s.lost for s in sensors]
                p = pressure.setdefault(
                    topic, {"first_lost_counters": values, "last_lost_counters": values}
                )
                p["last_lost_counters"] = values
    deltas = collections.Counter((b - a) & 0xFFFFFFFF for a, b in zip(ticks, ticks[1:]))
    return {
        "rows_by_topic": dict(counts),
        "rows_with_nonzero_motor_or_hand_errors": dict(faults),
        "motor_ranges": ranges,
        "body_tick_delta_histogram": dict(deltas),
        "pressure_sensor_lost_counters": pressure,
        "pressure_loss_counter_meaning": "Raw values retained; semantics not yet verified",
        "tau_units_verified": False,
        "physical_grasp_verified": False,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(audit(sys.argv[1]), allow_nan=False))
