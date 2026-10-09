"""Read a sealed DDS recording; print sampled shoulder load, without motion."""

import base64
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


SOURCE_SHA256 = "488ece61fc593c919ac7901bdabeb58e8713ca9d5faf724b354fe0ef0f7c65f4"
START_NS = 1791573871294168201
path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != SOURCE_SHA256:
    raise ValueError("Source checksum differs from the recorded segment")

bins = defaultdict(lambda: defaultdict(list))
last_sample = {}
with gzip.open(path, "rt") as source:
    for line in source:
        row = json.loads(line)
        topic = row.get("topic")
        stamp = row.get("unix_ns", 0)
        if topic not in ("rt/arm_sdk", "rt/lowstate"):
            continue
        if not START_NS + 94_000_000_000 <= stamp <= START_NS + 105_000_000_000:
            continue
        if stamp - last_sample.get(topic, 0) < 100_000_000:
            continue
        last_sample[topic] = stamp
        bucket = bins[(stamp - START_NS) // 1_000_000_000]
        decoder = LowState_ if topic == "rt/lowstate" else LowCmd_
        message = decoder.deserialize(base64.b64decode(row["cdr_base64"], validate=True))
        if topic == "rt/lowstate":
            motor = message.motor_state[16]
            bucket["state"].append((motor.q, motor.dq, motor.tau_est, list(motor.temperature)))
        else:
            motor = message.motor_cmd[16]
            bucket["command"].append((motor.q, motor.kp, motor.kd, motor.tau))

rows = []
for second, bucket in sorted(bins.items()):
    states, commands = bucket["state"], bucket["command"]
    out = {"seconds_after_start": second, "state_samples": len(states), "command_samples": len(commands)}
    if states:
        out.update(
            measured_q_mean_rad=sum(x[0] for x in states) / len(states),
            measured_speed_abs_max_rad_s=max(abs(x[1]) for x in states),
            reported_torque_mean_nm=sum(x[2] for x in states) / len(states),
            reported_torque_abs_max_nm=max(abs(x[2]) for x in states),
            last_casing_winding_c=states[-1][3],
        )
    if commands:
        out.update(
            commanded_q_mean_rad=sum(x[0] for x in commands) / len(commands),
            kp=commands[-1][1],
            kd=commands[-1][2],
            feedforward_mean_nm=sum(x[3] for x in commands) / len(commands),
        )
    rows.append(out)

print(json.dumps({
    "schema": "g1-recorded-shoulder-load/1",
    "source_file": path.name,
    "source_sha256": digest.hexdigest(),
    "recorder_session": "20261009T191749Z-f6bb3fab",
    "demo_start_unix_ns": START_NS,
    "joint": "left_shoulder_roll",
    "body_motor_index": 16,
    "method": "First callback sample per topic at least 100 ms after the previous selected sample; one-second summaries. Command and feedback streams are sampled independently. Torque is firmware-reported tau_est, not a calibrated external measurement. Temperatures are last selected samples, not per-second maxima. No publishers, live subscriptions or robot commands.",
    "bins": rows,
}, indent=2))
