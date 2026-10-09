"""Exercise real settling code with preserved DDS feedback, without a robot.

Availability and sample ages use recorder receipt monotonic times. The native
controller's exact read selections and callback times were not recorded.
"""

import argparse
import base64
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
from unittest.mock import patch

from control_settle import wait_for_stable_control

PHYSICAL = Path(__file__).resolve().parents[1]
EVIDENCE = PHYSICAL / "offline/validation/issue-control-settling-20261008"
NATIVE_SHA256 = "45b018d0d16590dc011412aa2d77f4108bbbfc01a06178c75a5db82e36314093"
TOPICS = {
    "rt/lowstate": "body",
    "rt/dex3/left/state": "left",
    "rt/dex3/right/state": "right",
}


def cases():
    raw = (EVIDENCE / "native-feedback.json.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != NATIVE_SHA256:
        raise ValueError("Native feedback artifact hash mismatch")
    data = json.loads(gzip.decompress(raw))
    for trial in data["trials"]:
        for row in trial["rows"]:
            if not row.get("cdr_base64"):
                raise ValueError("Native CDR bytes missing")
            # HandState_ final IDL fields are error:uint32[2], reserve:uint32[2].
            # Preserve the runtime's additional hand error guard from raw CDR.
            if TOPICS[row["topic"]] != "body":
                cdr = base64.b64decode(row["cdr_base64"], validate=True)
                if cdr[:4] != b"\x00\x01\x00\x00" or len(cdr) != 1428:
                    raise ValueError("Unexpected captured HandState CDR layout")
                row["sample"]["faults"].extend(struct.unpack("<4I", cdr[-16:])[:2])
    return data["trials"]


class ReceiptClock:
    def __init__(self):
        self.ticks_ns = 0

    def monotonic(self):
        return self.ticks_ns / 1e9

    def sleep(self, seconds):
        self.ticks_ns += round(seconds * 1e9)


class NativeFeedback:
    def __init__(self, trial, clock):
        self.clock = clock
        first = trial["rows"][0]
        self.start_ns = (
            first["monotonic_ns"] + trial["start_unix_ns"] - first["unix_ns"]
        )
        self.rows = sorted(trial["rows"], key=lambda row: row["monotonic_ns"])
        self.index = 0
        self.latest = {}
        self.reads = 0
        self.max_age_s = {name: 0.0 for name in TOPICS.values()}

    def read(self):
        now_ns = self.start_ns + self.clock.ticks_ns
        while (
            self.index < len(self.rows)
            and self.rows[self.index]["monotonic_ns"] <= now_ns
        ):
            row = self.rows[self.index]
            self.latest[TOPICS[row["topic"]]] = row
            self.index += 1
        if set(self.latest) != set(TOPICS.values()):
            raise ValueError("Missing original body/hand feedback at replay start")
        snapshot = {}
        for name, row in self.latest.items():
            sample = copy.deepcopy(row["sample"])
            sample["time"] = (row["monotonic_ns"] - self.start_ns) / 1e9
            self.max_age_s[name] = max(
                self.max_age_s[name], self.clock.monotonic() - sample["time"]
            )
            snapshot[name] = sample
        self.reads += 1
        return snapshot


def run_case(trial, limit=1.0, mutation=None):
    clock = ReceiptClock()
    feedback = NativeFeedback(trial, clock)
    events = []
    archives = []

    def read():
        snapshot = feedback.read()
        if mutation:
            mutation(snapshot, clock.monotonic())
        return snapshot

    with patch.dict(os.environ, {"HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S": str(limit)}):
        try:
            wait_for_stable_control(
                read,
                threading.Event(),
                lambda phase, **fields: events.append(dict(phase=phase, **fields)),
                lambda: archives.append(clock.monotonic()),
                clock,
            )
            error = None
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
    return dict(
        trial=trial["trial"],
        limit_rad_s=limit,
        settled=error is None,
        error=error,
        elapsed_s=clock.monotonic(),
        reads=feedback.reads,
        max_recorder_feedback_age_s=feedback.max_age_s,
        archive_checks=len(archives),
        events=events,
    )


def negative_cases():
    for joint in range(29):
        for speed in (-1.01, 1.0, 1.01):

            def fast(snapshot, now, joint=joint, speed=speed):
                snapshot["body"]["dq"][joint] = speed

            yield f"body-speed-joint-{joint}-{speed}", fast, "within 5 seconds"
    for name in ("body", "left", "right"):

        def stale(snapshot, now, name=name):
            snapshot[name]["time"] = now - 0.201

        yield f"stale-{name}", stale, f"Stale {name}"

        def fault(snapshot, now, name=name):
            snapshot[name]["faults"][0] = 1

        yield f"motor-fault-{name}", fault, f"{name} motor fault"
    for name in ("left", "right"):

        def hand_error(snapshot, now, name=name):
            snapshot[name]["faults"][7] = 1

        yield f"hand-device-error-{name}", hand_error, f"{name} motor fault"
    for joint in range(15):

        def unstable_base(snapshot, now, joint=joint):
            snapshot["body"]["q"][joint] += 0.1 * now

        yield f"unstable-base-joint-{joint}", unstable_base, "within 5 seconds"
    for axis in range(3):

        def unstable_imu(snapshot, now, axis=axis):
            snapshot["body"]["imu_rpy"][axis] += 0.1 * now

        yield f"unstable-imu-axis-{axis}", unstable_imu, "within 5 seconds"


def replay(repeats=3):
    if repeats < 3:
        raise ValueError("Acceptance requires at least three repeats")
    trials = cases()
    results = []
    negatives = []
    for repeat in range(repeats):
        for trial in trials:
            old = run_case(trial, 0.03)
            new = run_case(trial, 1.0)
            results.append(
                dict(
                    repeat=repeat + 1,
                    trial=trial["trial"],
                    red=old,
                    green=new,
                    accepted=not old["settled"]
                    and "within 5 seconds" in (old["error"] or "")
                    and new["settled"],
                )
            )
            for name, mutation, expected in negative_cases():
                result = run_case(trial, 1.0, mutation)
                negatives.append(
                    dict(
                        repeat=repeat + 1,
                        trial=trial["trial"],
                        scenario=name,
                        **{
                            key: result[key]
                            for key in ("settled", "error", "elapsed_s")
                        },
                        accepted=not result["settled"]
                        and expected in (result["error"] or ""),
                    )
                )
    return dict(
        method="Actual wait_for_stable_control; original native DDS body/left/right samples, recorder receipt time and preserved recorder feedback ages, 20ms replay reads",
        native_feedback_sha256=NATIVE_SHA256,
        repeats=repeats,
        limitations=[
            "Original controller callback times and exact loop sample selection unavailable; recorder-selected 20ms per-topic subset is used.",
            "Offline guard behavior only; no physical settling, hardware motion or ownership transition tested.",
        ],
        sources=[
            {
                key: trial[key]
                for key in (
                    "trial",
                    "source_segment",
                    "source_segment_sha256",
                    "start_unix_ns",
                    "end_unix_ns",
                )
            }
            for trial in trials
        ],
        red_green=results,
        negatives=negatives,
        accepted=all(row["accepted"] for row in results + negatives),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report", type=Path, default=EVIDENCE / "acceptance.json")
    args = parser.parse_args()
    report = replay(args.repeats)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                accepted=report["accepted"],
                red_green=len(report["red_green"]),
                negatives=len(report["negatives"]),
                report=str(args.report),
            )
        )
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
