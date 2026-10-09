"""Bounded passive profiling of the deployed observation worker, without DDS.

Run inside the physical image with PYTHONPATH=/app. Camera renewals are suppressed
as in benchmark_perception.py: public immutable reads only, no camera ownership.
Feedback comes from the existing controller's read-only status endpoint. Exact
frame archives go to a separate profile directory; calibration is never changed.
Durations are inclusive and concurrent spans must not be added together.
"""

import argparse
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


def status():
    return json.load(urllib.request.urlopen("http://127.0.0.1:8102/status", timeout=3))


def profiled_worker(connection, token, archive_path):
    os.environ["HANDOFF_DATA"] = archive_path
    import camera_input
    import runtime
    import yolo_client
    from target_tracker import TargetTracker
    from perception_worker import worker_main

    spans = []
    lock = threading.Lock()
    local = threading.local()
    sequence = iter(range(100000))

    def add(name, start, **fields):
        row = dict(
            name=name,
            start_s=start,
            end_s=time.monotonic(),
            thread=threading.current_thread().name,
            **fields,
        )
        row["ms"] = (row["end_s"] - start) * 1000
        with lock:
            spans.append(row)

    def wrap(owner, name, label, frame_arg=None):
        original = getattr(owner, name)

        def measured(*args, **kwargs):
            start = time.monotonic()
            frame = None if frame_arg is None else args[frame_arg]
            try:
                return original(*args, **kwargs)
            finally:
                add(
                    label,
                    start,
                    group=getattr(local, "group", None),
                    frame_id=None if frame is None else frame["meta"]["frame_id"],
                )

        setattr(owner, name, measured)

    original_request = camera_input.Camera.request

    def request(self, path, data=None):
        if path == "/lease/renew":
            return b"{}"
        start = time.monotonic()
        raw = None
        try:
            raw = original_request(self, path, data)
            return raw
        finally:
            add(
                "camera_http",
                start,
                path=path,
                group=getattr(local, "group", None),
                bytes=None if raw is None else len(raw),
            )

    camera_input.Camera.request = request
    original_frame = camera_input.Camera.frame

    def frame(self, *args, **kwargs):
        local.group = next(sequence)
        start = time.monotonic()
        result = None
        try:
            result = original_frame(self, *args, **kwargs)
            result["_profile_group"] = local.group
            return result
        finally:
            add(
                "camera_frame",
                start,
                group=local.group,
                frame_id=None if result is None else result["meta"]["frame_id"],
                exposure_s=None if result is None else result["timestamp"],
                received_s=(
                    None if result is None else result["meta"].get("received_monotonic")
                ),
                raw_received_s=(
                    None
                    if result is None
                    else result["meta"].get("raw_received_monotonic_ns", 0) / 1e9
                ),
            )

    camera_input.Camera.frame = frame
    wrap(camera_input, "decode_depth", "depth_decode")
    wrap(camera_input, "decode_color", "color_decode")
    wrap(yolo_client.YoloClient, "detect", "yolo_client", 1)
    wrap(TargetTracker, "select", "mask_or_visual_tracker", 1)
    wrap(camera_input, "fit_cylinder", "cylinder_fit")
    wrap(runtime, "geometry", "geometry_including_detection", 0)
    wrap(runtime, "materialize_rgbd", "archive_npz_encode", 0)
    wrap(runtime.Runtime, "archive_frame", "archive_total", 1)
    original_observe = runtime.Runtime._observe_frame
    original_next = runtime.Runtime.next_tracking_frame

    def next_frame(self):
        start = time.monotonic()
        result = original_next(self)
        self.event(
            "timing_frame",
            frame_id=result["meta"]["frame_id"],
            group=result["_profile_group"],
            exposure_s=result["timestamp"],
            used_s=time.monotonic(),
            next_frame_ms=(time.monotonic() - start) * 1000,
            source_metadata=result["meta"],
        )
        return result

    runtime.Runtime.next_tracking_frame = next_frame

    def observe(self, *args, **kwargs):
        if not getattr(self, "_profile_pose_wrapped", False):
            wrap(self, "exposure_config", "pose_alignment_ipc")
            self._profile_pose_wrapped = True
        try:
            return original_observe(self, *args, **kwargs)
        finally:
            with lock:
                rows = list(spans)
                spans.clear()
            self.event("timing_spans", spans=rows)

    runtime.Runtime._observe_frame = observe
    worker_main(connection, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.samples <= 32:
        parser.error("samples must be1..32")
    before = status()
    if before["demo_running"] or before["phase"] not in ("PASSIVE", "FROZEN"):
        raise RuntimeError("Do not compete with a running trial")
    from pose_alignment import PoseHistory

    # The real controller already imports Observation through camera_input.
    # Otherwise the first IPC unpickle imports SciPy inside the timed receive,
    # adding a profiler-only delay of about 100-160 ms on NX2.
    from g1_can_handoff.perception import Observation
    from perception_worker import PerceptionWorker
    from functools import partial

    history = PoseHistory()
    stop = threading.Event()
    poll_errors = []

    def poll():
        last = None
        while not stop.is_set():
            try:
                body = status()["measured"]["body"]
                if body["time"] != last:
                    history.add(body["time"], body["imu_rpy"], body["q"][12:15])
                    last = body["time"]
            except Exception as exc:
                poll_errors.append(str(exc))
            stop.wait(0.01)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    archive = "/data/perception-profiles/" + time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    config = json.loads(Path("/data/calibration.json").read_text())
    rows = []
    events = []
    started = time.monotonic()
    worker = None
    try:
        worker = PerceptionWorker(
            None, target=partial(profiled_worker, archive_path=archive)
        )
        startup_ms = (time.monotonic() - started) * 1000
        for index in range(args.samples):
            events.clear()
            began = time.monotonic()
            result = None
            error = None

            def event(phase, **fields):
                events.append(dict(phase=phase, **fields))

            try:
                result = worker.observe(
                    config, "passive-timing", True, None, event, pose_at=history.nearest
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finished = time.monotonic()
            rows.append(
                dict(
                    index=index,
                    start_s=began,
                    end_s=finished,
                    round_trip_ms=(finished - began) * 1000,
                    exposure_age_ms=(
                        None
                        if result is None
                        else (finished - result[0].timestamp) * 1000
                    ),
                    vision=None if result is None else result[1],
                    error=error,
                    events=list(events),
                )
            )
            time.sleep(0.01)
    finally:
        if worker is not None:
            worker.close()
        stop.set()
        thread.join(4)
    after = status()
    report = dict(
        schema="wendy.g1.perception-timing.v1",
        passive=True,
        new_actuator_created=False,
        camera_lease_acquired=False,
        deadline_ms=before["camera_max_age_ms"],
        startup_ms=startup_ms,
        archive_path=archive,
        before={
            k: before[k]
            for k in [
                "phase",
                "demo_running",
                "adapter_initialized",
                "motion_commands_sent",
            ]
        },
        after={
            k: after[k]
            for k in [
                "phase",
                "demo_running",
                "adapter_initialized",
                "motion_commands_sent",
            ]
        },
        pose_poll_errors=poll_errors,
        samples=rows,
        limitations=[
            "Inclusive overlapping spans; not an additive stage budget.",
            "Passive workload with feedback read through controller HTTP; real trial trace is separate.",
            "Frozen SDK may continue its existing hold publications; this probe creates no publisher.",
        ],
    )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
