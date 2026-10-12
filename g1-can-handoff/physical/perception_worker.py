"""Persistent RGB-D worker isolated from the control process's Python GIL.

The worker has no DDS or motor adapters. It shares the parent's existing camera
lease, preserves original timestamps, and finishes exact-frame persistence
before returning an observation. The parent still checks freshness on receipt.
"""

import multiprocessing
import threading
import time


def configure_visual_worker():
    """Restrict only this child to two CPUs outside the configured control CPUs."""
    import os
    from scene_reference import visual_threads
    from adapter.hotpath import cpu_from_env

    threads = visual_threads()
    result = dict(threads=threads, cpu_pool=None, reserved_control_cpus=None,
                  active=False, error=None)
    if threads == 1 or not hasattr(os, 'sched_setaffinity'):
        return result
    try:
        allowed = sorted(os.sched_getaffinity(0))
        reserved = {cpu_from_env(os.environ, 'G1_MOTOR_CPU', available=allowed,
                                 default='auto'),
                    cpu_from_env(os.environ, 'G1_LOWSTATE_CPU', available=allowed,
                                 default='auto', auto_offset=1)} - {None}
        pool = [cpu for cpu in allowed if cpu not in reserved][-2:]
        if len(pool) != 2:
            raise ValueError('Two non-control CPUs are unavailable')
        os.sched_setaffinity(0, set(pool))
        result.update(cpu_pool=pool, reserved_control_cpus=sorted(reserved), active=True)
    except (OSError, ValueError) as exc:
        # This performance option must not make startup depend on Linux CPU
        # layout or place a parallel vision pool on a reserved control CPU.
        os.environ['HANDOFF_VISUAL_THREADS'] = '1'
        result.update(threads=1, error=str(exc))
    return result


def worker_main(connection, token):
    """Run exact-frame perception in a child process that never owns robot joints."""
    scheduling = configure_visual_worker()
    from concurrent.futures import ThreadPoolExecutor
    from runtime import Runtime
    from camera_input import Camera
    from target_tracker import TargetTracker
    from perception_policy import semantic_recheck_interval_s
    from latest_frame import LatestFrameBuffer

    runtime = Runtime.__new__(Runtime)
    runtime.camera = Camera()
    runtime.camera.token = token
    runtime.tracker = TargetTracker(recheck_interval_s=semantic_recheck_interval_s())
    runtime.archive_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="perception-archive"
    )
    runtime.frame_prefetch = None
    runtime.pending_frame = None
    import os
    if os.getenv('HANDOFF_SCENE_REFERENCE', '0') == '1':
        from scene_reference import TableSceneReference
        runtime.scene_reference = TableSceneReference()
    runtime.latest_frames = LatestFrameBuffer(runtime.camera)
    events = []
    event_lock = threading.Lock()

    def event(phase, **fields):
        with event_lock:
            events.append((phase, fields))

    runtime.event = event

    def exposure_config(config, timestamp):
        if config.get("calibration_method") != "operator_accepted_factory_table":
            return config
        from pose_alignment import corrected_config, validate_pose_skew

        connection.send({"pose_request": timestamp})
        if not connection.poll(0.5):
            raise RuntimeError("Exposure pose response timeout")
        response = connection.recv()
        if response.get("error"):
            raise RuntimeError(response["error"])
        pose = response["pose"]
        validate_pose_skew(pose['timestamp'], timestamp)
        event("exposure_pose", frame_timestamp=timestamp, pose=pose)
        return corrected_config(config, pose)

    runtime.exposure_config = exposure_config
    try:
        from yolo_client import frame_age_limit_s

        runtime.latest_frames.wait_ready(max_age_s=0.75 * frame_age_limit_s())
        connection.send({"ready": True})
        while True:
            request = connection.recv()
            if request is None:
                break
            with event_lock:
                events.clear()
            started = time.monotonic()
            try:
                result = runtime._observe_frame(*request)
                response = {"result": result, "error": None}
            except Exception as exc:
                response = {"result": None, "error": f"{type(exc).__name__}: {exc}"}
            with event_lock:
                response["events"] = list(events)
            response["worker_ms"] = (time.monotonic() - started) * 1000
            response["worker_scheduling"] = scheduling
            connection.send(response)
    except (EOFError, BrokenPipeError):
        pass
    finally:
        runtime.latest_frames.close()
        runtime.archive_executor.shutdown(wait=True, cancel_futures=True)
        # The parent owns and releases the lease, including on child failure.
        runtime.camera.token = None
        runtime.camera.close()
        connection.close()


class PerceptionWorker:
    """Request isolated perception while servicing planning-time pose lookups."""

    def __init__(self, camera_token, *, target=worker_main, startup_timeout=10.0):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=target,
            args=(child, camera_token),
            daemon=True,
            name="rgbd-perception",
        )
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(startup_timeout):
                raise RuntimeError("Perception worker startup timeout")
            if self.connection.recv() != {"ready": True}:
                raise RuntimeError("Invalid perception worker startup")
        except Exception:
            self.close()
            raise

    def observe(
        self,
        config,
        phase,
        require_table,
        expected_center_xy,
        event,
        pose_at=None,
        publish=None,
    ):
        """Return one persisted, fresh observation; replay its events to the parent."""
        from yolo_client import frame_age_limit_s

        if not self.process.is_alive():
            raise RuntimeError("Perception worker exited")
        started = time.monotonic()
        self.connection.send((config, phase, require_table, expected_center_xy))
        # Leave two seconds for error reporting before stop_tracking's join
        # deadline. A stalled camera must not prevent worker/lease cleanup.
        deadline = started + 4.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.connection.poll(remaining):
                raise RuntimeError("Perception worker response timeout")
            response = self.connection.recv()
            if "pose_request" not in response:
                break
            try:
                if pose_at is None:
                    raise RuntimeError("Exposure pose history unavailable")
                reply = {"pose": pose_at(response["pose_request"])}
            except Exception as exc:
                reply = {"error": str(exc)}
            self.connection.send(reply)
        received = time.monotonic()
        receipt_error = None
        try:
            if response["error"]:
                raise RuntimeError(response["error"])
            observation, vision = response["result"]
            age = time.monotonic() - observation.timestamp
            if not 0 <= age <= frame_age_limit_s():
                raise RuntimeError(
                    f"Perception worker observation stale on receipt ({age*1000:.1f} ms)"
                )
            # Child exact-frame persistence is already joined. Publish only
            # after the parent's freshness and target/path acceptance callback,
            # before replaying descriptive events through synchronous file IO.
            if publish is not None:
                publish(observation, vision)
        except Exception as exc:
            receipt_error = exc
        relay_started = time.monotonic()
        for name, fields in response["events"]:
            event(name, **fields)
        event(
            "perception_worker_result",
            phase_name=phase,
            worker_ms=response["worker_ms"],
            round_trip_ms=(time.monotonic() - started) * 1000,
            receive_ms=(received - started) * 1000,
            event_relay_ms=(time.monotonic() - relay_started) * 1000,
            error=response["error"],
            receipt_error=None if receipt_error is None else str(receipt_error),
            worker_scheduling=response.get('worker_scheduling'),
        )
        if receipt_error is not None:
            raise receipt_error
        age = time.monotonic() - observation.timestamp
        if not 0 <= age <= frame_age_limit_s():
            raise RuntimeError(
                f"Perception worker observation stale on receipt ({age*1000:.1f} ms)"
            )
        return observation, vision

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(6.0)
        if self.process.is_alive():
            # This process has no actuators and never owns the camera lease.
            self.process.terminate()
            self.process.join(2.0)
        self.connection.close()
