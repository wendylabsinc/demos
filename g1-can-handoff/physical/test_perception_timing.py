"""Deterministic exposure timing across shared background/phase worker IPC.

PERCEPTION_TIMING_RUNTIME_SOURCE can point at a historical runtime.py to prove
the shared-publication regression fails without changing the working source.
No robot, camera lease, DDS publisher, or wall-clock latency is involved.
"""

from concurrent.futures import Future
import importlib.util
import multiprocessing
import os
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

_test_data = tempfile.TemporaryDirectory(prefix="perception-timing-")
with patch.dict(os.environ, {"HANDOFF_DATA": _test_data.name}):
    import runtime as current_runtime
from perception_worker import PerceptionWorker

runtime_module = current_runtime
if os.getenv("PERCEPTION_TIMING_RUNTIME_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "timing_historical_runtime", os.environ["PERCEPTION_TIMING_RUNTIME_SOURCE"]
    )
    runtime_module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {"HANDOFF_DATA": _test_data.name}):
        spec.loader.exec_module(runtime_module)


class ExposureClock:
    def __init__(self):
        self.now = 100.0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def set(self, now):
        with self.lock:
            if now < self.now:
                raise AssertionError("Clock must advance")
            self.now = now

    def sleep(self, seconds):
        with self.lock:
            self.now += seconds


class ControlledStop:
    """Admit two background samples; park the second behind a phase caller."""

    def __init__(self):
        self.stopped = False
        self.calls = 0
        self.second = threading.Event()

    def clear(self):
        self.stopped = False

    def set(self):
        self.stopped = True
        self.second.set()

    def wait(self, timeout):
        self.calls += 1
        if self.calls == 2:
            if not self.second.wait(2.0):
                raise AssertionError("Second sample was never admitted")
        return self.stopped or self.calls > 2


class ClockedIPCWorker:
    """Real duplex connection with a controlled exposure/processing schedule."""

    def __init__(self):
        self.parent, self.child = multiprocessing.Pipe()
        self.received = [threading.Event() for _ in range(4)]
        self.release = [threading.Event() for _ in range(4)]
        self.responses = [None] * 4
        self.requests = []
        self.error = None
        self.worker = PerceptionWorker.__new__(PerceptionWorker)
        self.worker.connection = self.parent
        self.worker.process = SimpleNamespace(is_alive=lambda: True)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        try:
            for index in range(4):
                request = self.child.recv()
                if request is None:
                    return
                self.requests.append(request)
                self.received[index].set()
                if not self.release[index].wait(2.0):
                    raise AssertionError("IPC response was never released")
                observation = SimpleNamespace(
                    timestamp=self.responses[index], center=np.array([0.35, 0.18, 0.82])
                )
                self.child.send(
                    dict(
                        result=(observation, {"frame_id": index}),
                        events=[("test_exact_frame_archived", {"frame_id": index})],
                        worker_ms=60.0,
                        error=None,
                    )
                )
        except Exception as exc:
            self.error = exc

    def finish(self, index, clock, receipt, exposure):
        clock.set(receipt)
        self.responses[index] = exposure
        self.release[index].set()

    def close(self):
        for item in self.release:
            item.set()
        try:
            self.parent.send(None)
        except (BrokenPipeError, OSError):
            pass
        self.thread.join(2.0)
        self.parent.close()
        self.child.close()


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class PerceptionTimingTests(unittest.TestCase):
    def runtime(self):
        result = runtime_module.Runtime.__new__(runtime_module.Runtime)
        result.lock = threading.RLock()
        result.observation_lock = threading.Lock()
        result.cancel = threading.Event()
        result.grasp_started = False
        result.active = False
        result.plant = None
        result.tracking_observation = None
        result.tracking_error = None
        result.tracking_vision = None
        result.events = []
        result.event = lambda phase, **fields: result.events.append((phase, fields))
        result.pose_history = SimpleNamespace(
            nearest=lambda timestamp: {"timestamp": timestamp}
        )
        return result

    def test_contended_phase_observation_refreshes_watchdog_before_next_background_response(
        self,
    ):
        clock = ExposureClock()
        ipc = ClockedIPCWorker()
        result = self.runtime()
        result.perception = ipc.worker
        result.tracking_stop = ControlledStop()
        baseline = SimpleNamespace(center=np.array([0.35, 0.18, 0.82]))
        errors = []
        phase_done = threading.Event()
        phase_entered = threading.Event()

        def start():
            try:
                result.start_tracking({}, baseline)
            except Exception as exc:
                errors.append(exc)

        def phase():
            phase_entered.set()
            try:
                result.observe_frame({}, "approach", True)
            except Exception as exc:
                errors.append(exc)
            finally:
                phase_done.set()

        starter = threading.Thread(target=start)
        phase_thread = threading.Thread(target=phase)
        try:
            with patch.object(current_runtime.time, "monotonic", clock.monotonic):
                starter.start()
                self.assertTrue(ipc.received[0].wait(2.0))
                ipc.finish(0, clock, 100.20, 100.0)
                starter.join(2.0)
                self.assertFalse(starter.is_alive())
                self.assertTrue(ipc.received[1].wait(2.0))
                # Background owns the actual serialization lock. Approach
                # must wait and cannot issue a concurrent worker request.
                phase_thread.start()
                self.assertTrue(phase_entered.wait(2.0))
                self.assertFalse(result.observation_lock.acquire(blocking=False))
                self.assertFalse(ipc.received[2].is_set())
                ipc.finish(1, clock, 100.28, 100.10)
                self.assertTrue(ipc.received[2].wait(2.0))
                self.assertEqual(ipc.requests[2][1], "approach")
                result.tracking_stop.second.set()
                ipc.finish(2, clock, 100.43, 100.24)
                self.assertTrue(phase_done.wait(2.0))
                self.assertTrue(ipc.received[3].wait(2.0))
                # At a motor tick, the old background exposure is 460 ms
                # old, but the already validated phase exposure is 320 ms.
                clock.set(100.56)
                try:
                    result.tracking_guard()
                finally:
                    ipc.finish(3, clock, 100.62, 100.42)
                result.tracking_thread.join(2.0)
                self.assertFalse(errors)
                self.assertIsNone(result.tracking_error)
                self.assertEqual(
                    [request[1] for request in ipc.requests],
                    [
                        "motion-tracking",
                        "motion-tracking",
                        "approach",
                        "motion-tracking",
                    ],
                )
                self.assertEqual(
                    sum(
                        name == "test_exact_frame_archived" for name, _ in result.events
                    ),
                    4,
                )
        finally:
            result.tracking_stop.set()
            for item in ipc.release:
                item.set()
            if starter.is_alive():
                starter.join(2.0)
            if phase_thread.is_alive():
                phase_thread.join(2.0)
            result.stop_tracking()
            ipc.close()
        self.assertIsNone(ipc.error)

    def acquisition_runtime(self, clock, frames):
        result = self.runtime()
        result.exposure_config = lambda config, timestamp: config
        result.frame_prefetch = ImmediateExecutor()
        result.archive_executor = ImmediateExecutor()
        result.archived = []
        result.archive_frame = lambda frame, phase: result.archived.append(
            frame["meta"]["frame_id"]
        )
        result.tracker = SimpleNamespace(
            state={"frame_id": 10}, accept=lambda *args: None
        )
        pending = Future()
        pending.set_result(frames[0])
        result.pending_frame = pending
        calls = []

        def capture(**kwargs):
            calls.append(1)
            return frames[min(len(calls), len(frames) - 1)]

        result.camera = SimpleNamespace(frame=capture)
        return result, calls

    def test_parked_prefetch_is_discarded_using_exposure_age_before_geometry(self):
        clock = ExposureClock()
        clock.set(100.25)
        frames = [
            {"meta": {"frame_id": 11}, "timestamp": 99.90},
            {"meta": {"frame_id": 12}, "timestamp": 100.18},
        ]
        result, calls = self.acquisition_runtime(clock, frames)
        seen = []

        def fit(frame, *args, **kwargs):
            seen.append(frame["meta"]["frame_id"])
            clock.sleep(0.06)
            return SimpleNamespace(timestamp=frame["timestamp"]), {}

        with (
            patch.object(current_runtime.time, "monotonic", clock.monotonic),
            patch.object(current_runtime.time, "sleep", clock.sleep),
            patch.object(runtime_module, "geometry", side_effect=fit),
        ):
            observation, _ = result._observe_frame({}, "approach", True)
        self.assertEqual(seen, [12])
        self.assertEqual(result.archived, [12])
        self.assertEqual(observation.timestamp, 100.18)
        self.assertLessEqual(clock.monotonic() - observation.timestamp, 0.4)
        self.assertEqual(len(calls), 2)

    def test_repeated_frames_abort_after_bounded_attempts_without_inference_or_archive(
        self,
    ):
        clock = ExposureClock()
        clock.set(100.25)
        frame = {"meta": {"frame_id": 10}, "timestamp": 100.10}
        result, calls = self.acquisition_runtime(clock, [frame])
        with (
            patch.object(current_runtime.time, "monotonic", clock.monotonic),
            patch.object(current_runtime.time, "sleep", clock.sleep),
            patch.object(runtime_module, "geometry") as geometry,
        ):
            with self.assertRaisesRegex(RuntimeError, "acquisition budget"):
                result._observe_frame({}, "approach", True)
        self.assertEqual(len(calls), 3)
        geometry.assert_not_called()
        self.assertEqual(result.archived, [])
        rejected = [
            fields
            for phase, fields in result.events
            if phase == "camera_acquisition_rejected"
        ]
        self.assertEqual(len(rejected[0]["attempts"]), 3)
        self.assertTrue(all(not item["distinct"] for item in rejected[0]["attempts"]))


if __name__ == "__main__":
    unittest.main()
