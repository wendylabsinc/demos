"""Real spawned worker IPC cannot admit stale/future observations or hide errors."""

import os
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from perception_worker import PerceptionWorker, worker_main


def fake_worker(connection, token):
    connection.send({"ready": True})
    while True:
        request = connection.recv()
        if request is None:
            break
        config, *_ = request
        pose = {}
        if config.get("request_pose"):
            connection.send({"pose_request": config["exposure"]})
            pose = connection.recv()
        connection.send(
            {
                "result": (
                    SimpleNamespace(timestamp=time.monotonic() - config.get("age", 0)),
                    {"pid": os.getpid()},
                ),
                "events": [("test_evidence", {"saved": True})],
                "worker_ms": 1,
                "error": pose.get("error") or config.get("error"),
            }
        )
    connection.close()


class WorkerTests(unittest.TestCase):
    def test_real_worker_reports_ready_only_after_camera_warmup(self):
        order = []
        connection = Mock()
        connection.recv.return_value = None
        connection.send.side_effect = lambda message: order.append(("send", message))
        with (
            patch("camera_input.Camera"),
            patch("target_tracker.TargetTracker"),
            patch("latest_frame.LatestFrameBuffer") as constructor,
        ):
            buffer = constructor.return_value
            buffer.wait_ready.side_effect = lambda **kw: order.append(("warm", kw))
            worker_main(connection, None)
            self.assertEqual([row[0] for row in order], ["warm", "send"])
            self.assertEqual(order[1][1], {"ready": True})
            self.assertAlmostEqual(order[0][1]["max_age_s"], 0.3)
            buffer.close.assert_called_once()
        connection.send.reset_mock()
        with (
            patch("camera_input.Camera"),
            patch("target_tracker.TargetTracker"),
            patch("latest_frame.LatestFrameBuffer") as constructor,
        ):
            constructor.return_value.wait_ready.side_effect = RuntimeError(
                "Latest camera readiness timeout"
            )
            with self.assertRaisesRegex(RuntimeError, "readiness timeout"):
                worker_main(connection, None)
            connection.send.assert_not_called()
            constructor.return_value.close.assert_called_once()

    def test_publish_checked_result_before_event_relay_but_never_publish_stale(self):
        worker = PerceptionWorker(None, target=fake_worker)
        order = []
        try:
            worker.observe(
                {},
                "test",
                True,
                None,
                lambda *a, **kw: order.append("event"),
                publish=lambda *a: order.append("publish"),
            )
            self.assertEqual(order[0], "publish")
            self.assertIn("event", order)
            publisher = Mock()
            with self.assertRaisesRegex(RuntimeError, "stale on receipt"):
                worker.observe(
                    {"age": 0.5}, "test", True, None, Mock(), publish=publisher
                )
            publisher.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "outside pickup path"):
                worker.observe(
                    {},
                    "test",
                    True,
                    None,
                    Mock(),
                    publish=Mock(side_effect=RuntimeError("outside pickup path")),
                )
        finally:
            worker.close()

    def test_child_can_request_exact_exposure_pose_without_another_dds_reader(self):
        worker = PerceptionWorker(None, target=fake_worker)
        provider = Mock(return_value={"timestamp": 12.5})
        try:
            worker.observe(
                {"request_pose": True, "exposure": 12.5},
                "test",
                True,
                None,
                Mock(),
                pose_at=provider,
            )
            provider.assert_called_once_with(12.5)
            with self.assertRaisesRegex(RuntimeError, "history unavailable"):
                worker.observe(
                    {"request_pose": True, "exposure": 12.5}, "test", True, None, Mock()
                )
        finally:
            worker.close()

    def test_stalled_request_expires_before_tracking_join_deadline(self):
        worker = PerceptionWorker.__new__(PerceptionWorker)
        worker.process = Mock()
        worker.connection = Mock()
        worker.connection.poll.return_value = False
        with self.assertRaisesRegex(RuntimeError, "response timeout"):
            worker.observe({}, "test", True, None, Mock())
        self.assertLess(worker.connection.poll.call_args.args[0], 6.0)

    def test_spawn_round_trip_and_shutdown(self):
        worker = PerceptionWorker(None, target=fake_worker)
        event = Mock()
        try:
            _, vision = worker.observe({}, "test", True, None, event)
            self.assertNotEqual(vision["pid"], os.getpid())
            event.assert_any_call("test_evidence", saved=True)
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())

    def test_stale_and_future_observations_are_rejected_in_parent(self):
        worker = PerceptionWorker(None, target=fake_worker)
        try:
            for age in (0.5, -1.0):
                with (
                    self.subTest(age=age),
                    self.assertRaisesRegex(RuntimeError, "stale on receipt"),
                ):
                    worker.observe({"age": age}, "test", True, None, Mock())
        finally:
            worker.close()

    def test_error_is_not_converted_to_success(self):
        worker = PerceptionWorker(None, target=fake_worker)
        try:
            with self.assertRaisesRegex(RuntimeError, "geometry failed"):
                worker.observe({"error": "geometry failed"}, "test", True, None, Mock())
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
