import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from offline.stale_replay import cases, command_tick, replay
from perception_worker import PerceptionWorker


class NativeStaleReplayTests(unittest.TestCase):
    def test_all_nine_original_stops_reproduce_and_same_input500ms_handoff_passes_repeatedly(
        self,
    ):
        report = replay(500, "receipt", 3)
        self.assertEqual(report["scenarios"], 9)
        self.assertEqual(report["accepted"], 9)
        for case in report["cases"]:
            self.assertTrue(case["original400ms_rejected"])
            self.assertTrue(
                all(tick["recording_sink_calls"] > 0 for tick in case["ticks"])
            )

    def test_fixed_budget_preserves_future_uncertainty_staleness_and_no_resume_at_write_gate(
        self,
    ):
        for age, error, cancelled in [
            (-0.001, None, False),
            (0.501, None, False),
            (0.1, "identity uncertain", False),
            (0.1, None, True),
        ]:
            with self.subTest(age=age, error=error, cancelled=cancelled):
                result = command_tick(
                    100 - age, 100, 500, uncertain=error, cancelled=cancelled
                )
                self.assertFalse(result["accepted"])
                self.assertEqual(result["recording_sink_calls"], 0)

    def test_packet_availability_never_precedes_validated_result_or_geometry_completion(
        self,
    ):
        for case in cases():
            for packet in case["packets"]:
                self.assertGreaterEqual(
                    packet["receipt_at"], packet["validated_result_at"]
                )
                self.assertGreaterEqual(
                    packet["receipt_at"], packet["geometry_complete_at"]
                )

    def test_archived_frame_with_failed_geometry_cannot_publish_to_motion_guard(self):
        worker = PerceptionWorker.__new__(PerceptionWorker)
        worker.process = SimpleNamespace(is_alive=lambda: True)
        worker.connection = Mock()
        worker.connection.poll.return_value = True
        worker.connection.recv.return_value = dict(
            result=None,
            error="Physical target cylinder fit exceeds6mm",
            worker_ms=50.0,
            events=[
                ("phase_camera_evidence", {"saved": True}),
                ("perception_result", {"geometry": None, "error": "fit rejected"}),
            ],
        )
        publish = Mock()
        event = Mock()
        with self.assertRaisesRegex(RuntimeError, "cylinder fit"):
            worker.observe({}, "motion-tracking", True, None, event, publish=publish)
        publish.assert_not_called()
        event.assert_any_call("phase_camera_evidence", saved=True)


if __name__ == "__main__":
    unittest.main()
