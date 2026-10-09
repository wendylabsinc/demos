"""Keep the new NX2 failures repeatable after the robot is unavailable."""

from pathlib import Path
import tempfile
import unittest
from offline.replay_real_stops import extract_archives, run


class RealStopsReplayTests(unittest.TestCase):
    def test_native_waist_and_planning_deadline_failures_reproduce_without_motion(self):
        source = Path(__file__).parent / "offline/validation/tuned-real-replay-20261008"
        with tempfile.TemporaryDirectory() as directory:
            destination = extract_archives(source, directory)
            result = run(destination)
        self.assertTrue(result["all_errors_reproduced"])
        self.assertFalse(result["hardware_used"])
        self.assertFalse(result["physical_pickup_verified"])
        self.assertIn("Waist moved beyond", result["waist"]["error"])
        self.assertEqual(
            result["deadline"]["error"],
            "ValueError: Grasp orientation search time limit reached",
        )
        self.assertGreater(result["deadline"]["total_validation_seconds"], 30.0)
        self.assertTrue(result["deadline"]["candidate_inputs_identical"])


if __name__ == "__main__":
    unittest.main()
