"""Original native table failures must become accepted observations, not rejection."""

import unittest
from offline.table_geometry_replay import inputs, replay, frame, table_quality


class NativeTableGeometryAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.poses = inputs()

    def test_original_frozen_pose_reproduces_both_native_table_errors(self):
        for case in self.manifest["cases"]:
            with self.subTest(attempt=case["attempt"]):
                result = replay(case, self.poses, correct=False)
                self.assertFalse(result["accepted"])
                self.assertEqual(
                    result["results"][-1]["frame_id"], case["fault_frame_id"]
                )
                self.assertEqual(
                    result["results"][-1]["error"], "Table plane differs by over 5 mm"
                )

    def test_exposure_pose_correction_accepts_both_original_faults_repeatably(self):
        for case in self.manifest["cases"]:
            runs = [replay(case, self.poses, correct=True) for _ in range(5)]
            with self.subTest(attempt=case["attempt"]):
                self.assertTrue(all(r["accepted"] for r in runs))
                self.assertTrue(all(r == runs[0] for r in runs))
                self.assertEqual(
                    runs[0]["results"][-1]["frame_id"], case["fault_frame_id"]
                )
                for result in runs[0]["results"]:
                    self.assertGreaterEqual(result["vision"]["pixels"], 40)
                    self.assertLessEqual(result["vision"]["fit_error_m"], 0.006)
                    self.assertLessEqual(abs(result["table_median_error_mm"]), 5)
                    self.assertGreaterEqual(result["table_support"], 100)
                    self.assertLessEqual(result["pose_skew_ms"], 20)

    def test_native_calibration_frames_keep_disjoint_table_validation(self):
        for case in self.manifest["cases"]:
            quality = table_quality(frame(case["frames"][0]))
            with self.subTest(attempt=case["attempt"]):
                self.assertLessEqual(quality["heldout_rmse_m"], 0.004)
                self.assertLessEqual(quality["heldout_normal_angle_deg"], 0.5)
                self.assertGreaterEqual(quality["heldout_points"], 500)
                self.assertEqual(
                    quality["method"],
                    "tape_associated_broad_tabletop_with_disjoint_pixel_validation",
                )


if __name__ == "__main__":
    unittest.main()
