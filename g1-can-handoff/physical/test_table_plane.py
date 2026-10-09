"""Archived tape fits must generalize to unseen pixels on the tabletop."""

from pathlib import Path
import json
import unittest
import numpy as np
from table_plane import fit_table_plane


class TablePlaneTests(unittest.TestCase):
    def surfaces(self):
        rng = np.random.default_rng(3)

        def points(width, n):
            x = rng.uniform(-0.3, 0.3, n)
            y = rng.uniform(-width, width, n)
            return np.c_[x, y, 0.7 + rng.normal(0, 0.0007, n)]

        return points(0.015, 1500), points(0.3, 5000), points(0.3, 5000)

    def test_narrow_surface_support_is_rejected_despite_low_tape_residual(self):
        tape, _, held = self.surfaces()
        with self.assertRaisesRegex(ValueError, "too narrow"):
            fit_table_plane(tape, tape, held)

    def test_disagreeing_heldout_plane_is_rejected(self):
        tape, train, held = self.surfaces()
        held[:, 2] += held[:, 0] * 0.02
        with self.assertRaisesRegex(ValueError, "disagree"):
            fit_table_plane(tape, train, held)

    def test_objects_and_invalid_depth_do_not_rotate_the_table(self):
        tape, train, held = self.surfaces()
        train = np.r_[train, np.full((100, 3), np.nan), train[:500] + [0, 0, 0.08]]
        normal, center, _, quality = fit_table_plane(tape, train, held)
        self.assertAlmostEqual(center[2], 0.7, places=3)
        self.assertGreater(abs(normal[2]), 0.9999)
        self.assertLess(quality["heldout_rmse_m"], 0.001)

    def test_recorded_calibrations_explain_the_wide_table(self):
        # Reference normals identify the broad tabletop cluster only; fitting
        # uses the disjoint train pixels, never this held-out test selection.
        references = {
            20903: [-0.01062727, -0.59383839, -0.80451417],
            38920: [-0.00210495, -0.60784334, -0.79405421],
            41836: [-0.01090847, -0.57688212, -0.81675458],
        }
        audit = json.loads(
            (
                Path(__file__).parent / "offline/evidence/table-plane-audit.json"
            ).read_text()
        )
        old_normals = {row["frame_id"]: row["oldnormal"] for row in audit}
        with np.load(Path(__file__).parent / "fixtures/table-plane.npz") as data:
            for fid, reference in references.items():
                with self.subTest(frame=fid):
                    tape = data[f"tape_{fid}"]
                    train = data[f"train_{fid}"]
                    held = data[f"holdout_{fid}"]
                    normal, center, _, _ = fit_table_plane(tape, train, held)
                    held = held[
                        np.isfinite(held).all(1) & (held[:, 2] > 0.1) & (held[:, 2] < 2)
                    ]
                    distance = held @ reference
                    offset = np.median(tape @ reference)
                    table = held[np.abs(distance - offset) < 0.006]
                    self.assertGreater(len(table), 1000)
                    if fid in (20903, 41836):
                        # Original tape normal passes its strip but fails these
                        # independent broad-table pixels before the repair.
                        old = np.array(old_normals[fid])
                        offset = np.median(tape @ old)
                        self.assertGreater(
                            np.sqrt(np.mean((table @ old - offset) ** 2)), 0.006
                        )
                    errors = (table - center) @ normal
                    self.assertLess(np.sqrt(np.mean(errors**2)), 0.004)
                    self.assertLess(np.quantile(np.abs(errors), 0.95), 0.006)


if __name__ == "__main__":
    unittest.main()
