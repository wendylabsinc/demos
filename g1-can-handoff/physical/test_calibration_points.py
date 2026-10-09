import json
import unittest
import numpy as np
from calibration_points import extract


class MarkerFreeCalibrationTests(unittest.TestCase):
    def sample(self):
        y, x = np.mgrid[:30, :30]
        rays = np.stack([x / 30, y / 30, np.ones_like(x)], axis=-1)
        frame = dict(
            depth_m=np.ones((30, 30)),
            rays=rays,
            metadata_json=json.dumps(dict(camera_serial="test", frame_id=42)),
        )
        points = []
        for i, (u, v) in enumerate(
            [(3, 3), (25, 3), (3, 25), (25, 25), (8, 10), (17, 20)]
        ):
            points.append(
                dict(
                    pixel_uv=[u, v],
                    world_xyz_m=(rays[v, u] + [0.2, -0.1, 0.3]).tolist(),
                    validation=i >= 4,
                )
            )
        return frame, dict(camera_serial="test", frame_id=42, points=points)

    def test_marker_free_points_fit_but_never_commission(self):
        frame, points = self.sample()
        result = extract(frame, points)
        self.assertFalse(result["commissioned"])
        np.testing.assert_allclose(
            np.array(result["camera_to_world"])[:3, 3], [0.2, -0.1, 0.3], atol=1e-12
        )

    def test_wrong_frame_and_depth_discontinuities_rejected(self):
        frame, points = self.sample()
        points["frame_id"] = 41
        with self.assertRaisesRegex(ValueError, "exact captured frame"):
            extract(frame, points)
        points["frame_id"] = 42
        frame["depth_m"][3, 3] = 1.5
        with self.assertRaisesRegex(ValueError, "depth edge"):
            extract(frame, points)


if __name__ == "__main__":
    unittest.main()
