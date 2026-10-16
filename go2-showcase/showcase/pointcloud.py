"""Project the Go2 bridge's body-frame voxel cloud into obstacle sectors.

The bridge publishes an accumulated occupied-voxel map, not a laser scan.
Missing sectors therefore remain unknown rather than becoming clear space.
"""
import math
from dataclasses import dataclass


SCAN_COUNT = 72
SCAN_INCREMENT = 2 * math.pi / SCAN_COUNT


@dataclass(frozen=True)
class ProjectedScan:
    ranges: tuple
    angle_min: float = -math.pi
    angle_increment: float = SCAN_INCREMENT

    @property
    def coverage(self):
        return tuple(math.isfinite(value) for value in self.ranges)


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def project_cloud(message, odometry):
    """Return 72 five-degree obstacle sectors from `/utlidar/cloud_base`.

    PointCloud2 validation and the body-height slice match the Go2 simulator
    examples. Odometry levels roll and pitch while preserving robot-relative
    heading. A sector without an occupied return is `inf` (unknown).
    """
    if message.header.frame_id != "base_link":
        raise ValueError("Expected /utlidar/cloud_base in base_link")
    if (type(message.width) is not int or type(message.height) is not int
            or not 0 < message.width * message.height <= 200000
            or not 12 <= message.point_step <= 256
            or not message.width * message.point_step <= message.row_step
            or len(message.data) != message.row_step * message.height
            or len(message.data) > 16000000):
        raise ValueError("Invalid PointCloud2 dimensions or row stride")

    fields = {}
    for field in message.fields:
        if field.name in ("x", "y", "z"):
            size = 4 if field.datatype == 7 else 8
            if (field.name in fields or field.count != 1 or field.datatype not in (7, 8)
                    or field.offset < 0 or field.offset + size > message.point_step):
                raise ValueError("Invalid PointCloud2 XYZ layout")
            fields[field.name] = (field.offset, "f" if field.datatype == 7 else "d")
    if set(fields) != {"x", "y", "z"}:
        raise ValueError("PointCloud2 requires XYZ fields")

    if (odometry.header.frame_id != "odom" or odometry.child_frame_id != "base_link"
            or abs(_stamp_ns(message.header.stamp) - _stamp_ns(odometry.header.stamp)) > 100_000_000):
        raise ValueError("Cloud requires matching odom to base_link orientation")

    q = odometry.pose.pose.orientation
    values = (q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Invalid cloud orientation")
    norm = math.hypot(*values)
    if abs(norm - 1) > 0.01:
        raise ValueError("Invalid cloud orientation")

    import numpy as np
    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype({"names": list("xyz"),
                      "formats": [endian + fields[name][1] for name in "xyz"],
                      "offsets": [fields[name][0] for name in "xyz"],
                      "itemsize": message.point_step})
    points = np.ndarray((message.height, message.width), dtype=dtype,
                        buffer=bytes(message.data),
                        strides=(message.row_step, message.point_step))
    x, y, z = (points[name].astype(np.float64).reshape(-1) for name in "xyz")

    qx, qy, qz, qw = (value / norm for value in values)
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    level = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]]) @ rotation
    x, y, z = level @ np.vstack((x, y, z))

    radii = np.hypot(x, y)
    valid = (np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
             & (z >= -0.2) & (z <= 0.4) & (radii > 0) & (radii <= 12.0))
    indices = np.floor((np.arctan2(y[valid], x[valid]) + math.pi)
                       / SCAN_INCREMENT + 0.5).astype(int) % SCAN_COUNT
    ranges = np.full(SCAN_COUNT, np.inf)
    np.minimum.at(ranges, indices, radii[valid])
    return ProjectedScan(tuple(float(value) for value in ranges))
