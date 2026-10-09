"""Measured visual/depth tracking with an explicit acquisition policy.

Selection proposes pixels; accept() commits only after full geometry/freshness
checks. color_depth uses no classifier. yolo_once requires one initial identity
and stops on uncertainty. yolo_reacquire retains the earlier semantic policy.
"""

import hashlib
import numpy as np
from scipy.ndimage import binary_closing, binary_erosion, label

from yolo_client import YoloClient, fresh, validate_result


def color_bins(image):
    values = image.astype(np.uint16) // 32
    return values[:, :, 0] * 64 + values[:, :, 1] * 8 + values[:, :, 2]


def histogram(bins, mask):
    values = np.bincount(bins[mask], minlength=512).astype(float)
    return values / max(1, values.sum())


class TargetTracker:
    def __init__(self, detector=None, recheck_interval_s=1.0, identity_mode=None):
        from perception_policy import target_identity_mode

        self.identity_mode = (
            target_identity_mode() if identity_mode is None else identity_mode
        )
        if self.identity_mode not in ("color_depth", "yolo_once", "yolo_reacquire"):
            raise ValueError("Unknown target identity mode")
        if self.identity_mode != "yolo_reacquire":
            recheck_interval_s = None
        if recheck_interval_s is not None and not 0 < recheck_interval_s <= 1.0:
            raise ValueError("Semantic recheck interval must be None or within0..1s")
        self.detector = (
            detector
            if detector is not None
            else (None if self.identity_mode == "color_depth" else YoloClient())
        )
        self.recheck_interval_s = recheck_interval_s
        self.state = None
        self.pending = None

    def reset(self):
        self.state = None
        self.pending = None

    def reacquire(self, frame, config, world, require_table, expected_center_xy):
        """Replace an uncertain, uncommitted track using this exact image."""
        fresh(frame)
        if (
            self.pending is None
            or self.pending["frame_id"] != frame["meta"]["frame_id"]
        ):
            raise RuntimeError(
                "Reacquisition requires the current pending tracking frame"
            )
        self.pending = None
        return self._semantic(
            frame,
            config,
            world,
            require_table,
            expected_center_xy,
            None,
            "cylinder_fit_uncertain",
        )

    def _semantic(
        self, frame, config, world, require_table, expected_center_xy, result, reason
    ):
        if self.identity_mode == "color_depth":
            return self._color_depth(
                frame, config, world, require_table, expected_center_xy, reason
            )
        if self.identity_mode == "yolo_once" and self.state is not None:
            raise RuntimeError(
                "Visual/depth tracking uncertain: "
                + reason
                + "; YOLO reacquisition disabled"
            )
        from camera_input import target_coke_mask

        detection = validate_result(
            frame, self.detector.detect(frame) if result is None else result
        )
        mask, diagnostic = target_coke_mask(
            frame, config, world, require_table=require_table, result=detection
        )
        diagnostic = {
            **diagnostic,
            "tracking": {
                "mode": "yolo",
                "reason": reason,
                "semantic_frame_id": frame["meta"]["frame_id"],
            },
        }
        self.pending = {
            "frame_id": frame["meta"]["frame_id"],
            "mask": mask,
            "histogram": histogram(color_bins(frame["color_bgr"]), mask),
            "diagnostic": diagnostic,
            "semantic_timestamp": frame["timestamp"],
            "semantic_frame_id": frame["meta"]["frame_id"],
        }
        return mask, diagnostic

    def _color_depth(
        self, frame, config, world, require_table, expected_center_xy, reason
    ):
        """Acquire measured red can pixels; YOLO cannot enable or veto this mode."""
        from camera_input import target_red_mask

        fresh(frame)
        depth = frame["depth_m"]
        roi = np.isfinite(world).all(axis=2)
        if require_table:
            roi &= (world[:, :, 2] > config["table_height_m"] + 0.012) & (
                world[:, :, 2]
                < config["table_height_m"] + config["can_height_m"] + 0.02
            )
        center = expected_center_xy
        if center is None and self.state is not None:
            center = self.state["center_xy"]
        if center is not None:
            center = np.asarray(center, dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all():
                raise RuntimeError("Tracker expected center is invalid")
            roi &= (
                (world[:, :, 0] - center[0]) ** 2 + (world[:, :, 1] - center[1]) ** 2
            ) < (config["can_radius_m"] + 0.04) ** 2
        # Preserve the single-component rule even though geometry constrains
        # the ROI. The legacy mask function permits merged components for
        # lifted semantic targets; it is too permissive for initial identity.
        from scipy.ndimage import label as components

        rgb = frame["color_bgr"][..., ::-1]
        r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
        red = (
            roi
            & np.isfinite(depth)
            & (depth > 0.1)
            & (depth < 2.5)
            & (r > 75)
            & (r > g * 1.7)
            & (r > b * 1.7)
        )
        from camera_input import close_red_label

        labels, _ = components(close_red_label(red))
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if np.count_nonzero(sizes > 100) != 1:
            raise RuntimeError("Color/depth can target absent/ambiguous")
        mask = target_red_mask(rgb, depth, roi=roi)
        diagnostic = {
            "detector": "color_depth",
            "image_sha256": hashlib.sha256(frame["color_bgr"].tobytes()).hexdigest(),
            "tracking": {
                "mode": "color_depth_acquisition",
                "reason": reason,
                "identity_source": "color_depth",
                "semantic_frame_id": None,
            },
        }
        self.pending = {
            "frame_id": frame["meta"]["frame_id"],
            "mask": mask,
            "histogram": histogram(color_bins(frame["color_bgr"]), mask),
            "diagnostic": diagnostic,
            "semantic_timestamp": None,
            "semantic_frame_id": None,
        }
        return mask, diagnostic

    def select(
        self,
        frame,
        config,
        world,
        require_table=True,
        expected_center_xy=None,
        result=None,
    ):
        fresh(frame)
        self.pending = None
        if frame["meta"]["camera_serial"] != config["camera_serial"]:
            raise RuntimeError("Tracked camera serial changed")
        if self.state is None:
            return self._semantic(
                frame,
                config,
                world,
                require_table,
                expected_center_xy,
                result,
                "initial_identity",
            )
        state = self.state
        if (
            frame["meta"]["frame_id"] <= state["frame_id"]
            or frame["timestamp"] <= state["timestamp"]
        ):
            raise RuntimeError(
                "Tracker requires a distinct increasing exact camera frame"
            )
        if (
            self.recheck_interval_s is not None
            and frame["timestamp"] - state["semantic_timestamp"]
            >= self.recheck_interval_s
        ):
            return self._semantic(
                frame,
                config,
                world,
                require_table,
                expected_center_xy,
                result,
                "periodic_recheck",
            )
        depth = frame["depth_m"]
        center = np.asarray(
            state["center_xy"] if expected_center_xy is None else expected_center_xy
        )
        if center.shape != (2,) or not np.isfinite(center).all():
            raise RuntimeError("Tracker expected center is invalid")
        distance2 = (world[:, :, 0] - center[0]) ** 2 + (
            world[:, :, 1] - center[1]
        ) ** 2
        # Nonfinite X/Y already fail the squared-distance comparison. Retain
        # an explicit Z check without reducing a full HxWx3 temporary.
        valid = (
            np.isfinite(depth)
            & (depth > 0.1)
            & (depth < 2.5)
            & np.isfinite(world[:, :, 2])
        )
        roi = (
            valid
            & (distance2 < (config["can_radius_m"] + 0.04) ** 2)
            & (np.abs(depth - state["depth_median"]) < 0.12)
        )
        if require_table:
            roi &= (world[:, :, 2] > config["table_height_m"] + 0.012) & (
                world[:, :, 2]
                < config["table_height_m"] + config["can_height_m"] + 0.02
            )
        # Bound work by ALL eligible measured-world pixels, retaining other
        # possible components for ambiguity checks. Four false border pixels
        # preserve the existing two-iteration closing at crop boundaries.
        rows, columns = np.nonzero(roi)
        if not len(rows):
            return self._semantic(
                frame,
                config,
                world,
                require_table,
                expected_center_xy,
                result,
                "insufficient_visual_depth_support",
            )
        ys = slice(
            max(0, int(rows.min()) - 4), min(depth.shape[0], int(rows.max()) + 5)
        )
        xs = slice(
            max(0, int(columns.min()) - 4), min(depth.shape[1], int(columns.max()) + 5)
        )
        full_world = world
        roi = roi[ys, xs]
        depth = depth[ys, xs]
        world = world[ys, xs]
        bins = color_bins(frame["color_bgr"][ys, xs])
        # Keep measured pixels whose color belonged to the semantically
        # confirmed can. Closing groups printing/finger fragments only within
        # this bounded measured-depth/world ROI; it never invents fit points.
        selected = roi & (state["histogram"][bins] >= 0.002)
        if selected.sum() >= 40:
            z = world[selected, 2]
            z95 = float(np.quantile(z, 0.95))
            span = z95 - float(np.quantile(z, 0.05))
            if require_table:
                top = config["table_height_m"] + config["can_height_m"]
                if span > 0.03:
                    top = min(top, z95)
            else:
                top = z95 if span > 0.03 else None
            if top is not None:
                selected &= world[:, :, 2] < top - 0.01
            if selected.sum() >= 40:
                selected &= (
                    np.abs(depth - float(np.median(depth[selected])))
                    <= config["can_radius_m"] + 0.01
                )
        components, count = label(binary_closing(selected, iterations=2) & roi)
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]
        largest = int(order[0])
        minimum = max(40, int(state["initial_support"] * 0.2))
        reason = None
        if sizes[largest] < minimum:
            reason = "insufficient_visual_depth_support"
        elif len(order) > 1 and sizes[order[1]] >= max(40, sizes[largest] * 0.2):
            reason = "ambiguous_visual_depth_components"
        else:
            mask = binary_erosion(selected & (components == largest), iterations=2)
            support = int(mask.sum())
            agreement = float(np.sqrt(state["histogram"] * histogram(bins, mask)).sum())
            full_mask = np.zeros(frame["depth_m"].shape, dtype=bool)
            full_mask[ys, xs] = mask
            mask = full_mask
            if support < minimum:
                reason = "insufficient_eroded_support"
            elif agreement < 0.72:
                reason = "visual_identity_uncertain"
            elif (
                mask[:6].any()
                or mask[-6:].any()
                or mask[:, :6].any()
                or mask[:, -6:].any()
            ):
                reason = "tracked_target_clipped"
        if reason is not None:
            return self._semantic(
                frame,
                config,
                full_world,
                require_table,
                expected_center_xy,
                result,
                reason,
            )
        fresh(frame)
        diagnostic = {
            **state["diagnostic"],
            "detector": "coke_visual_depth_tracker",
            "image_sha256": hashlib.sha256(frame["color_bgr"].tobytes()).hexdigest(),
            "tracking": {
                "mode": "visual_depth",
                "semantic_frame_id": state["semantic_frame_id"],
                "semantic_age_s": (
                    None
                    if state["semantic_timestamp"] is None
                    else frame["timestamp"] - state["semantic_timestamp"]
                ),
                "visual_agreement": agreement,
                "support_pixels": support,
                "roi_shape": list(roi.shape),
            },
        }
        self.pending = {
            **state,
            "frame_id": frame["meta"]["frame_id"],
            "mask": mask,
            "diagnostic": diagnostic,
        }
        return mask, diagnostic

    def accept(self, observation, vision, frame):
        fresh(frame)
        if (
            self.pending is None
            or self.pending["frame_id"] != frame["meta"]["frame_id"]
        ):
            raise RuntimeError("Tracker acceptance does not match pending exact frame")
        # The caller has validated the same fit/table/workspace before this
        # commit; the tracker cannot convert a proposed mask into motion.
        center = np.asarray(observation.center[:2], dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise RuntimeError("Accepted tracker center invalid")
        if (
            self.state is not None
            and np.linalg.norm(center - self.state["center_xy"]) > 0.04
        ):
            raise RuntimeError("Tracked target center differs by over40mm")
        state = self.pending
        state.update(
            timestamp=frame["timestamp"],
            center_xy=center,
            depth_median=float(np.median(frame["depth_m"][state["mask"]])),
            initial_support=(
                int(state["mask"].sum())
                if self.state is None
                else self.state["initial_support"]
            ),
        )
        self.state = state
        self.pending = None
