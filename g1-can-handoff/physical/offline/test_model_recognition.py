"""Real saved checkpoint recognition, on CPU, with the deployed decoder.

Separate profile because Torch is large; no model/fixture downloads or robot.
Float32 CPU recognition does not reproduce CUDA FP16 latency/bitwise output.
"""

import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "yolo"
sys.path.insert(0, str(ROOT))
from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates
from qualification import fixtures, check
from confidence import MIN_CONFIDENCE


class SavedCheckpointRecognition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ultralytics import YOLO

        source = json.loads((ROOT / "model/SOURCE.json").read_text())
        model = ROOT / "model/coke-can.pt"
        if hashlib.sha256(model.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError("Pinned checkpoint changed")
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.torch = torch
        cls.model = YOLO(model)
        if cls.model.names != {0: "Coca-Cola"}:
            raise RuntimeError("Wrong checkpoint class")
        cls.network = cls.model.model.fuse(verbose=False).eval().cpu().float()
        cls.inputs = {
            row["image"]: (row, image) for row, image in fixtures(ROOT / "fixtures")
        }

    def assert_recognized(self, name):
        row, image = self.inputs[name]
        chw, scale, px, py = letterbox_bgr(image, 640)
        with self.torch.inference_mode():
            raw = self.network(
                self.torch.from_numpy(chw).float().div_(255).unsqueeze(0)
            )
            prediction = (raw[0] if isinstance(raw, (tuple, list)) else raw)[0]
            self.assertEqual(tuple(prediction.shape), (5, 8400))
            values, indices = prediction[4].topk(129, largest=True, sorted=True)
            candidates = prediction[:, indices[:128]].T.numpy()
            saturated = bool(values[-1] > MIN_CONFIDENCE)
        boxes = decode_one_class_candidates(
            candidates,
            confidence=MIN_CONFIDENCE,
            nms_iou=0.7,
            max_detections=21,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
            scale=scale,
            pad_x=px,
            pad_y=py,
        )
        boxes = [
            dict(
                xyxy=[b["x1"], b["y1"], b["x2"], b["y2"]],
                confidence=b["conf"],
                class_id=0,
                class_name="Coca-Cola",
            )
            for b in boxes
        ]
        check(row, boxes, saturated)


# Each real image gets an independently named test and outcome in the report.
manifest = json.loads((ROOT / "fixtures/qualification.json").read_text())
for row in manifest["fixtures"]:
    name = row["image"]

    def test(self, image=name):
        self.assert_recognized(image)

    setattr(
        SavedCheckpointRecognition,
        "test_" + name.replace("-", "_").replace(".", "_"),
        test,
    )

if __name__ == "__main__":
    unittest.main()
