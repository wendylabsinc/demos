"""CPU ONNX incident inference from camera RGB pixels alone."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def image_tensor(image, size):
    if not isinstance(image, Image.Image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
            raise ValueError("Incident CNN requires uint8 RGB camera pixels")
        image = Image.fromarray(image[:, :, :3])
    pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    return pixels.transpose(2, 0, 1)[None]


class CNNIncidentClassifier:
    def __init__(self, modelpath):
        import onnxruntime as ort
        self.path = Path(modelpath)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text())
        if self.metadata.get("version") != 1 or self.metadata.get("classes") != ["fire", "oil", "other"]:
            raise ValueError("Unsupported incident CNN metadata")
        if self.metadata.get("modelSha256") != file_hash(self.path):
            raise ValueError("Incident CNN checksum mismatch")
        self.size = int(self.metadata["imageSize"])
        if self.size not in (64, 96):
            raise ValueError("Incident CNN image size must be 64 or 96")
        settings = self.metadata["abstention"]
        self.temperature = float(self.metadata["temperature"])
        self.confidence_threshold = float(settings["confidenceThreshold"])
        self.margin_threshold = float(settings["marginThreshold"])
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid incident CNN temperature")
        if not (0 <= self.confidence_threshold <= 1 and 0 <= self.margin_threshold <= 1):
            raise ValueError("Invalid incident CNN confidence thresholds")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        if [item.name for item in self._session.get_inputs()] != ["rgb"]:
            raise ValueError("Incident CNN requires one RGB tensor input")

    def classify(self, image):
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            pixels = np.asarray(image)
            if pixels.ndim != 3:
                raise ValueError("Incident CNN requires an RGB camera image")
            height, width = pixels.shape[:2]
        aspect = max(width / height, height / width)
        aspect_max = float(self.metadata.get("inputDomain", {}).get("aspectMax", float("inf")))
        outside_domain = aspect > aspect_max
        inputs = image_tensor(image, self.size)
        logits = self._session.run(["logits"], {"rgb": inputs})[0]
        if logits.shape != (1, 3) or not np.isfinite(logits).all():
            raise RuntimeError("Incident CNN returned invalid logits")
        logits = logits[0].astype(np.float64) / self.temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1]
        best, second = int(order[0]), int(order[1])
        confidence = float(probabilities[best])
        margin = confidence - float(probabilities[second])
        accepted = confidence >= self.confidence_threshold and margin >= self.margin_threshold and not outside_domain
        predicted = self.metadata["classes"][best]
        return {"label": predicted if accepted else "other", "predictedLabel": predicted,
                "confidence": confidence, "uncertain": not accepted or predicted == "other",
                "abstained": not accepted, "abstention": not accepted, "margin": margin,
                "outOfDomain": outside_domain, "cropAspect": aspect,
                "reason": f"Crop aspect {aspect:.3f} exceeds represented training range {aspect_max:.3f}" if outside_domain else None,
                "candidates": [{"label": self.metadata["classes"][int(index)], "confidence": float(probabilities[index])}
                               for index in order], "source": "trained-yard-cnn-onnx",
                "scoreType": "temperature-scaled classifier score; simulator calibration only"}

    def status(self):
        return {"ready": True, "architecture": "Learned three-convolution RGB incident CNN",
                "executionProvider": "CPUExecutionProvider", "input": "RGB camera pixels only",
                "modelSha256": self.metadata["modelSha256"], "validation": self.metadata.get("validation"),
                "calibration": self.metadata.get("calibration"), "scope": self.metadata.get("scope"),
                "inputDomain": self.metadata.get("inputDomain"),
                "confidenceThreshold": self.confidence_threshold, "marginThreshold": self.margin_threshold}
