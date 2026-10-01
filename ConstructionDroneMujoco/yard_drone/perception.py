"""CPU incident inference and worker badge decoding from camera pixels.

The incident CNN learns from rendered training crops. MobileCLIP embeddings
remain available for training comparisons; YOLOS is an optional detector.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image


class VisionUnavailable(RuntimeError):
    pass


PROMPTS = {
    "fire": ["a photo of flames burning on the ground", "a fire with orange flames and smoke"],
    "oil leak": ["a photo of black oil spilled on concrete", "a dark liquid oil spill on the ground"],
    "water puddle": ["a photo of a water puddle on the ground", "a wet patch on concrete"],
    "shadow": ["a photo of a shadow on the ground", "a dark shadow cast on concrete"],
    "ordinary ground": ["a photo of bare concrete ground", "an empty construction yard floor"],
    "construction materials": ["a photo of construction materials", "a stack of building supplies on the ground"],
    "person": ["a photo of a construction worker", "a person standing on the ground"],
    "smoke": ["a photo of smoke rising from the ground", "a cloud of smoke in a construction yard"],
}


def _rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Vision input must be an RGB camera image")
    if array.dtype != np.uint8:
        raise ValueError("Vision input must use uint8 RGB pixels")
    return Image.fromarray(array[:, :, :3]).convert("RGB")


def _normalize(features):
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-10)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _hash(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


class VisionService:
    def __init__(self, modelsroot=None):
        self.models_root = Path(modelsroot or os.environ.get("YARD_DRONE_VISION_MODELS", "/opt/models/vision"))
        self._lock = threading.RLock()
        self._error = None
        self._ready = False
        self._last_latency_ms = None
        self._calls = 0
        self._probe = None
        self._cnn = None
        self._last_coco_at = float("-inf")
        self._coco_enabled = os.environ.get("YARD_DRONE_COCO_PERSON_DETECTION", "0") == "1"
        import cv2
        parameters = cv2.aruco.DetectorParameters()
        parameters.errorCorrectionRate = 0.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._marker_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        try:
            self._load()
            self._ready = True
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"

    def _load(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        manifest_path = Path(__file__).resolve().parent.parent / "vision-models.json"
        if not manifest_path.is_file():
            raise VisionUnavailable("vision-models.json is missing")
        manifest = json.loads(manifest_path.read_text())
        for model in manifest["models"]:
            for asset in model["files"]:
                path = self.models_root / model["directory"] / asset["local"]
                if not path.is_file():
                    raise VisionUnavailable(f"Missing vision asset {path}; run download_vision_models.py")
                if _hash(path) != asset["sha256"]:
                    raise VisionUnavailable(f"Vision asset checksum mismatch: {path.name}")

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.environ.get("YARD_DRONE_VISION_THREADS", "2")))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._vision = ort.InferenceSession(str(self.models_root / "mobileclip-s0/vision_model.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        text = ort.InferenceSession(str(self.models_root / "mobileclip-s0/text_model.onnx"), options,
                                    providers=["CPUExecutionProvider"])
        tokenizer = Tokenizer.from_file(str(self.models_root / "mobileclip-s0/tokenizer.json"))
        tokenizer.enable_truncation(max_length=77)
        tokenizer.enable_padding(length=77, pad_id=0, pad_token="!")
        descriptions = [prompt for prompts in PROMPTS.values() for prompt in prompts]
        encodings = tokenizer.encode_batch(descriptions)
        values = {"input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                  "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64)}
        text_inputs = {input.name: values[input.name] for input in text.get_inputs()}
        output_names = [output.name for output in text.get_outputs()]
        embeddings = text.run(["text_embeds"] if "text_embeds" in output_names else None, text_inputs)[0]
        embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))
        self._labels = list(PROMPTS)
        self._text_embeddings = _normalize(np.stack([embeddings[i:i+2].mean(axis=0)
                                                     for i in range(0, len(embeddings), 2)]))
        # Prompt embeddings are frozen after model inference; text weights can
        # be released while physics and the image encoder continue running.
        del text
        self._people = ort.InferenceSession(str(self.models_root / "yolos-tiny/model.onnx"), options,
                                           providers=["CPUExecutionProvider"])
        self._clip_config = json.loads((self.models_root / "mobileclip-s0/preprocessor_config.json").read_text())
        self._people_config = json.loads((self.models_root / "yolos-tiny/preprocessor_config.json").read_text())
        detector_config = json.loads((self.models_root / "yolos-tiny/config.json").read_text())
        self._person_id = next(int(index) for index, label in detector_config["id2label"].items() if label == "person")
        probe_setting = os.environ.get("YARD_DRONE_INCIDENT_PROBE")
        if probe_setting:
            from .probe import IncidentProbe
            probe_path = Path(probe_setting)
            self._probe = IncidentProbe(probe_path, _hash(self.models_root / "mobileclip-s0/vision_model.onnx"))
        cnn_path = Path(os.environ.get("YARD_DRONE_INCIDENT_CNN", str(Path(__file__).with_name("incident_cnn.onnx"))))
        if not cnn_path.is_file():
            raise VisionUnavailable("Incident CNN is missing; train or restore incident_cnn.onnx and its metadata")
        from .cnn import CNNIncidentClassifier
        self._cnn = CNNIncidentClassifier(cnn_path)

    def status(self):
        return {"ready": self._ready, "error": self._error,
                "classifier": "Learned RGB yard incident CNN" if self._cnn else "MobileCLIP-S0 + trained yard classifier" if self._probe else "Apple MobileCLIP-S0, zero-shot",
                "incidentCNN": self._cnn.status() if self._cnn else None,
                "incidentProbe": self._probe.status() if self._probe else None,
                "personDetector": "OpenCV ArUco 4x4_50 id7 worker badge, decoded camera pixels",
                "optionalPersonDetector": "COCO YOLOS-tiny, quantized ONNX",
                "optionalPersonDetectorEnabled": self._coco_enabled,
                "executionProvider": "CPUExecutionProvider", "input": "RGB camera pixels",
                "scoreMeaning": "classifier score; simulator calibration only" if self._cnn or self._probe else "relative text/image similarity; uncalibrated",
                "modelsRoot": str(self.models_root), "lastInferenceMs": self._last_latency_ms,
                "inferenceCalls": self._calls}

    def _require_ready(self):
        if not self._ready:
            raise VisionUnavailable(self._error or "Vision models are unavailable")

    def embed(self, image):
        """Return frozen normalized MobileCLIP image features from RGB pixels."""
        self._require_ready()
        with self._lock:
            frame = _rgb(image)
            size = self._clip_config["size"]["shortest_edge"]
            width, height = frame.size
            ratio = size / min(width, height)
            frame = frame.resize((round(width * ratio), round(height * ratio)), Image.Resampling.BILINEAR)
            crop = self._clip_config["crop_size"]
            left, top = (frame.width - crop["width"]) // 2, (frame.height - crop["height"]) // 2
            frame = frame.crop((left, top, left + crop["width"], top + crop["height"]))
            pixels = np.asarray(frame, dtype=np.float32).transpose(2, 0, 1)[None] / 255.
            names = [output.name for output in self._vision.get_outputs()]
            features = self._vision.run(["image_embeds"] if "image_embeds" in names else None,
                                       {"pixel_values": pixels})[0]
            return _normalize(np.asarray(features, dtype=np.float32))[0]

    def classify(self, image, context=None):
        """Learned labels from RGB pixels only, with explicit abstention."""
        started = time.perf_counter()
        self._require_ready()
        with self._lock:
            if self._cnn:
                result = self._cnn.classify(image)
            else:
                features = self.embed(image)
                if self._probe:
                    result = self._probe.classify(features)
                    self._calls += 1
                    self._last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
                    result["inferenceMs"] = self._last_latency_ms
                    return result
                similarities = features @ self._text_embeddings.T
                scores = _softmax(100. * similarities)
                order = np.argsort(scores)[::-1]
                best, runner_up = int(order[0]), int(order[1])
                confidence = float(scores[best])
                margin = float(similarities[best] - similarities[runner_up])
                accepted = confidence >= .60 and float(similarities[best]) >= .24 and margin >= .025
                result = {"label": self._labels[best] if accepted else "unknown visual change",
                    "confidence": confidence, "uncertain": not accepted, "source": "mobileclip-s0-zero-shot",
                    "scoreType": "relative similarity, uncalibrated", "margin": margin,
                    "candidates": [{"label": self._labels[int(i)], "confidence": float(scores[i]),
                                    "similarity": float(similarities[i])} for i in order]}
            self._calls += 1
            self._last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            result["inferenceMs"] = self._last_latency_ms
            return result

    def detect_people(self, image):
        """Return visible worker badge boxes, with optional pretrained detection.

        ArUco is a real camera fiducial decoder. A successful result identifies
        a tagged worker; it does not recognize arbitrary untagged people. Its
        binary integrity score is not a neural detection probability. Slow COCO
        inference is opt-in and limited to one attempt per six seconds.
        """
        import cv2
        started = time.perf_counter()
        frame = _rgb(image)
        pixels = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY)
        with self._lock:
            corners, ids, _ = self._marker_detector.detectMarkers(pixels)
            scale = 1.
            if ids is None:
                corners, ids, _ = self._marker_detector.detectMarkers(
                    cv2.resize(pixels, None, fx=2., fy=2., interpolation=cv2.INTER_NEAREST))
                scale = 2.
            detections = []
            if ids is not None:
                for points, marker_id in zip(corners, ids.reshape(-1)):
                    if int(marker_id) != 7:
                        continue
                    points = points.reshape(4, 2) / scale
                    x1, y1 = points.min(axis=0)
                    x2, y2 = points.max(axis=0)
                    detections.append({"box": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": 1., "label": "tagged worker", "source": "worker-fiducial",
                        "markerId": 7, "dictionary": "DICT_4X4_50", "uncertain": False,
                        "scoreType": "binary exact marker decode; not an ML probability",
                        "corners": points.astype(float).tolist()})
            self._last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self._calls += 1
            if detections or not self._coco_enabled:
                return detections
            if time.monotonic() - self._last_coco_at < 6.:
                return []
            self._last_coco_at = time.monotonic()
        return self.detect_people_coco(frame)

    def detect_people_coco(self, image):
        """Explicit, potentially slow COCO person inference from image pixels."""
        self._require_ready()
        started = time.perf_counter()
        with self._lock:
            frame = _rgb(image)
            width, height = frame.size
            settings = self._people_config
            ratio = min(settings["size"]["shortest_edge"] / min(width, height),
                        settings["size"]["longest_edge"] / max(width, height))
            resized = frame.resize((round(width * ratio), round(height * ratio)), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32) / 255.
            pixels = (pixels - np.array(settings["image_mean"], dtype=np.float32)) / np.array(settings["image_std"], dtype=np.float32)
            pixels = pixels.transpose(2, 0, 1)[None]
            inputs = {"pixel_values": pixels}
            if any(input.name == "pixel_mask" for input in self._people.get_inputs()):
                inputs["pixel_mask"] = np.ones((1, resized.height, resized.width), dtype=np.int64)
            outputs = dict(zip([output.name for output in self._people.get_outputs()], self._people.run(None, inputs)))
            logits = outputs.get("logits")
            boxes = outputs.get("pred_boxes")
            if logits is None or boxes is None:
                raise VisionUnavailable(f"Unexpected detector outputs: {list(outputs)}")
            probabilities = _softmax(logits[0])
            threshold = float(os.environ.get("YARD_DRONE_PERSON_CONFIDENCE", ".35"))
            detections = []
            for probability, box in zip(probabilities, boxes[0]):
                confidence = float(probability[self._person_id])
                if int(np.argmax(probability)) != self._person_id or confidence < threshold:
                    continue
                cx, cy, bw, bh = np.asarray(box, dtype=float)
                xyxy = [max(0., (cx-bw/2)*width), max(0., (cy-bh/2)*height),
                        min(float(width), (cx+bw/2)*width), min(float(height), (cy+bh/2)*height)]
                if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                    continue
                detections.append({"box": xyxy, "confidence": confidence, "label": "person",
                                   "source": "yolos-tiny-coco", "uncertain": confidence < .70})
            self._calls += 1
            self._last_latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return sorted(detections, key=lambda detection: detection["confidence"], reverse=True)
