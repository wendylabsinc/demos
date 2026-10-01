"""A learned incident head over frozen camera-image embeddings.

The artifact contains a fitted multinomial linear classifier. It receives only
the encoder's normalized 512-dimensional feature vector. Scene labels, camera
poses and incident injection state are not runtime inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np


class ProbeUnavailable(ValueError):
    """The learned artifact is missing, invalid or uses another image encoder."""


class IncidentProbe:
    def __init__(self, path, encoder_hash):
        self.path = Path(path)
        if not re.fullmatch(r"[0-9a-f]{64}", str(encoder_hash)):
            raise ProbeUnavailable("Expected encoder hash must be a SHA-256 hex string")
        try:
            with np.load(self.path, allow_pickle=False) as artifact:
                required = {"weights", "bias", "classes", "metadata"}
                if not required.issubset(artifact.files):
                    raise ProbeUnavailable("Incident probe is missing required arrays")
                weights = np.array(artifact["weights"], dtype=np.float32, copy=True)
                bias = np.array(artifact["bias"], dtype=np.float32, copy=True)
                classes = artifact["classes"].tolist()
                raw_metadata = artifact["metadata"]
                if raw_metadata.shape != () or raw_metadata.dtype.kind not in ("U", "S"):
                    raise ProbeUnavailable("Incident probe metadata must be a JSON string scalar")
                raw = raw_metadata.item()
                metadata = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except ProbeUnavailable:
            raise
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProbeUnavailable(f"Cannot load incident probe {self.path.name}: {exc}") from exc
        if weights.shape != (512, 3) or bias.shape != (3,):
            raise ProbeUnavailable("Incident probe must contain 512x3 weights and 3 bias values")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ProbeUnavailable("Incident probe contains non-finite parameters")
        if classes != ["fire", "oil", "other"]:
            raise ProbeUnavailable("Incident probe class order must be fire, oil, other")
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ProbeUnavailable("Unsupported incident probe metadata version")
        if metadata.get("encoderSha256") != encoder_hash:
            raise ProbeUnavailable("Incident probe image encoder checksum does not match")
        if "classes" in metadata and metadata["classes"] != classes:
            raise ProbeUnavailable("Incident probe class metadata does not match arrays")
        abstention = metadata.get("abstention", {})
        if not isinstance(abstention, dict):
            raise ProbeUnavailable("Incident probe abstention settings must be an object")
        try:
            confidence_threshold = float(abstention.get("confidenceThreshold", .70))
            margin_threshold = float(abstention.get("marginThreshold", .20))
        except (TypeError, ValueError) as exc:
            raise ProbeUnavailable("Incident probe abstention thresholds must be numbers") from exc
        if not (.0 <= confidence_threshold <= 1. and .0 <= margin_threshold <= 1.):
            raise ProbeUnavailable("Incident probe abstention thresholds must be within 0 and 1")
        weights.setflags(write=False)
        bias.setflags(write=False)
        self.weights, self.bias, self.classes = weights, bias, classes
        self.metadata = metadata
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def status(self):
        return {"ready": True, "artifact": self.path.name,
                "architecture": "Frozen MobileCLIP-S0 + learned multinomial linear probe",
                "input": "Normalized 512D camera image embedding only",
                "encoderSha256": self.metadata["encoderSha256"],
                "training": self.metadata.get("training"),
                "validation": self.metadata.get("validation"),
                "scoreMeaning": "uncalibrated classifier softmax, simulator domain",
                "confidenceThreshold": self.confidence_threshold,
                "marginThreshold": self.margin_threshold,
                "thresholdBasis": self.metadata.get("abstention", {}).get("basis", "conservative settings; not calibrated")}

    def classify(self, embedding):
        features = np.asarray(embedding, dtype=np.float32)
        if features.shape == (1, 512):
            features = features.reshape(512)
        if features.shape != (512,) or not np.isfinite(features).all():
            raise ValueError("Incident probe input must contain 512 finite embedding values")
        norm = float(np.linalg.norm(features))
        if not .98 <= norm <= 1.02:
            raise ValueError("Incident probe input must be an L2-normalized image embedding")
        logits = features @ self.weights + self.bias
        probabilities = np.exp(logits.astype(np.float64) - float(np.max(logits)))
        probabilities /= float(probabilities.sum())
        order = np.argsort(probabilities)[::-1]
        best, second = int(order[0]), int(order[1])
        confidence = float(probabilities[best])
        margin = confidence - float(probabilities[second])
        accepted = confidence >= self.confidence_threshold and margin >= self.margin_threshold
        predicted = self.classes[best]
        return {"label": predicted if accepted else "other", "predictedLabel": predicted,
                "confidence": confidence, "uncertain": not accepted or predicted == "other",
                "source": "mobileclip-s0-trained-yard-probe",
                "scoreType": "uncalibrated learned classifier softmax, simulator domain",
                "candidates": [{"label": self.classes[int(index)], "confidence": float(probabilities[index])}
                               for index in order],
                "margin": margin, "abstained": not accepted,
                "confidenceThreshold": self.confidence_threshold,
                "marginThreshold": self.margin_threshold}
