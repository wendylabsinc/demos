"""Validate learned artifact integrity and camera-feature abstention."""
import json

import numpy as np
import pytest

from yard_drone.probe import IncidentProbe, ProbeUnavailable

ENCODER_HASH = "fcbd153d1aa1314fb72ea39b20c37e0572e7e7b05359b51f3efee5d682658472"


def artifact(path, **updates):
    weights = np.zeros((512, 3), dtype=np.float32)
    weights[0, 0] = 8.
    weights[1, 1] = 8.
    weights[2, 2] = 8.
    arrays = {"weights": weights, "bias": np.zeros(3, dtype=np.float32),
              "classes": np.array(["fire", "oil", "other"]),
              "metadata": np.array(json.dumps({"version": 1, "encoderSha256": ENCODER_HASH}))}
    arrays.update(updates)
    np.savez_compressed(path, **arrays)
    return path


def test_probe_requires_matching_encoder_and_finite_parameters(tmp_path):
    path = artifact(tmp_path / "valid.npz")
    with pytest.raises(ProbeUnavailable, match="checksum"):
        IncidentProbe(path, "0" * 64)
    weights = np.zeros((512, 3), dtype=np.float32)
    weights[0, 0] = np.nan
    with pytest.raises(ProbeUnavailable, match="non-finite"):
        IncidentProbe(artifact(tmp_path / "nan.npz", weights=weights), ENCODER_HASH)
    with pytest.raises(ProbeUnavailable, match="512x3"):
        IncidentProbe(artifact(tmp_path / "shape.npz", weights=np.zeros((511, 3))), ENCODER_HASH)
    with pytest.raises(ProbeUnavailable, match="class order"):
        IncidentProbe(artifact(tmp_path / "classes.npz", classes=np.array(["oil", "fire", "other"])), ENCODER_HASH)
    with pytest.raises(ProbeUnavailable, match="version"):
        IncidentProbe(artifact(tmp_path / "version.npz", metadata=np.array(json.dumps({"version": 2}))), ENCODER_HASH)
    with pytest.raises(ProbeUnavailable):
        IncidentProbe(artifact(tmp_path / "pickle.npz", weights=np.array({"secret": "invalid"}, dtype=object)), ENCODER_HASH)


def test_probe_classifies_embeddings_and_abstains_without_evidence(tmp_path):
    probe = IncidentProbe(artifact(tmp_path / "probe.npz"), ENCODER_HASH)
    for index, label in enumerate(["fire", "oil", "other"]):
        features = np.zeros(512, dtype=np.float32)
        features[index] = 1.
        result = probe.classify(features)
        assert result["label"] == label and result["confidence"] > .99
        assert result["uncertain"] == (label == "other")
        assert sum(candidate["confidence"] for candidate in result["candidates"]) == pytest.approx(1.)
    ambiguous = np.zeros(512, dtype=np.float32)
    ambiguous[10] = 1.
    result = probe.classify(ambiguous)
    assert result["label"] == "other" and result["uncertain"] and result["abstained"]
    with pytest.raises(ValueError, match="normalized"):
        probe.classify(np.zeros(512))
    with pytest.raises(ValueError, match="512 finite"):
        probe.classify(np.ones(511))
    assert probe.status()["scoreMeaning"] == "uncalibrated classifier softmax, simulator domain"


def test_probe_accepts_artifact_abstention_settings(tmp_path):
    metadata = {"version": 1, "encoderSha256": ENCODER_HASH,
                "abstention": {"confidenceThreshold": .8, "marginThreshold": .3}}
    probe = IncidentProbe(artifact(tmp_path / "probe.npz", metadata=np.array(json.dumps(metadata))), ENCODER_HASH)
    assert probe.confidence_threshold == .8 and probe.margin_threshold == .3
    metadata["abstention"]["marginThreshold"] = float("nan")
    with pytest.raises(ProbeUnavailable, match="within"):
        IncidentProbe(artifact(tmp_path / "nan-threshold.npz", metadata=np.array(json.dumps(metadata))), ENCODER_HASH)
