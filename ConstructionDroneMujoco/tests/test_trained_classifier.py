"""Probe math checks and independent validation of the generated model report.

Synthetic vectors test optimizer correctness only. Report checks inspect the
actual rendered-data artifact produced by train_incident_classifier.py.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from train_incident_classifier import CLASSES, evaluate, fit_probe, save_probe, spatial_cells


def test_probe_learns_from_training_features_and_generalizes():
    # Controlled, independent feature sets verify learning, with no hard-coded
    # class weights in the fitted probe. This is not a vision accuracy test.
    rng = np.random.default_rng(17)
    centres = np.zeros((3, 512))
    centres[:, :3] = np.eye(3) * 2

    def sample(count):
        labels = np.arange(count) % 3
        features = centres[labels] + rng.normal(0, .08, (count, 512))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        return features.astype(np.float32), labels

    train_x, train_y = sample(60)
    held_x, held_y = sample(30)
    weights, bias = fit_probe(train_x, train_y, epochs=500)
    result = evaluate(held_x, held_y, weights, bias)
    assert result["confusionMatrix"] == [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    assert result["accuracy"] == 1
    assert np.isfinite(weights).all() and np.linalg.norm(weights) > 0


def test_spatial_partitions_are_disjoint_and_obstacles_excluded():
    from yard_drone.environment import NX, NY
    blocked = np.zeros((NY, NX), dtype=bool)
    blocked[:, :8] = True
    environment = SimpleNamespace(blocked=blocked)
    train = spatial_cells(environment, "train")
    validation = spatial_cells(environment, "validation")
    assert train and validation and not set(train).intersection(validation)
    assert all((x + y) % 2 == 0 for x, y in train)
    assert all((x + y) % 2 == 1 for x, y in validation)
    assert all(x > -5.25 for points in [*train.values(), *validation.values()] for x, _ in points)


def test_probe_export_is_numeric_and_loads_without_pickle(tmp_path):
    weights = np.zeros((512, 3), np.float32)
    bias = np.zeros(3, np.float32)
    path = tmp_path / "classifier.npz"
    save_probe(path, weights, bias, {"version": 1, "runtimeInputs": "image embedding only"})
    with np.load(path, allow_pickle=False) as model:
        assert model["weights"].shape == (512, 3)
        assert model["classes"].tolist() == list(CLASSES)
        assert json.loads(model["metadata"].item())["runtimeInputs"] == "image embedding only"


def test_rendered_classifier_report_matches_held_out_results():
    root = Path(__file__).resolve().parent.parent
    path = root / "yard_drone/incident_classifier.npz"
    if not path.is_file():
        pytest.skip("Generate the actual RGB training artifact first")
    with np.load(path, allow_pickle=False) as model:
        weights, bias = model["weights"], model["bias"]
        metadata = json.loads(model["metadata"].item())
        validation_features, validation_labels = model["validation_features"], model["validation_labels"]
        assert weights.shape == (512, 3) and bias.shape == (3,)
        assert np.isfinite(weights).all() and np.isfinite(bias).all()
        assert np.linalg.norm(weights) > 0 and model["classes"].tolist() == list(CLASSES)
    assert metadata["trainingSeed"] != metadata["validationSeed"]
    assert len(metadata["encoderSha256"]) == 64
    assert metadata["trainingRgbSha256"] != metadata["validationRgbSha256"]
    assert not {tuple(cell) for cell in metadata["trainingCells"]}.intersection(
        tuple(cell) for cell in metadata["validationCells"])
    result = metadata["validation"]
    assert evaluate(validation_features, validation_labels, weights, bias) == result
    matrix = np.array(result["confusionMatrix"])
    assert matrix.shape == (3, 3) and matrix.sum() == result["examples"]
    assert result["examples"] >= 40 and matrix.sum(axis=1).min() > 0
    assert result["accuracy"] == pytest.approx(np.trace(matrix) / matrix.sum())
    for index, label in enumerate(CLASSES):
        assert result["recall"][label] == pytest.approx(matrix[index, index] / matrix[index].sum())
    runtime = result["runtimeEvaluation"]
    assert runtime["accepted"] + runtime["abstained"] == result["examples"]
    # Verify the deployed adapter's actual uncertainty decisions, rather than
    # assuming raw argmax accuracy also measures usable incident hypotheses.
    from yard_drone.probe import IncidentProbe
    probe = IncidentProbe(path, metadata["encoderSha256"])
    runtime_matrix = np.zeros((3, 4), dtype=int)
    for features, truth in zip(validation_features, validation_labels):
        prediction = probe.classify(features)
        column = 3 if prediction["abstained"] else CLASSES.index(prediction["predictedLabel"])
        runtime_matrix[truth, column] += 1
    assert runtime_matrix.tolist() == runtime["confusionMatrix"]
