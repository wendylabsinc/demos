"""Train/validation separation and runtime image tensor checks."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yard_drone.cnn import image_tensor


def test_image_tensor_preserves_rgb_pixel_scale_and_rejects_other_inputs():
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[..., 0] = 255; image[..., 1] = 91
    tensor = image_tensor(image, 96)
    assert tensor.shape == (1, 3, 96, 96) and tensor.dtype == np.float32
    assert np.all(tensor[0, 0] == 255) and np.all(tensor[0, 1] == 91)
    with pytest.raises(ValueError, match="RGB"):
        image_tensor(np.zeros((20, 20)), 96)


def test_internal_split_is_stratified_disjoint_and_reproducible():
    pytest.importorskip("torch")
    from train_cnn_incident import internal_split
    rows = [(Path(f"{label}-{index}.png"), label) for label in range(3) for index in range(30)]
    fit, selection = internal_split(rows, 42)
    assert len(fit) == 72 and len(selection) == 18
    assert not set(fit).intersection(selection)
    assert internal_split(rows, 42) == (fit, selection)
    assert [sum(label == i for _, label in selection) for i in range(3)] == [6, 6, 6]


def test_imbalanced_backgrounds_receive_inverse_frequency_weights():
    pytest.importorskip("torch")
    from train_cnn_incident import class_weights
    rows = [(Path(f"{label}-{index}.png"), label) for label, count in enumerate((30, 30, 120)) for index in range(count)]
    weights = class_weights(rows).numpy()
    assert weights.mean() == pytest.approx(1.)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[0] / weights[2] == pytest.approx(4.)


def test_negative_crop_sampling_uses_actual_image_and_stays_in_bounds():
    pytest.importorskip("torch")
    from generate_cnn_negatives import crop_box
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, 150:] = 200
    rng = np.random.default_rng(812)
    for mode in ("random", "edge", "worker"):
        for _ in range(20):
            x, y, width, height = crop_box(image, rng, mode, [-50, 300])
            assert 20 <= width <= 200 and width == height
            assert 0 <= x <= 320-width and 0 <= y <= 240-height


def test_export_normalization_and_actual_onnx_runtime_agree(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from train_cnn_incident import IncidentCNN
    import onnxruntime as ort
    model = IncidentCNN().eval()
    features = torch.from_numpy(image_tensor(Image.new("RGB", (60, 40), (220, 91, 14)), 64))
    with torch.no_grad():
        expected = model(features).numpy()
    path = tmp_path / "test.onnx"
    torch.onnx.export(model, features, str(path), input_names=["rgb"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"rgb": features.numpy()})[0]
    assert np.allclose(expected, actual, atol=1e-5)


def test_domain_guard_abstains_without_changing_learned_candidate_scores():
    from yard_drone.cnn import CNNIncidentClassifier
    class Session:
        def run(self, names, inputs):
            return [np.array([[9., 0., 0.]], dtype=np.float32)]
    classifier = CNNIncidentClassifier.__new__(CNNIncidentClassifier)
    classifier.metadata = {"classes": ["fire", "oil", "other"], "inputDomain": {"aspectMax": 1.5}}
    classifier.size = 96; classifier.temperature = 1.
    classifier.confidence_threshold = .7; classifier.margin_threshold = .2
    classifier._session = Session()
    square = classifier.classify(np.zeros((96, 96, 3), dtype=np.uint8))
    elongated = classifier.classify(np.zeros((240, 40, 3), dtype=np.uint8))
    assert square["label"] == "fire" and not square["outOfDomain"]
    assert elongated["label"] == "other" and elongated["uncertain"] and elongated["outOfDomain"]
    assert elongated["predictedLabel"] == "fire" and elongated["candidates"] == square["candidates"]
    assert "represented training range" in elongated["reason"]


def test_validation_coverage_counts_out_of_domain_crops():
    pytest.importorskip("torch")
    from train_cnn_incident import evaluate
    logits = np.array([[9., 0., 0.], [0., 9., 0.]])
    metrics = evaluate(logits, np.array([0, 1]), confidence=.7, margin=.2, domain_mask=[True, False])
    assert metrics["accuracy"] == 1. and metrics["acceptedExamples"] == 1
    assert metrics["coverage"] == .5 and metrics["outOfDomainExamples"] == 1
