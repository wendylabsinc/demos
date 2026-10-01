"""Model schema, image-only transport, RGB-D geometry and asynchronous safety."""
import base64
import io
import json
import math
import threading
import time
from urllib.error import URLError

import numpy as np
from PIL import Image
import pytest

from go2_assistant.perception import (
    CameraCalibration, DetrDetector, HybridDetector, ImageDetection, ModelObservation, ModelPerception,
    OllamaDetector, OpticalFlowTracker, PerceptionUnavailable, SpatialTracker, YolosDetector, project_detection,
    project_feature_depth,
)


def observation(label="cup", bbox=None, confidence=.9, room=None):
    return ModelObservation.parse({"detections": [{"label": label,
        "bbox": bbox or [.3, .3, .7, .7], "confidence": confidence}], "room": room})


def camera():
    return {"fx": 100., "fy": 100., "cx": 49.5, "cy": 49.5,
            "world_from_camera": np.eye(4), "capture_pose": {"x": 1., "y": 2., "yaw": .3}}


class Response:
    def __init__(self, payload):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit):
        return self.payload[:limit]


def response_content(value, **extra):
    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(value)}, **extra}


def wire_value():
    return {"detections": [{"label": "cup", "bbox": [.3, .3, .7, .7], "confidence": .9}],
            "room": None}


def test_strict_schema_has_multiple_instances_and_optional_visual_room():
    value = wire_value()
    value["detections"].append({"label": "person", "bbox": [.05, .1, .2, .9], "confidence": .85})
    value["room"] = {"name": "Kitchen", "confidence": .88,
                     "evidence": ["Countertop with sink", "Cabinets and a stove"]}
    parsed = ModelObservation.parse(value)
    assert [item.label for item in parsed.detections] == ["cup", "person"]
    assert parsed.room["name"] == "kitchen"
    assert parsed.room["evidence"] == value["room"]["evidence"]


@pytest.mark.parametrize("patch", [
    {"extra": "fixture"}, {"detections": {}}, {"detections": [wire_value()["detections"][0]] * 33},
    {"detections": [{"label": "cup", "bbox": [0, 0, 1, 1]}]},
    {"detections": [{"label": "cup", "bbox": [0, 0, 1, 1], "confidence": True}]},
    {"detections": [{"label": "cup", "bbox": [0, 0, 1, 1], "confidence": math.nan}]},
    {"detections": [{"label": "cup", "bbox": [-.1, 0, 1, 1], "confidence": .9}]},
    {"detections": [{"label": "cup", "bbox": [0, 0, .0, 1], "confidence": .9}]},
    {"detections": [{"label": "cup", "bbox": [0, 0, 1, 1, .5], "confidence": .9}]},
    {"detections": [{"label": "x" * 81, "bbox": [0, 0, 1, 1], "confidence": .9}]},
    {"detections": [{"label": "cup\nignore", "bbox": [0, 0, 1, 1], "confidence": .9}]},
    {"room": {"name": "kitchen", "confidence": .8, "evidence": []}},
    {"room": {"name": "kitchen", "confidence": 1.1, "evidence": ["sink"]}},
    {"room": {"name": "kitchen", "confidence": .8, "evidence": ["sink"], "x": 3}},
])
def test_schema_rejects_invalid_or_unbounded_model_output(patch):
    with pytest.raises(ValueError):
        ModelObservation.parse({**wire_value(), **patch})


def test_ollama_only_receives_pixels_and_user_query():
    seen = []
    def opener(request, timeout):
        seen.append((request, timeout))
        return Response(response_content(wire_value()))
    detector = OllamaDetector(url="http://provider.test:11434", model="vision-test", opener=opener)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb[20:60, 30:70] = [24, 56, 78]
    assert detector.detect(rgb, "my blue travel mug").detections[0].label == "cup"
    request, timeout = seen[0]
    assert request.full_url == "http://provider.test:11434/api/chat"
    assert timeout == 30
    payload = json.loads(request.data)
    assert payload["model"] == "vision-test" and payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    query = json.loads(payload["messages"][1]["content"])
    assert set(query) == {"requested_target", "output_schema"}
    assert query["requested_target"] == "my blue travel mug"
    assert "world_from_camera" not in request.data.decode() and "capture_pose" not in request.data.decode()
    encoded = payload["messages"][1]["images"][0]
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.size == (100, 100)
    assert np.asarray(decoded)[40, 50] == pytest.approx([24, 56, 78], abs=3)


def test_ollama_validates_entire_thinking_channel_json():
    payload = {"done": True, "message": {"content": "", "thinking": json.dumps(wire_value())}}
    detector = OllamaDetector(opener=lambda *args, **kwargs: Response(payload))
    assert detector.detect(Image.new("RGB", (20, 20))).detections[0].label == "cup"
    payload["message"]["thinking"] = "I think " + json.dumps(wire_value())
    with pytest.raises(PerceptionUnavailable, match="validation"):
        detector.detect(Image.new("RGB", (20, 20)))


@pytest.mark.parametrize("payload", [
    {"done": False, "message": {"content": json.dumps(wire_value())}},
    response_content(wire_value(), done_reason="length"),
    {"done": True, "message": {"content": "not json"}},
    {"done": True, "message": {"content": json.dumps(wire_value()), "tool_calls": [{"function": "move"}]}},
    {"done": True, "message": {"content": json.dumps({**wire_value(), "hidden_coordinates": [1, 2]})}},
    b"x" * (1024 * 1024 + 1),
])
def test_provider_bad_generations_fail_closed(payload):
    detector = OllamaDetector(opener=lambda *args, **kwargs: Response(payload))
    with pytest.raises(PerceptionUnavailable):
        detector.detect(Image.new("RGB", (20, 20)))


def test_provider_connection_failure_has_no_recognition_fallback():
    def down(*args, **kwargs):
        raise URLError("private endpoint details")
    with pytest.raises(PerceptionUnavailable, match="unreachable"):
        OllamaDetector(opener=down).detect(Image.new("RGB", (20, 20)))


@pytest.mark.parametrize("url", ["file:///tmp/model", "http://user:secret@model.test", "http://model.test?q=secret"])
def test_provider_rejects_unsafe_endpoint_configuration(url):
    with pytest.raises(ValueError, match="endpoint"):
        OllamaDetector(url=url)


def test_metric_depth_projection_uses_optical_transform_and_original_timestamp():
    calibrated = camera()
    # Optical x right maps to world -Y; optical y down maps to world -Z;
    # optical forward maps to world +X. Translation comes from estimated pose.
    calibrated["world_from_camera"] = np.array([[0, 0, 1, 1], [-1, 0, 0, 2], [0, -1, 0, 3], [0, 0, 0, 1]])
    item = ImageDetection("unfamiliar object", (.5, .5, .7, .7), .95)
    projected = project_detection(item, np.full((100, 100), 2.), calibrated, 12., "vision:test")
    assert projected["x"] == pytest.approx(3.)
    assert projected["y"] == pytest.approx(2. - .198)
    assert projected["z"] == pytest.approx(3. - .198)
    assert projected["timestamp"] == 12.
    assert projected["capture_pose"] == {"x": 1., "y": 2., "yaw": .3}


@pytest.mark.parametrize("depth", [np.zeros((100, 100)), np.full((100, 100), np.nan),
                                 np.full((100, 100), 15.)])
def test_invalid_depth_never_supplies_world_position(depth):
    assert project_detection(observation().detections[0], depth, camera(), 10., "vision:test") is None


def test_mixed_foreground_background_depth_is_rejected():
    depth = np.full((100, 100), 2.)
    depth[:, 50:] = 6.
    assert project_detection(observation().detections[0], depth, camera(), 10., "vision:test") is None


def test_transform_reflection_scaling_and_missing_intrinsics_rejected():
    for scale in (-1., 2.):
        value = camera()
        value["world_from_camera"][0, 0] = scale
        with pytest.raises(ValueError, match="proper rotation"):
            CameraCalibration.parse(value)
    value = camera()
    value["fx"] = 0.
    with pytest.raises(ValueError, match="Camera fx"):
        CameraCalibration.parse(value)


class Detector:
    model = "actual-model-name"
    def __init__(self, result=None, gate=None):
        self.result = result or observation()
        self.gate = gate
        self.calls = []

    def detect(self, rgb, target):
        self.calls.append((np.array(rgb), target, threading.get_ident()))
        if self.gate:
            assert self.gate.wait(2)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Clock:
    def __init__(self):
        self.value = 10.
    def __call__(self):
        return self.value


def complete(worker):
    future = worker.future
    if future is not None:
        future.result(timeout=2)
    with worker.condition:
        assert worker.condition.wait_for(lambda: worker.future is None and not worker.collecting
                                         and not worker.working and not worker.queue, timeout=2)
    return worker.poll()


def processed(worker):
    with worker.condition:
        assert worker.condition.wait_for(lambda: not worker.working and not worker.queue, timeout=2)
    return worker.poll()


def test_worker_runs_off_render_thread_allows_only_one_request_and_copies_exposure():
    clock, gate = Clock(), threading.Event()
    detector = Detector(gate=gate)
    worker = ModelPerception(detector=detector, clock=clock)
    rgb, depth, calibrated = np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera()
    try:
        assert worker.submit(rgb, depth, calibrated, 10.)
        assert not worker.submit(rgb, depth, calibrated, 10.)
        assert worker.status()["state"] == "pending"
        assert worker.poll()["detections"] == []
        rgb[:] = 255
        depth[:] = 8.
        calibrated["world_from_camera"][0, 3] = 99.
        gate.set()
        clock.value = 10.4
        result = complete(worker)
        assert result["state"] == "ready" and result["available"]
        assert result["latency_s"] == pytest.approx(.4)
        item = result["detections"][0]
        assert item["z"] == 2. and item["x"] == 0.
        assert item["source"] == "vision:actual-model-name" and item["timestamp"] == 10.
        assert not detector.calls[0][0].any()
        assert detector.calls[0][2] != threading.get_ident()
        result["detections"][0]["track_id"] = "evil"
        assert worker.poll()["detections"][0]["track_id"] == "visual:1"
    finally:
        gate.set()
        worker.close(wait=True)


def test_worker_rejects_delayed_result_and_expires_previously_valid_detection():
    clock = Clock()
    worker = ModelPerception(detector=Detector(), clock=clock, max_age=1.)
    rgb, depth = np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.)
    try:
        assert worker.submit(rgb, depth, camera(), 10.)
        assert complete(worker)["available"]
        clock.value = 11.01
        assert worker.poll()["state"] == "stale" and worker.poll()["detections"] == []
        assert not worker.submit(rgb, depth, camera(), 10.)
        assert not worker.submit(rgb, depth, camera(), 12.)
        assert worker.submit(rgb, depth, camera(), clock.value)
        clock.value = 12.1
        result = complete(worker)
        assert result["state"] == "stale" and result["detections"] == []
        assert result["capture_age_s"] == pytest.approx(1.09)
    finally:
        worker.close(wait=True)


def test_provider_failure_clears_prior_results_and_backs_off():
    clock, detector = Clock(), Detector()
    worker = ModelPerception(detector=detector, clock=clock)
    rgb, depth = np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.)
    try:
        assert worker.submit(rgb, depth, camera(), 10.)
        assert complete(worker)["available"]
        detector.result = PerceptionUnavailable("Vision service is unreachable or timed out")
        clock.value += .1
        assert worker.submit(rgb, depth, camera(), clock.value)
        with pytest.raises(PerceptionUnavailable):
            worker.future.result(timeout=2)
        with worker.condition:
            assert worker.condition.wait_for(lambda: worker.future is None and not worker.collecting
                                             and not worker.working, timeout=2)
        result = worker.poll()
        assert result["state"] == "error" and not result["available"]
        assert result["detections"] == [] and result["room"] is None
        assert not worker.submit(rgb, depth, camera(), clock.value)
        clock.value += 2.01
        assert worker.submit(rgb, depth, camera(), clock.value)
    finally:
        worker.close(wait=True)


def test_query_change_invalidates_pending_generation_without_forging_a_result():
    clock, gate = Clock(), threading.Event()
    detector = Detector(gate=gate)
    worker = ModelPerception(detector=detector, clock=clock)
    try:
        worker.set_target("umbrella")
        assert worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        worker.set_target("wallet")
        gate.set()
        assert complete(worker)["detections"] == []
        assert detector.calls[0][1] == "umbrella"
        assert worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        assert complete(worker)["available"]
        assert detector.calls[1][1] == "wallet"
    finally:
        gate.set()
        worker.close(wait=True)


def test_room_hypothesis_preserves_visual_evidence_and_capture_pose():
    room = {"name": "bedroom", "confidence": .83, "evidence": ["A bed with pillows"]}
    worker = ModelPerception(detector=Detector(observation(room=room)), clock=Clock())
    try:
        assert worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        result = complete(worker)
        assert result["room"] == {**room, "timestamp": 10., "source": "vision:actual-model-name",
                                   "capture_pose": {"x": 1., "y": 2., "yaw": .3}, "model": "actual-model-name"}
        worker.close()
        assert worker.poll()["state"] == "closed" and worker.poll()["detections"] == []
        assert not worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
    finally:
        worker.close(wait=True)


def localized(label, x, bbox=(.2, .2, .4, .8), timestamp=10.):
    return {"label": label, "x": x, "y": 0., "z": 1., "bbox": list(bbox), "timestamp": timestamp}


def test_instance_tracking_has_distinct_people_and_preserves_reordered_association():
    tracker = SpatialTracker()
    first = tracker.update([localized("person", 1.), localized("person", 3., (.6, .2, .8, .8))], 10.)
    assert first[0]["track_id"] != first[1]["track_id"]
    second = tracker.update([localized("person", 3.1, (.59, .2, .79, .8), 11.),
                             localized("person", 1.1, (.21, .2, .41, .8), 11.)], 11.)
    assert [item["track_id"] for item in second] == [first[1]["track_id"], first[0]["track_id"]]
    # A different, distant person never gets a universal person singleton ID.
    third = tracker.update([localized("person", 8., timestamp=12.)], 12.)
    assert third[0]["track_id"] not in {item["track_id"] for item in first}


def test_track_expiry_label_change_and_one_to_one_matching():
    tracker = SpatialTracker(ttl=2.)
    first = tracker.update([localized("cup", 1.)], 10.)[0]
    second = tracker.update([localized("bottle", 1., timestamp=11.)], 11.)[0]
    assert first["track_id"] != second["track_id"]
    late = tracker.update([localized("cup", 1., timestamp=13.)], 13.)[0]
    assert late["track_id"] != first["track_id"]
    multiple = tracker.update([localized("cup", 1.1, timestamp=14.), localized("cup", 1.2, timestamp=14.)], 14.)
    assert len({item["track_id"] for item in multiple}) == 2


def test_adjacent_crossing_people_cannot_inherit_an_ambiguous_model_identity():
    tracker = SpatialTracker()
    first = tracker.update([localized("person", 1., (.2, .2, .5, .8)),
                            localized("person", 1.4, (.35, .2, .65, .8))], 10.)
    old_ids = {item["track_id"] for item in first}
    # Both old people plausibly fit each newly recognized crossing instance.
    crossing = tracker.update([localized("person", 1.18, (.27, .2, .57, .8), 11.),
                               localized("person", 1.22, (.28, .2, .58, .8), 11.)], 11.)
    assert all(item["track_id"] not in old_ids for item in crossing)
    assert len({item["track_id"] for item in crossing}) == 2


def test_nearby_person_without_image_overlap_cannot_take_an_existing_identity():
    tracker = SpatialTracker()
    first = tracker.update([localized("person", 1., (.1, .2, .3, .8))], 10.)[0]
    adjacent = tracker.update([localized("person", 1.1, (.6, .2, .8, .8), 11.)], 11.)[0]
    assert adjacent["track_id"] != first["track_id"]


def test_low_model_confidence_never_becomes_a_localized_detection():
    worker = ModelPerception(detector=Detector(observation(confidence=.3)), clock=Clock())
    try:
        assert worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        assert complete(worker)["detections"] == []
    finally:
        worker.close(wait=True)


def test_model_target_matching_preserves_recognized_label_and_query_generation():
    value = wire_value()
    value["detections"][0].update(label="ceramic mug", matches_target=True)
    worker = ModelPerception(detector=Detector(ModelObservation.parse(value)), clock=Clock())
    try:
        worker.set_target("Cup")
        assert worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        result = complete(worker)
        assert result["detections"][0]["label"] == "cup"
        assert result["detections"][0]["recognized_label"] == "ceramic mug"
        assert result["detections"][0]["matches_target"] is True
        assert result["detections"][0]["target_generation"] == 1
        assert result["captured_at"] == 10. and result["capture_pose"] == camera()["capture_pose"]
        worker.set_target("Wallet")
        assert worker.status()["target_generation"] == 2
        assert worker.poll()["detections"] == [] and worker.status()["captured_at"] is None
    finally:
        worker.close(wait=True)


def test_empty_model_observation_has_capture_proof_without_invented_labels():
    worker = ModelPerception(detector=Detector(ModelObservation((), None)), clock=Clock())
    try:
        worker.submit(np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.), camera(), 10.)
        result = complete(worker)
        assert result["detections"] == [] and result["room"] is None
        assert result["captured_at"] == 10. and result["capture_pose"] == camera()["capture_pose"]
    finally:
        worker.close(wait=True)


def textured_frame(offset=0, seed=11):
    rng = np.random.default_rng(seed)
    rgb = np.full((120, 160, 3), 40, dtype=np.uint8)
    texture = rng.integers(50, 240, (70, 44, 1), dtype=np.uint8)
    rgb[25:95, 40 + offset:84 + offset] = texture
    return rgb


def tracking_camera():
    value = camera()
    value.update(cx=79.5, cy=59.5)
    return value


def tracked_seed(timestamp=10.):
    detection = ImageDetection("person", (40 / 159, 25 / 119, 84 / 159, 95 / 119), .9)
    item = project_detection(detection, np.full((120, 160), 2.), tracking_camera(), timestamp, "vision:real-model")
    return {**item, "track_id": "visual:19", "recognition": "model", "semantic_timestamp": timestamp}


def test_actual_klt_translation_updates_measured_depth_timestamp_and_same_identity():
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10.)
    assert len(tracker.tracks["visual:19"]["points"]) >= 8
    measured_before = project_feature_depth(tracker.tracks["visual:19"]["points"].reshape(-1, 2),
                                            tracker.gray.shape, np.full((120, 160), 2.), tracking_camera(), 10.)
    result, lost = tracker.update(textured_frame(4), np.full((120, 160), 2.), tracking_camera(), 10.2)
    assert not lost and len(result) == 1
    item = result[0]
    assert item["track_id"] == "visual:19" and item["timestamp"] == 10.2
    assert item["semantic_timestamp"] == 10.
    assert item["source"] == "vision:real-model:tracked" and item["recognition"] == "optical-flow"
    assert item["bbox"][0] == pytest.approx(44 / 159, abs=.005)
    assert item["x"] - measured_before["x"] == pytest.approx(.08, abs=.01)


@pytest.mark.parametrize("failure", ["occluded", "different_person", "gap", "semantic_expired", "bad_depth", "scene_change"])
def test_klt_fails_closed_and_never_reacquires_lost_person(failure):
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10.)
    depth, timestamp = np.full((120, 160), 2.), 10.2
    frame = textured_frame(4)
    if failure == "occluded":
        frame[:] = 40
    elif failure == "different_person":
        frame = textured_frame(4, seed=33)
    elif failure == "gap":
        timestamp = 11.
    elif failure == "semantic_expired":
        tracker.timestamp = 20.
        timestamp = 20.2
    elif failure == "bad_depth":
        depth[:] = np.nan
    elif failure == "scene_change":
        frame = np.random.default_rng(42).integers(0, 256, frame.shape, dtype=np.uint8)
    result, lost = tracker.update(frame, depth, tracking_camera(), timestamp)
    assert result == [] and lost == {"visual:19"}
    result, _ = tracker.update(textured_frame(8), np.full((120, 160), 2.), tracking_camera(), timestamp + .2)
    assert result == [] and not tracker.tracks


def test_klt_does_not_seed_untextured_model_instance_or_ambiguous_people():
    tracker = OpticalFlowTracker()
    tracker.seed(np.full((120, 160, 3), 80, dtype=np.uint8), [tracked_seed()], 10.)
    assert not tracker.tracks
    twin = {**tracked_seed(), "track_id": "visual:20"}
    tracker.seed(textured_frame(), [tracked_seed(), twin], 10.)
    result, lost = tracker.update(textured_frame(4), np.full((120, 160), 2.), tracking_camera(), 10.2)
    assert result == [] and lost == {"visual:19", "visual:20"}


def test_worker_replays_exposures_captured_during_inference_then_tracks_live_and_forgets_loss():
    clock, gate = Clock(), threading.Event()
    value = wire_value()
    value["detections"][0].update(label="person", bbox=tracked_seed()["bbox"])
    detector = Detector(ModelObservation.parse(value), gate=gate)
    worker = ModelPerception(detector=detector, clock=clock)
    depth = np.full((120, 160), 2.)
    try:
        assert worker.submit(textured_frame(), depth, tracking_camera(), 10.)
        for offset in (2, 4, 6):
            clock.value += .2
            assert not worker.submit(textured_frame(offset), depth, tracking_camera(), clock.value)
        gate.set()
        result = complete(worker)
        assert result["recognition"] == "optical-flow"
        original_id = result["detections"][0]["track_id"]
        assert result["detections"][0]["timestamp"] == pytest.approx(10.6)
        assert result["detections"][0]["semantic_timestamp"] == 10.
        assert result["captured_at"] == 10.
        assert result["detections"][0]["bbox"][0] == pytest.approx(46 / 159, abs=.005)
        # New frames use actual flow even while another provider request is pending.
        gate.clear()
        clock.value += .2
        assert worker.submit(textured_frame(8), depth, tracking_camera(), clock.value)
        assert processed(worker)["detections"][0]["timestamp"] == pytest.approx(10.8)
        clock.value += .2
        assert not worker.submit(np.full((120, 160, 3), 40, dtype=np.uint8), depth, tracking_camera(), clock.value)
        assert processed(worker)["detections"] == []
        assert original_id in worker.invalid_ids
        gate.set()
        # A delayed old observation cannot revive the lost identity.
        result = complete(worker)
        assert all(item["track_id"] != original_id for item in result["detections"])
    finally:
        gate.set()
        worker.close(wait=True)


def test_renderer_calls_never_wait_for_model_result_projection_or_tracking(monkeypatch):
    clock, entered, release = Clock(), threading.Event(), threading.Event()
    value = wire_value()
    value["detections"][0].update(label="person", bbox=tracked_seed()["bbox"])
    worker = ModelPerception(detector=Detector(ModelObservation.parse(value)), clock=clock)
    original_seed = worker.flow.seed
    thread_ids = []

    def slow_seed(*args):
        thread_ids.append(threading.get_ident())
        entered.set()
        assert release.wait(2)
        return original_seed(*args)

    monkeypatch.setattr(worker.flow, "seed", slow_seed)
    depth = np.full((120, 160), 2.)
    try:
        assert worker.submit(textured_frame(), depth, tracking_camera(), 10.)
        assert entered.wait(2)
        started = time.perf_counter()
        clock.value = 10.2
        assert not worker.submit(textured_frame(4), depth, tracking_camera(), clock.value)
        assert worker.status()["pending"]
        assert worker.poll()["detections"] == []
        assert time.perf_counter() - started < .05
        assert thread_ids == [worker.coordinator.ident]
        assert thread_ids[0] != threading.get_ident()
        release.set()
        result = complete(worker)
        assert result["captured_at"] == 10.
        assert result["detections"][0]["timestamp"] == 10.2
        assert result["detections"][0]["semantic_timestamp"] == 10.
    finally:
        release.set()
        worker.close(wait=True)


@pytest.mark.parametrize("model_result", ["empty", "unrelated"])
def test_detector_false_negative_preserves_measured_flow_without_refreshing_semantics(model_result):
    clock, gate = Clock(), threading.Event()
    value = wire_value()
    value["detections"][0].update(label="person", bbox=tracked_seed()["bbox"])
    gate.set()
    detector = Detector(ModelObservation.parse(value), gate=gate)
    worker = ModelPerception(detector=detector, clock=clock)
    depth = np.full((120, 160), 2.)
    try:
        assert worker.submit(textured_frame(), depth, tracking_camera(), clock.value)
        original_id = complete(worker)["detections"][0]["track_id"]
        gate.clear()
        detector.result = (ModelObservation((), None) if model_result == "empty"
                           else observation("cup", bbox=[.05, .1, .2, .2]))
        clock.value = 10.2
        assert worker.submit(textured_frame(2), depth, tracking_camera(), clock.value)
        for offset, timestamp in ((4, 10.4), (6, 10.6)):
            clock.value = timestamp
            assert not worker.submit(textured_frame(offset), depth, tracking_camera(), clock.value)
        assert processed(worker)["recognition"] == "optical-flow"
        gate.set()
        result = complete(worker)
        person = next(item for item in result["detections"] if item["label"] == "person")
        assert person["track_id"] == original_id and person["timestamp"] == 10.6
        assert person["semantic_timestamp"] == 10.
        assert person["source"] == "vision:actual-model-name:tracked"
        assert result["captured_at"] == 10.2 and result["recognition"] == "optical-flow"
        # Occlusion still loses the ID; another empty result cannot revive it.
        gate.clear()
        clock.value = 10.8
        assert worker.submit(np.full((120, 160, 3), 40, dtype=np.uint8), depth,
                             tracking_camera(), clock.value)
        assert not any(item["label"] == "person" for item in processed(worker)["detections"])
        gate.set()
        assert not any(item["label"] == "person" for item in complete(worker)["detections"])
        assert original_id in worker.invalid_ids
    finally:
        gate.set()
        worker.close(wait=True)


@pytest.mark.parametrize("rejection", ["stale", "conflicting", "recognized_person"])
def test_false_negative_retention_cannot_hide_stale_or_conflicting_evidence(rejection):
    clock, gate = Clock(), threading.Event()
    value = wire_value()
    value["detections"][0].update(label="person", bbox=tracked_seed()["bbox"])
    gate.set()
    detector = Detector(ModelObservation.parse(value), gate=gate)
    worker = ModelPerception(detector=detector, clock=clock)
    depth = np.full((120, 160), 2.)
    try:
        assert worker.submit(textured_frame(), depth, tracking_camera(), 10.)
        original_id = complete(worker)["detections"][0]["track_id"]
        gate.clear()
        detector.result = (observation("bench", bbox=tracked_seed()["bbox"])
                           if rejection == "conflicting" else observation("person", bbox=[.7, .1, .95, .8])
                           if rejection == "recognized_person" else ModelObservation((), None))
        clock.value = 10.2
        assert worker.submit(textured_frame(2), depth, tracking_camera(), clock.value)
        assert processed(worker)["recognition"] == "optical-flow"
        if rejection == "stale":
            clock.value = 11.
        gate.set()
        result = complete(worker)
        assert all(item["track_id"] != original_id for item in result["detections"])
        assert original_id in worker.invalid_ids
    finally:
        gate.set()
        worker.close(wait=True)


def test_tracking_resolution_is_bounded_but_depth_geometry_uses_full_exposure():
    frame = np.repeat(np.repeat(textured_frame(), 4, axis=0), 4, axis=1)
    depth = np.full(frame.shape[:2], 2.)
    calibrated = tracking_camera()
    calibrated.update(fx=400., fy=400., cx=319.5, cy=239.5)
    item = tracked_seed()
    item.update(project_detection(ImageDetection("person", tuple(item["bbox"]), .9), depth,
                                  calibrated, 10., "vision:real-model"))
    tracker = OpticalFlowTracker()
    tracker.seed(frame, [item], 10., depth)
    assert tracker.gray.shape == (240, 320)
    measured_before = project_feature_depth(tracker.tracks["visual:19"]["points"].reshape(-1, 2),
                                            tracker.gray.shape, depth, calibrated, 10.)
    shifted = np.repeat(np.repeat(textured_frame(4), 4, axis=0), 4, axis=1)
    result, lost = tracker.update(shifted, depth, calibrated, 10.2)
    assert not lost and result[0]["timestamp"] == 10.2
    assert result[0]["x"] - measured_before["x"] == pytest.approx(.08, abs=.01)


def feature_depth_layer(tracker, failure=None):
    points = tracker.tracks["visual:19"]["points"].reshape(-1, 2) + [4., 0.]
    depth = np.full((120, 160), np.nan)
    values = np.full(len(points), 2.)
    if failure == "insufficient_points":
        values[7:] = np.nan
    elif failure == "insufficient_fraction":
        values[int(len(values) * .65):] = np.nan
    elif failure == "incoherent":
        values[::2] = 5.
    elif failure == "spread":
        values[:] = np.linspace(1.72, 2.28, len(values))
    for point, value in zip(points, values):
        u, v = np.rint(point).astype(int)
        depth[v, u] = value
    return depth


def test_tracking_uses_actual_feature_depth_when_central_person_box_mixes_background():
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10., np.full((120, 160), 2.))
    depth = np.full((120, 160), 5.)
    # The central patch has conflicting person/background surfaces. Current
    # feature pixels still have directly measured depth on the person layer.
    depth[49:72, 59:74] = np.where(np.indices((23, 15)).sum(axis=0) % 2, 2., 5.)
    feature_depth = feature_depth_layer(tracker)
    depth[np.isfinite(feature_depth)] = feature_depth[np.isfinite(feature_depth)]
    box = ImageDetection("person", (44 / 159, 25 / 119, 88 / 159, 95 / 119), .9)
    assert project_detection(box, depth, tracking_camera(), 10.2, "vision:real-model") is None
    result, lost = tracker.update(textured_frame(4), depth, tracking_camera(), 10.2)
    assert not lost and len(result) == 1
    item = result[0]
    assert item["track_id"] == "visual:19" and item["z"] == 2.
    assert item["depth_points"] >= 8 and item["depth_points"] >= .7 * item["tracking_points"]
    assert item["timestamp"] == 10.2 and item["semantic_timestamp"] == 10.


@pytest.mark.parametrize("failure", ["insufficient_points", "insufficient_fraction", "incoherent", "spread"])
def test_tracking_loses_person_without_enough_coherent_current_feature_depth(failure):
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10., np.full((120, 160), 2.))
    result, lost = tracker.update(textured_frame(4), feature_depth_layer(tracker, failure), tracking_camera(), 10.2)
    assert result == [] and lost == {"visual:19"}
    assert tracker.last_loss["gate"] == "depth" and tracker.last_loss["captured_at"] == 10.2


def edge_person_frame(offset=0):
    frame = np.full((120, 160, 3), 40, dtype=np.uint8)
    texture = np.random.default_rng(123).integers(50, 240, (70, 18, 1), dtype=np.uint8)
    frame[25:95, 108 + offset:126 + offset] = texture
    return frame


def edge_person_seed():
    item = project_detection(ImageDetection("person", (100 / 159, 25 / 119, 156 / 159, 95 / 119), .9),
                             np.full((120, 160), 2.), tracking_camera(), 10., "vision:real-model")
    return {**item, "track_id": "edge:person", "semantic_timestamp": 10.}


def test_verified_partial_person_box_is_clipped_when_at_least_eighty_percent_remains_visible():
    tracker = OpticalFlowTracker()
    tracker.seed(edge_person_frame(), [edge_person_seed()], 10.)
    result, lost = tracker.update(edge_person_frame(8), np.full((120, 160), 2.), tracking_camera(), 10.2)
    assert not lost and len(result) == 1
    item = result[0]
    assert item["track_id"] == "edge:person"
    assert item["bbox"][0] == pytest.approx(108 / 159, abs=.005)
    assert item["bbox"][2] == 1.
    assert item["timestamp"] == 10.2 and item["semantic_timestamp"] == 10.
    assert item["source"] == "vision:real-model:tracked" and item["z"] == 2.
    assert item["tracking_points"] >= 8 and tracker.last_loss is None


def test_coherent_features_cannot_keep_person_when_less_than_eighty_percent_is_visible():
    tracker = OpticalFlowTracker()
    tracker.seed(edge_person_frame(), [edge_person_seed()], 10.)
    # The textured feature cluster remains visible, but most of the recognized
    # box no longer does. Feature agreement alone cannot retain the identity.
    result, lost = tracker.update(edge_person_frame(20), np.full((120, 160), 2.), tracking_camera(), 10.2)
    assert result == [] and lost == {"edge:person"}
    assert tracker.last_loss == {"gate": "visibility", "captured_at": 10.2, "track_ids": ["edge:person"]}


def test_published_tracking_loss_is_bounded_and_uses_actual_capture_timestamp():
    clock = Clock()
    value = wire_value()
    value["detections"][0].update(label="person", bbox=tracked_seed()["bbox"])
    detector = Detector(ModelObservation.parse(value))
    worker = ModelPerception(detector=detector, clock=clock)
    depth = np.full((120, 160), 2.)
    try:
        assert worker.submit(textured_frame(), depth, tracking_camera(), 10.)
        original_id = complete(worker)["detections"][0]["track_id"]
        # Use an empty next recognition so it cannot replace the loss with a
        # newly recognized identity after the observed texture disappears.
        detector.result = ModelObservation((), None)
        clock.value = 10.2
        assert worker.submit(np.full((120, 160, 3), 40, dtype=np.uint8), depth, tracking_camera(), 10.2)
        result = complete(worker)
        loss = result["tracking"]["last_loss"]
        assert {key: loss[key] for key in ("gate", "captured_at", "track_ids")} == {
            "gate": "feature_match", "captured_at": 10.2, "track_ids": [original_id]}
        assert 0 <= loss["good_points"] < 8 and loss["total_points"] >= 8
        assert result["tracking"]["active"] == 0
        loss["gate"] = "changed externally"
        assert worker.status()["tracking"]["last_loss"]["gate"] == "feature_match"
    finally:
        worker.close(wait=True)


@pytest.mark.parametrize("scale", [1.3, math.inf])
def test_affine_loss_exposes_measured_motion_and_filters_nonfinite_values(monkeypatch, scale):
    import cv2
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10.)

    def excessive_affine(old, new, **kwargs):
        return np.array([[scale, 0., 0.], [0., scale, 0.]]), np.ones((len(old), 1), dtype=np.uint8)

    monkeypatch.setattr(cv2, "estimateAffinePartial2D", excessive_affine)
    result, lost = tracker.update(textured_frame(4), np.full((120, 160), 2.), tracking_camera(), 10.2)
    assert result == [] and lost == {"visual:19"}
    loss = tracker.last_loss
    assert loss["gate"] == "affine_motion" and loss["captured_at"] == 10.2
    assert loss["turn_rad"] == 0. and loss["frame_gap_s"] == pytest.approx(.2)
    if math.isfinite(scale):
        assert loss["scale"] == scale
    else:
        assert "scale" not in loss
    json.dumps(loss, allow_nan=False)


def test_depth_motion_loss_exposes_measured_distance_and_unchanged_limit():
    tracker = OpticalFlowTracker()
    tracker.seed(textured_frame(), [tracked_seed()], 10.)
    result, lost = tracker.update(textured_frame(4), np.full((120, 160), 3.), tracking_camera(), 10.2)
    assert result == [] and lost == {"visual:19"}
    loss = tracker.last_loss
    assert loss["gate"] == "depth_motion" and loss["captured_at"] == 10.2
    assert loss["delta_m"] > 1. and loss["allowed_m"] == pytest.approx(.7)
    json.dumps(loss, allow_nan=False)


class NeuralSession:
    def __init__(self):
        self.feeds = []
        self.logits = np.array([[[-5., -5., 8., -5.], [-5., 7., -5., -5.],
                                 [-5., -5., -5., 8.], [-5., -5., 7., -5.]]], dtype=np.float32)
        self.boxes = np.array([[[.5, .5, .2, .4], [.2, .5, .2, .8],
                                [.8, .5, .2, .3], [.501, .501, .2, .4]]], dtype=np.float32)

    def run(self, outputs, feeds):
        assert outputs == ["logits", "pred_boxes"]
        self.feeds.append(feeds)
        return self.logits, self.boxes


def neural_detector(session=None):
    return YolosDetector(session=session or NeuralSession(), config={"id2label": {"0": "N/A", "1": "person", "2": "cup"}},
                         preprocessor={"size": {"shortest_edge": 32, "longest_edge": 64},
                                       "rescale_factor": 1 / 255, "image_mean": [.4, .5, .6],
                                       "image_std": [.2, .2, .2]})


def test_yolos_runs_pixel_only_neural_input_and_decodes_trained_classes_boxes():
    detector = neural_detector()
    result = detector.detect(Image.new("RGB", (100, 50), (102, 128, 153)), "a cup")
    assert {item.label for item in result.detections} == {"cup", "person"}
    assert len(result.detections) == 2  # duplicate neural query suppressed
    assert result.room is None
    cup = next(item for item in result.detections if item.label == "cup")
    assert cup.bbox == pytest.approx((.4, .3, .6, .7))
    assert cup.matches_target and cup.confidence > .99
    assert set(detector.session.feeds[0]) == {"pixel_values"}
    pixels = detector.session.feeds[0]["pixel_values"]
    assert pixels.shape == (1, 3, 32, 64) and pixels.dtype == np.float32
    assert pixels[0, :, 0, 0] == pytest.approx([0., (128 / 255 - .5) / .2, 0.], abs=1e-6)


@pytest.mark.parametrize("target", ["keys", "my cup", "red cup", "person wearing red", "cup hiddenbehindwall"])
def test_closed_vocabulary_does_not_claim_unsupported_object_or_attribute(target):
    detector = neural_detector()
    with pytest.raises(ValueError, match="cannot verify"):
        detector.target_class(target)
    worker = ModelPerception(detector=detector, clock=Clock())
    try:
        with pytest.raises(ValueError, match="cannot verify"):
            worker.set_target(target)
        assert worker.future is None and worker.poll()["detections"] == []
    finally:
        worker.close(wait=True)


def test_fixed_vocabulary_is_model_metadata_and_default_provider_is_local(monkeypatch):
    detector = neural_detector()
    assert detector.supported_targets() == ["cup", "person"]
    assert detector.target_class("the cups") == "cup"
    monkeypatch.delenv("GO2_VISION_PROVIDER", raising=False)
    worker = ModelPerception(clock=Clock())
    try:
        assert isinstance(worker.detector, HybridDetector)
        assert worker.status()["provider"] == "local"
        assert worker.status()["vocabulary"] == "fixed learned classes"
    finally:
        worker.close(wait=True)


@pytest.mark.parametrize("provider, expected", [("yolos", YolosDetector), ("ollama", OllamaDetector)])
def test_explicit_vision_providers_remain_available(monkeypatch, provider, expected):
    monkeypatch.setenv("GO2_VISION_PROVIDER", provider)
    worker = ModelPerception(clock=Clock())
    try:
        assert isinstance(worker.detector, expected)
        assert worker.status()["provider"] == provider
    finally:
        worker.close(wait=True)


def local_detector():
    fast, accurate = neural_detector(), neural_detector()
    fast.model, accurate.model = "fast-trained-model", "accurate-trained-model"
    return HybridDetector(yolos=fast, detr=accurate)


@pytest.mark.parametrize("target", [None, "person", "a person", "the persons"])
def test_local_router_uses_fast_learned_model_for_passive_and_person_awareness(target):
    detector = local_detector()
    assert detector.model_for_target(target) == "fast-trained-model"
    result = detector.detect(Image.new("RGB", (40, 40)), target)
    assert {item.label for item in result.detections} == {"person", "cup"}
    assert len(detector.yolos.session.feeds) == 1 and detector.detr.session.feeds == []


def test_local_router_uses_accurate_model_for_supported_object_without_expected_locations():
    detector = local_detector()
    assert detector.target_class("the cups") == "cup"
    assert detector.model_for_target("a cup") == "accurate-trained-model"
    result = detector.detect(Image.new("RGB", (40, 40)), "a cup")
    assert next(item for item in result.detections if item.label == "cup").matches_target
    assert detector.yolos.session.feeds == [] and len(detector.detr.session.feeds) == 1
    for target in ("keys", "red cup", "my cup"):
        with pytest.raises(ValueError, match="cannot verify"):
            detector.model_for_target(target)
    assert len(detector.detr.session.feeds) == 1  # no false unsupported recognition


def test_local_job_provenance_follows_router_and_query_release():
    clock, detector = Clock(), local_detector()
    worker = ModelPerception(detector=detector, clock=clock)
    rgb, depth = np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.)
    try:
        assert worker.submit(rgb, depth, camera(), clock.value)
        result = complete(worker)
        assert result["provider"] == "local" and result["model"] == "fast-trained-model"
        assert result["last_model"] == "fast-trained-model"
        assert all(item["source"] == "vision:fast-trained-model" for item in result["detections"])
        worker.set_target("cup")
        assert worker.status()["captured_at"] is None
        clock.value += .2
        assert worker.submit(rgb, depth, camera(), clock.value)
        result = complete(worker)
        assert result["model"] == result["last_model"] == "accurate-trained-model"
        assert all(item["source"] == "vision:accurate-trained-model" for item in result["detections"])
        assert next(item for item in result["detections"] if item["label"] == "cup")["matches_target"]
        worker.set_target(None)
        clock.value += .2
        assert worker.submit(rgb, depth, camera(), clock.value)
        result = complete(worker)
        assert result["model"] == "fast-trained-model"
        assert all(item["source"] == "vision:fast-trained-model" for item in result["detections"])
        assert len(detector.yolos.session.feeds) == 2 and len(detector.detr.session.feeds) == 1
    finally:
        worker.close(wait=True)


def test_local_pending_job_keeps_its_original_model_after_target_changes():
    clock, detector, gate = Clock(), local_detector(), threading.Event()
    run = detector.yolos.session.run

    def blocked(*args):
        assert gate.wait(2)
        return run(*args)

    detector.yolos.session.run = blocked
    worker = ModelPerception(detector=detector, clock=clock)
    rgb, depth = np.zeros((100, 100, 3), dtype=np.uint8), np.full((100, 100), 2.)
    try:
        assert worker.submit(rgb, depth, camera(), clock.value)
        worker.set_target("cup")
        assert worker.status()["pending_model"] == "fast-trained-model"
        assert not worker.submit(rgb, depth, camera(), clock.value)
        gate.set()
        assert complete(worker)["detections"] == []
        assert detector.detr.session.feeds == []
        assert worker.submit(rgb, depth, camera(), clock.value)
        result = complete(worker)
        assert result["model"] == "accurate-trained-model"
        assert all(item["source"] == "vision:accurate-trained-model" for item in result["detections"])
    finally:
        gate.set()
        worker.close(wait=True)


def test_local_router_never_substitutes_other_model_when_selected_inference_fails():
    detector = local_detector()
    detector.detr.session.run = lambda *_: (_ for _ in ()).throw(RuntimeError("failed inference"))
    with pytest.raises(PerceptionUnavailable, match="inference failed"):
        detector.detect(Image.new("RGB", (40, 40)), "cup")
    assert detector.yolos.session.feeds == []


def test_missing_neural_model_assets_fail_without_recognition_fallback(tmp_path):
    detector = YolosDetector(assets=tmp_path)
    with pytest.raises(PerceptionUnavailable, match="assets are missing"):
        detector.detect(Image.new("RGB", (20, 20)))


@pytest.mark.parametrize("failure", ["nan", "shape", "inference"])
def test_neural_invalid_output_and_inference_failure_are_closed(failure):
    session = NeuralSession()
    if failure == "nan":
        session.logits[0, 0, 1] = math.nan
    elif failure == "shape":
        session.boxes = session.boxes[:, :, :3]
    else:
        def failed(*args):
            raise RuntimeError("ONNX failure")
        session.run = failed
    with pytest.raises(PerceptionUnavailable, match="inference failed"):
        neural_detector(session).detect(Image.new("RGB", (20, 20)))


def test_detr_uses_export_declared_validity_mask_and_pixel_only_input(monkeypatch):
    from types import SimpleNamespace
    session = NeuralSession()
    session.get_inputs = lambda: [SimpleNamespace(name="pixel_values", shape=["batch", 3, "height", "width"]),
                                 SimpleNamespace(name="pixel_mask", shape=["batch", 64, 64])]
    metadata = neural_detector(session)
    detector = DetrDetector(session=session, config=metadata.config, preprocessor=metadata.preprocessor)
    result = detector.detect(Image.new("RGB", (100, 50)), "cup")
    assert {item.label for item in result.detections} == {"cup", "person"}
    feeds = session.feeds[0]
    assert set(feeds) == {"pixel_values", "pixel_mask"}
    assert feeds["pixel_mask"].shape == (1, 64, 64)
    assert feeds["pixel_mask"].dtype == np.int64 and feeds["pixel_mask"].all()
    monkeypatch.setenv("GO2_VISION_PROVIDER", "detr")
    worker = ModelPerception(clock=Clock())
    try:
        assert isinstance(worker.detector, DetrDetector)
        assert worker.status()["provider"] == "detr"
    finally:
        worker.close(wait=True)
