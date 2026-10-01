"""HTTP boundaries over real offline physics and a rendered onboard frame.

Voice failure stubs below test HTTP error handling only. Trained transcription
is exercised separately by the opt-in audio fixture in test_voice.py.
"""

import copy
import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from yard_drone.sensors import SensorRig
from yard_drone.server import DemoServer
from yard_drone.simulation import Simulation


@pytest.fixture
def http_sim(tmp_path):
    sim = Simulation(data_dir=tmp_path)
    sim.step()
    server = DemoServer(("127.0.0.1", 0), sim)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, body=None, *, raw=None, headers=None):
        if body is not None:
            raw = json.dumps(body).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(base + path, raw, headers=headers or {})
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            payload = response.read()
            if "application/json" in response.headers.get("Content-Type", ""):
                payload = json.loads(payload)
            return response.status, payload, response.headers

    yield sim, request
    server.stopping.set()
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
    sim.close()


def test_physical_actions_worker_goals_and_readiness(http_sim):
    sim, request = http_sim
    status, health, _ = request("/api/health")
    assert status == 503  # Models and sensor loop have not started offline.
    assert health["engine"] == "MuJoCo"
    assert health["sensor"]["ready"] is False
    assert request("/api/state")[1]["ready"] is True

    before = sim.snapshot()["position"]
    assert request("/api/action", {"action": "goto", "zone": "A"})[1]["ok"]
    assert sim.mode == "navigate"
    assert sim.snapshot()["position"] == before  # A command is not a teleport.
    assert request("/api/action", {"action": "hover"})[1]["ok"]

    worker_before = sim.environment.worker.position.copy()
    status, moved, _ = request("/api/worker", {"dx": .4, "dy": .2})
    assert status == 200 and moved["ok"]
    assert np.allclose(moved["workerPosition"], worker_before)
    assert np.allclose(moved["workerGoal"], worker_before[:2] + [.4, .2])
    sim.run_headless(.5)
    assert np.linalg.norm(sim.environment.worker.position[:2] - worker_before[:2]) > .01
    assert request("/api/worker", {"stop": True})[1]["ok"]
    assert np.allclose(sim.environment.worker.goal, sim.environment.worker.position[:2])

    for body in ({"action": "goto", "zone": "unknown"}, {"action": "fly_to_moon"},
                 {"action": "baseline"}, {"action": "follow"}):
        status, result, _ = request("/api/action", body)
        assert status == 400 and result["ok"] is False


def test_arbitrary_incident_and_invalid_json_boundaries(http_sim):
    sim, request = http_sim
    status, result, _ = request("/api/incident", {"type": "oil", "position": [1, 1]})
    assert status == 200 and result["ok"]
    fields = sim.snapshot()["environment"]["hazardFields"]
    assert fields["lastRelease"]["position"] == [1.0, 1.0, 0.0]
    assert fields["oilVolumeM3"] == pytest.approx(.018)
    assert request("/api/action", {"action": "clear"})[1]["ok"]
    assert sim.snapshot()["environment"]["hazardFields"]["cells"] == []

    invalid = [
        ("/api/incident", {"type": "oil", "position": [20, 1]}),
        ("/api/incident", {"type": "steam", "position": [1, 1]}),
        ("/api/worker", {"dx": 10, "dy": 0}),
        ("/api/worker", {"dx": "invalid"}),
        ("/api/command", {"text": 123}),
        ("/api/command", {"text": "x" * 501}),
        ("/api/action", []),
    ]
    for path, body in invalid:
        status, result, _ = request(path, body)
        assert status == 400 and result["ok"] is False
    assert request("/api/action", raw=b"{", headers={"Content-Type": "application/json"})[0] == 400
    assert request("/api/action", raw=b"{}", headers={"Content-Length": "4097"})[0] == 400
    assert request("/api/not-an-endpoint", {})[0] == 404


def test_real_camera_frame_and_persisted_evidence(http_sim):
    sim, request = http_sim
    status, unavailable, _ = request("/api/camera.jpg")
    assert status == 503 and unavailable["error"]

    assert request("/api/action", {"action": "takeoff"})[1]["ok"]
    sim.run_headless(8)
    assert sim.snapshot()["position"][2] > 4
    # Render from the physical airframe's current pose; no synthetic imagery.
    with SensorRig(sim.model) as rig:
        frame = rig.capture(copy.copy(sim.data))
    assert frame["image"].shape == (240, 320, 3)
    assert np.ptp(frame["image"]) > 20
    encoded = io.BytesIO()
    Image.fromarray(frame["image"]).save(encoded, format="JPEG")
    sim._camera_jpeg = encoded.getvalue()
    status, camera, headers = request("/api/camera.jpg?frame=1")
    assert status == 200 and headers["Content-Type"] == "image/jpeg"
    assert Image.open(io.BytesIO(camera)).size == (320, 240)

    sim.missions.begin("baseline")
    prepared = sim.missions.prepare("V00", frame, sim.snapshot()["position"], "A", sim.missions.mission["id"])
    assert sim.missions.record(prepared, coverage=.1)
    name = prepared["observation"]["image"]
    status, evidence, headers = request("/api/evidence/" + name)
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert np.array_equal(np.array(Image.open(io.BytesIO(evidence))), frame["image"])
    assert request("/api/evidence/000000000000-V99.png")[0] == 404
    assert request("/api/evidence/../baseline-camera.json")[0] == 404
    assert request("/static/../../baseline-camera.json")[0] == 404


def test_voice_transport_exposes_empty_speech_and_model_failures(http_sim):
    sim, request = http_sim

    class EmptyVoice:
        def transcribe(self, audio, mime):
            assert audio == b"recorded-audio" and mime == "audio/webm"
            return ""

    class UnavailableVoice:
        def transcribe(self, audio, mime):
            raise RuntimeError("Local speech weights are unavailable")

    sim.voice = EmptyVoice()
    status, result, _ = request("/api/voice", raw=b"recorded-audio", headers={"Content-Type": "audio/webm"})
    assert status == 400 and result["ok"] is False and "No speech" in result["message"]
    assert request("/api/voice", raw=b"")[0] == 400
    assert request("/api/voice", raw=b"x", headers={"Content-Length": str(8 * 1024 * 1024 + 1)})[0] == 400
    sim.voice = UnavailableVoice()
    status, result, _ = request("/api/voice", raw=b"recorded-audio")
    assert status == 503 and result["ok"] is False and "unavailable" in result["message"]
