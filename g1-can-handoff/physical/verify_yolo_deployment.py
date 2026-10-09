"""Read-only source, checkpoint, CUDA and fixture verification after deployment."""

import hashlib
import json
from pathlib import Path
import time
from operator_panel import Wendy
from trial_evidence import result_payload


def main():
    root = Path(__file__).parent / "yolo"
    names = [
        "service.py",
        "preprocessing.py",
        "postprocessing.py",
        "qualification.py",
        "frame_limits.py",
        "confidence.py",
    ]
    expected = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    limit = float(
        json.loads((root / "wendy.json").read_text())["env"][
            "HANDOFF_CAMERA_MAX_AGE_MS"
        ]
    )
    manifest = root / "fixtures/qualification.json"
    fixture_count = len(json.loads(manifest.read_text())["fixtures"])
    fixture_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    script = (
        """import hashlib,json,pathlib,urllib.request
expected=EXPECTED
actual={name:hashlib.sha256((pathlib.Path('/app')/name).read_bytes()).hexdigest() for name in expected}
assert actual==expected,'Deployed inference sources differ'
health=json.load(urllib.request.urlopen('http://127.0.0.1:8112/health',timeout=5))
assert health['healthy'] and health['backend']=='pytorch_cuda' and health['execution']=='cuda_graph_fp16',health
assert health['model_sha256']=='fab6252b0abc0613ef39c36ea2d75a061d80bb087f5eb9034ca33614c26301e9',health
assert health['camera_max_age_ms']==LIMIT and health['confidence_strictly_above']==.1,health
assert not health['camera_owned'] and health['motion_commands_sent']==0 and len(health['qualification'])==FIXTURE_COUNT,health
assert hashlib.sha256(pathlib.Path('/fixtures/qualification.json').read_bytes()).hexdigest()==FIXTURE_HASH,'Inference fixture manifest differs'
print(json.dumps({'inference_verified':True,'source_sha256':actual,'health':health,'physical_pickup_verified':False}))
""".replace("EXPECTED", repr(expected))
        .replace("LIMIT", repr(limit))
        .replace("FIXTURE_COUNT", repr(fixture_count))
        .replace("FIXTURE_HASH", repr(fixture_hash))
    )
    client = Wendy()
    try:
        for attempt in range(10):
            try:
                payload = result_payload(
                    client.tool(
                        "container_exec",
                        dict(
                            app_name="DEV-g1-handoff-coke-yolo",
                            command=["python3", "-c", script],
                            timeout_seconds=15,
                            max_bytes=12000,
                        ),
                    )
                )
                print(payload["stdout"], end="")
                return 0
            except RuntimeError:
                if attempt == 9:
                    raise
                time.sleep(1)
    finally:
        client.process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
