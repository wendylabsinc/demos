"""Export native trial artifacts through Wendy, verify locally, then acknowledge."""

import base64
import hashlib
import json
from pathlib import Path
import tarfile


def result_payload(result):
    row = result.get("structuredContent")
    if row is None:
        row = json.loads(
            next(c["text"] for c in result["content"] if c["type"] == "text")
        )
    if row.get("exit_code") != 0 or row.get("truncated"):
        raise RuntimeError(row.get("stderr") or "Incomplete native artifact transfer")
    return row


def recorder_status(client):
    script = "import json,urllib.request\ns=json.load(urllib.request.urlopen('http://127.0.0.1:8101/status',timeout=5))\nprint(json.dumps(s))"
    return json.loads(
        result_payload(
            client.tool(
                "container_exec",
                dict(
                    app_name="DEV-g1-handoff-physical",
                    command=["python3", "-c", script],
                    timeout_seconds=15,
                    max_bytes=100000,
                ),
            )
        )["stdout"]
    )


def export_trial(client, snapshot, start_ns, end_ns, destination):
    native = Path(snapshot["native_event_file"])
    if native.parent != Path("/data") or native.suffix != ".jsonl":
        raise RuntimeError("Unexpected native event path")
    source = (Path(__file__).parent / "native_trial_export.py").read_text()
    script = (
        source
        + "\nimport sys\nprint(json.dumps(create_export(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])))\n"
    )
    remote = "/data/trial-exports/" + str(start_ns) + "-" + Path(destination).name
    metadata = json.loads(
        result_payload(
            client.tool(
                "container_exec",
                dict(
                    app_name="DEV-g1-handoff-physical",
                    command=[
                        "python3",
                        "-c",
                        script,
                        str(native),
                        str(start_ns),
                        str(end_ns),
                        remote,
                    ],
                    timeout_seconds=55,
                    max_bytes=10000,
                ),
            )
        )["stdout"]
    )
    return fetch_artifact(
        client,
        "DEV-g1-handoff-physical",
        remote,
        metadata,
        destination,
        max_bytes=256000000,
    )


def fetch_artifact(client, app_name, remote, metadata, destination, max_bytes=32000000):
    if not 0 < metadata["bytes"] <= max_bytes:
        raise RuntimeError("Invalid artifact transfer size")
    destination = Path(destination)
    pending = destination.with_suffix(destination.suffix + ".pending")
    pending.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    try:
        with pending.open("wb") as out:
            chunk_size = 480000 if metadata["bytes"] > 32000000 else 120000
            for offset in range(0, metadata["bytes"], chunk_size):
                count = min(chunk_size, metadata["bytes"] - offset)
                script = "import base64,sys\nwith open(sys.argv[1],'rb') as f:\n f.seek(int(sys.argv[2]));print(base64.b64encode(f.read(int(sys.argv[3]))).decode())"
                result = result_payload(
                    client.tool(
                        "container_exec",
                        dict(
                            app_name=app_name,
                            command=[
                                "python3",
                                "-c",
                                script,
                                remote,
                                str(offset),
                                str(count),
                            ],
                            timeout_seconds=15,
                            max_bytes=max(170000, count * 2),
                        ),
                    )
                )
                raw = base64.b64decode(result["stdout"].strip(), validate=True)
                if len(raw) != count:
                    raise RuntimeError("Short trial artifact chunk")
                out.write(raw)
                digest.update(raw)
                written += len(raw)
        if written != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
            raise RuntimeError("Trial artifact checksum mismatch")
        with tarfile.open(pending) as tar:
            manifest = json.load(tar.extractfile("manifest.json"))
            for name, row in manifest["files"].items():
                raw = tar.extractfile(name).read()
                if (
                    len(raw) != row["bytes"]
                    or hashlib.sha256(raw).hexdigest() != row["sha256"]
                ):
                    raise RuntimeError(
                        "Trial artifact member checksum mismatch: " + name
                    )
        pending.replace(destination)
    except Exception:
        if pending.exists():
            pending.unlink()
        raise
    return dict(**metadata, local_path=str(destination), verified=True)
