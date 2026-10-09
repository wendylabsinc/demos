"""One numbered regular demo per invocation; reset acknowledgement between runs.

No deployment, release, controller bypass, or automatic retry. Every start is
reserved on disk first so an uncertain response cannot cause a duplicate start.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

from operator_panel import Wendy, DEPLOYING
from trial_evidence import export_trial, result_payload

APP = "DEV-g1-handoff-physical"


def snapshot(client, expected=None):
    script = """import hashlib,io,json,pathlib,time,urllib.request,zipfile
def get(port,path):return json.load(urllib.request.urlopen('http://127.0.0.1:'+str(port)+path,timeout=5))
result=dict(unix_ns=time.time_ns(),status=get(8102,'/status'),recorder=get(8101,'/status'))
if result['status']['phase']=='FROZEN':
 with pathlib.Path(result['status']['events']).open('rb') as f:
  f.seek(max(0,f.seek(0,2)-262144));tail=f.read().splitlines()[1:]
 rows=[json.loads(line) for line in tail]
 result['failure_hold']=next((row for row in reversed(rows) if row['phase']=='failure_hold'),None)
expected=EXPECTED
if expected is not None:
 actual={n:hashlib.sha256((pathlib.Path('/app')/n).read_bytes()).hexdigest() for n in expected}
 assert actual==expected,'Controller source differs from verified release'
 result['source_sha256']=actual
 result['camera']=get(8001,'/health')
 result['yolo']=get(8112,'/health') if result['status'].get('target_identity')!='color_depth' else {'required':False}
 raw=urllib.request.urlopen('http://127.0.0.1:8001/frame.tracking-rgbd',timeout=5).read()
 with zipfile.ZipFile(io.BytesIO(raw)) as z:meta=json.loads(z.read('metadata.json'))['source_metadata']
 result['camera_exposure_age_ms']=(time.monotonic()-meta['depth_time'])*1000
 assert meta['timestamp_valid'] and abs(meta['color_time']-meta['depth_time'])<=.012,meta
 result['camera_source_metadata']=meta
 result['calibration']=json.loads(pathlib.Path('/data/calibration.json').read_text())
print(json.dumps(result))
""".replace("EXPECTED", repr(expected))
    result = client.tool(
        "container_exec",
        dict(
            app_name=APP,
            command=["python3", "-c", script],
            timeout_seconds=15,
            max_bytes=100000,
        ),
    )
    return json.loads(result_payload(result)["stdout"])


def preflight(before, camera_max_age_ms=500, *, resume_pregrasp=False):
    state, recorder = before["status"], before["recorder"]
    resumable = (
        resume_pregrasp
        and state["phase"] == "FROZEN"
        and state["adapter_initialized"]
        and (before.get("failure_hold") or {}).get("grasp_started") is False
    )
    if not resumable and (
        state["phase"] not in ("PASSIVE", "BLOCKED")
        or state["adapter_initialized"]
        or state["demo_running"]
    ):
        raise RuntimeError(
            "Reset/release required: controller must be idle without ownership"
        )
    if state["demo_running"]:
        raise RuntimeError("A demo is already running")
    if not state["telemetry_healthy"] or not state["calibration_present"]:
        raise RuntimeError("Telemetry or accepted calibration unavailable")
    from yolo.frame_limits import frame_age_limit_s

    limit_ms = frame_age_limit_s(camera_max_age_ms) * 1000
    if (
        state["camera_max_age_ms"] != limit_ms
        or state["yolo_confidence_strictly_above"] != 0.1
    ):
        raise RuntimeError("Unexpected configured frame/confidence limits")
    if not (
        recorder["recording"]
        and not recorder["error"]
        and not sum(recorder["dropped"].values())
        and not recorder["missing_or_stale_required"]
    ):
        raise RuntimeError("Recorder is not healthy")
    camera, yolo = before["camera"], before["yolo"]
    if (
        not camera["healthy"]
        or camera["serial"] != "338622073335"
        or not 0 <= before["camera_exposure_age_ms"] < limit_ms
    ):
        raise RuntimeError("Camera exposure is stale or camera identity is wrong")
    if state.get("target_identity") == "color_depth":
        return
    if (
        not yolo["healthy"]
        or yolo["model_sha256"]
        != "fab6252b0abc0613ef39c36ea2d75a061d80bb087f5eb9034ca33614c26301e9"
    ):
        raise RuntimeError("YOLO is unavailable or model identity differs")
    if yolo.get("camera_max_age_ms") != limit_ms:
        raise RuntimeError("YOLO and controller exposure budgets differ")


def reserve(directory, index, reset_confirmed):
    if not 1 <= index <= 10:
        raise ValueError("Trial index must be 1..10")
    if index > 1:
        previous = json.loads((directory / f"trial-{index-1:02d}.json").read_text())
        if not reset_confirmed or previous.get("demo_running_after", True):
            raise RuntimeError(
                "Previous terminal trial and explicit operator reset confirmation required"
            )
    path = directory / f"trial-{index:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        json.dump(
            dict(index=index, state="reserved", reset_confirmed=reset_confirmed), output
        )
    return path


def save(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2) + "\n")
    pending.replace(path)


def beep():
    results = []
    for _ in range(3):
        result = subprocess.run(
            ["/usr/bin/afplay", "-v", "3", "/System/Library/Sounds/Glass.aiff"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        results.append(dict(exit_code=result.returncode, error=result.stderr.strip()))
    return results


def run(
    directory,
    index,
    reset_confirmed,
    timeout_s=180,
    verified_release=None,
    *,
    resume_pregrasp=False,
):
    if DEPLOYING.is_file():
        raise RuntimeError("Deployment in progress")
    if (directory / f"trial-{index:02d}.json").exists():
        raise RuntimeError(
            "Trial already reserved; inspect it instead of issuing another start"
        )
    verified_release = verified_release or (
        Path(__file__).parent
        / "offline/validation/start-button-20261008/controller-readiness.json"
    )
    release = json.loads(Path(verified_release).read_text())
    expected = release["source_sha256"]
    client = None
    path = None
    data = {}
    before = None
    try:
        client = Wendy()
        before = snapshot(client, expected)
        preflight(
            before,
            release["controller"]["camera_max_age_ms"],
            resume_pregrasp=resume_pregrasp,
        )
        path = reserve(directory, index, reset_confirmed)
        if index > 1:
            previous_path = directory / f"trial-{index-1:02d}.json"
            previous = json.loads(previous_path.read_text())
            previous["reset_confirmed_after"] = True
            previous["post_reset_status"] = before["status"]
            save(previous_path, previous)
        data = dict(
            index=index,
            requested_trials=10,
            reset_confirmed=reset_confirmed,
            resume_pregrasp=resume_pregrasp,
            started_unix_ns=before["unix_ns"],
            controller_before=before["status"],
            recorder_before=before["recorder"],
            preflight=before,
            state="start_requested",
        )
        save(path, data)
        # The regular button's exact backend action, with all controller checks.
        data["acknowledgement"] = client.request(action="start")
        save(path, data)
        print(json.dumps(dict(index=index, start_accepted=True)), flush=True)
        deadline = time.monotonic() + timeout_s
        phase = None
        stopped = False
        while True:
            after = snapshot(client)
            state = after["status"]
            if state["phase"] != phase:
                phase = state["phase"]
                print(json.dumps(dict(index=index, phase=phase)), flush=True)
            if not state["demo_running"]:
                break
            if time.monotonic() >= deadline:
                if stopped:
                    raise RuntimeError(
                        "Trial did not terminate after bounded Stop request"
                    )
                data["supervisor_stop"] = client.request(action="stop")
                stopped = True
                deadline = time.monotonic() + 15
                save(path, data)
            time.sleep(2)
        data.update(
            finished_unix_ns=after["unix_ns"],
            controller_after=state,
            recorder_after=after["recorder"],
            native_event_file=state["events"],
            demo_running_after=False,
            state="terminal",
            physical_lift_verified=bool(
                (state.get("demo_result") or {}).get("physical_lift_verified")
            ),
            primary_failure=state.get("first_failure"),
            error=state.get("demo_error") or state.get("error"),
        )
        save(path, data)
        data["attention_beep"] = beep()
        save(path, data)
        print(
            json.dumps(
                dict(
                    index=index,
                    terminal=True,
                    error=data["error"],
                    phase=state["phase"],
                    reset_required=True,
                )
            ),
            flush=True,
        )
        try:
            data["artifact"] = export_trial(
                client,
                data,
                data["started_unix_ns"],
                data["finished_unix_ns"],
                path.with_suffix(".tar.gz"),
            )
        except Exception as exc:
            data["artifact_error"] = str(exc)
        save(path, data)
        return data
    except Exception as exc:
        if path is not None:
            data["supervisor_error"] = str(exc)
            save(path, data)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "preflight-errors.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        dict(
                            index=index,
                            unix_ns=time.time_ns(),
                            error=str(exc),
                            before=before,
                            start_issued=False,
                        )
                    )
                    + "\n"
                )
        raise
    finally:
        if client is not None:
            client.process.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--index", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--reset-confirmed", action="store_true")
    parser.add_argument(
        "--resume-pregrasp",
        action="store_true",
        help="Explicit retry from recorded pregrasp FROZEN hold; all controller checks remain",
    )
    parser.add_argument(
        "--verified-release",
        type=Path,
        help="Read-only startup verification JSON for the release under test",
    )
    args = parser.parse_args()
    result = run(
        args.directory,
        args.index,
        args.reset_confirmed,
        verified_release=args.verified_release,
        resume_pregrasp=args.resume_pregrasp,
    )
    print(
        json.dumps(
            dict(
                index=args.index,
                recorded=str(args.directory),
                artifact_verified=result.get("artifact", {}).get("verified", False),
            )
        ),
        flush=True,
    )
