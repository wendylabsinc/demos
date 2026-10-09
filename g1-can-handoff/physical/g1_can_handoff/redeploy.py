#!/usr/bin/env python3
"""Redeploy this checkout's NX2 controller and verify PASSIVE startup.

Run from any directory. --check performs only the read-only ownership check.
This does not release joints or start a demo. The controls page is local to the
Mac; UI-only edits require restarting operator_panel.py, not this deployment.
"""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
WRAPPER = WORKSPACE / "scripts/wendy-run-timed"
AUDIT = WORKSPACE / "scripts/check-wendy-deployment-wrappers"
MARKER = Path("/tmp/wendy-g1-nx2-deployment.lock")
DEVICE = "unitree-g1-nx-2.local:50052"
VENV = WORKSPACE / "demos-g1-nx2-yolo/.venv/bin/python"

sys.path.insert(0, str(ROOT))
from deployment_version import describe_version, source_version, write_stamp

PREFLIGHT = """
import json,sys
from operator_panel import Wendy,DEVICE
from deployment_version import describe_version
assert DEVICE==sys.argv[1], 'Operator and deployment targets differ'
client=Wendy()
try:
    state=client.request(action='status')
    print('Currently running: '+describe_version(state.get('build_info') or {}),flush=True)
    if state.get('adapter_initialized') is not False or state.get('demo_running') is not False:
        raise RuntimeError('Controller owns the arms or a demo is running. Support and release before redeploying.')
    if state.get('phase') not in ('PASSIVE','BLOCKED'):
        raise RuntimeError('Idle controller state could not be verified: '+json.dumps(state))
    if state.get('telemetry_healthy') is not True:
        print('Motion remains blocked: '+str(state.get('telemetry_error') or 'robot feedback unverified')+'. Released PASSIVE software deployment is allowed.',flush=True)
    print('Read-only preflight passed: controller idle, arm ownership released.')
finally:
    client.process.terminate()
"""

READBACK = """
import json,pathlib
from operator_panel import Wendy
from deployment_version import describe_version
expected=json.loads(pathlib.Path('deployment-version.json').read_text())
client=Wendy()
try:
    state=client.request(action='status')
    actual=state.get('build_info')
    if actual!=expected:
        raise RuntimeError('Running version does not match the requested build: '+json.dumps(actual))
    print('VERIFIED RUNNING: '+describe_version(actual),flush=True)
    print('Built at: '+actual['built_at_utc'],flush=True)
    print('Controller phase: '+state['phase']+' | arm ownership: '+str(state['adapter_initialized']),flush=True)
    if not state.get('telemetry_healthy'):
        print('Motion readiness BLOCKED: '+str(state.get('telemetry_error')),flush=True)
finally:
    client.process.terminate()
"""


@contextmanager
def build_environment():
    """Keep unavailable Docker credential helpers from breaking public builds.

    Preserve the user's Docker configuration. A filtered temporary copy is
    used only by this deployment, with the same resolved Docker endpoint.
    """
    env = os.environ.copy()
    directory = Path(env.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
    path = directory / "config.json"
    config = json.loads(path.read_text()) if path.exists() else {}
    missing = (
        lambda name: bool(name) and shutil.which("docker-credential-" + name) is None
    )
    unavailable_store = missing(config.get("credsStore"))
    unavailable_helpers = [
        registry
        for registry, helper in config.get("credHelpers", {}).items()
        if missing(helper)
    ]
    if not unavailable_store and not unavailable_helpers:
        yield env
        return
    endpoint = env.get("DOCKER_HOST")
    if not endpoint or env.get("DOCKER_CONTEXT"):
        endpoint = subprocess.check_output(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            env=env,
            text=True,
        ).strip()
    if unavailable_store:
        config.pop("credsStore", None)
    for registry in unavailable_helpers:
        config["credHelpers"].pop(registry)
    config.pop("currentContext", None)
    with tempfile.TemporaryDirectory(prefix="g1-redeploy-docker-") as temporary:
        filtered = Path(temporary) / "config.json"
        filtered.write_text(json.dumps(config))
        filtered.chmod(0o600)
        env["DOCKER_CONFIG"] = temporary
        env["DOCKER_HOST"] = endpoint
        env.pop("DOCKER_CONTEXT", None)
        print("Docker endpoint: " + endpoint, flush=True)
        print(
            "Unavailable credential helpers omitted for this build only; your Docker settings are unchanged.",
            flush=True,
        )
        yield env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Read-only check; do not deploy"
    )
    parser.add_argument(
        "--python",
        default=str(VENV if VENV.exists() else Path(sys.executable)),
        help="Python with the physical project dependencies installed",
    )
    args = parser.parse_args(argv)
    print("Project: " + str(ROOT), flush=True)
    print("Device: " + DEVICE, flush=True)
    print("Requested source: " + describe_version(source_version(ROOT)), flush=True)
    if args.check:
        print("Read-only version/ownership check; no deployment.", flush=True)
        subprocess.run([args.python, "-c", PREFLIGHT, DEVICE], cwd=ROOT, check=True)
        return 0
    if not WRAPPER.is_file() or not AUDIT.is_file():
        parser.error(
            "Wendy workspace timing wrapper or deployment audit is missing: "
            + str(WORKSPACE / "scripts")
        )
    try:
        print("[1/4] Auditing the native Wendy deployment path…", flush=True)
        with MARKER.open("x") as marker:
            marker.write(f"redeploy pid={os.getpid()}\n")
    except FileExistsError:
        parser.error("Another deployment is in progress: " + str(MARKER))
    try:
        subprocess.run(
            [sys.executable, str(AUDIT), "--root", str(WORKSPACE)], check=True
        )
        print(
            "[2/4] Reading running version and verifying released ownership…",
            flush=True,
        )
        subprocess.run([args.python, "-c", PREFLIGHT, DEVICE], cwd=ROOT, check=True)
        info = write_stamp(source_version(ROOT), ROOT)
        print("[3/4] Building and deploying: " + describe_version(info), flush=True)
        print("Build stamp: " + str(ROOT / "deployment-version.json"), flush=True)
        with build_environment() as env:
            env["WENDY_TIMING_FUNCTIONAL_READINESS_COMMAND"] = shlex.join(
                [args.python, str(ROOT / "verify_passive_deployment.py")]
            )
            subprocess.run(
                [
                    sys.executable,
                    str(WRAPPER),
                    "--device",
                    DEVICE,
                    "--dockerfile",
                    "build.stagefile.yaml",
                    "--yes",
                    "--detach",
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        print("[4/4] Reading back the version from the running controller…", flush=True)
        subprocess.run([args.python, "-c", READBACK], cwd=ROOT, check=True)
        print(
            "NX2 controller redeployed and PASSIVE startup verified. No demo started.",
            flush=True,
        )
    finally:
        MARKER.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(
            "Redeploy failed; the requested version has not been verified. See the error above before retrying.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(exc.returncode)
