#!/usr/bin/env bash
set -euo pipefail

# The simulator and its controller run inside WendyOS. The browser only renders
# the state returned by the app's HTTP server.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vm_name="${WENDY_VM_NAME:-house-vacuum}"
vm_port="${WENDY_VM_PORT:-50061}"
cli_name="${WENDY_CLI:-wendy}"
builder="${WENDY_BUILDER:-}"
build_host="${WENDY_BUILD_HOST:-}"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*" >&2; }

cli="$(command -v "$cli_name")" || fail "Wendy CLI not found: $cli_name. Install Wendy or set WENDY_CLI to its executable."
command -v qemu-system-aarch64 >/dev/null || fail "QEMU is missing. Install qemu, then rerun this script."
command -v python3 >/dev/null || fail "Python 3 is required to read Wendy's JSON results."
[[ "$vm_name" =~ ^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$ ]] || fail "WENDY_VM_NAME must contain 1 to 32 lowercase letters, digits or dashes."
[[ "$vm_port" =~ ^[0-9]{1,5}$ ]] || fail "WENDY_VM_PORT must be 1 to 65534, with the next port free too."
vm_port="$((10#$vm_port))"
(( vm_port >= 1 && vm_port <= 65534 )) || fail "WENDY_VM_PORT must be 1 to 65534, with the next port free too."
[[ -z "$builder" || -z "$build_host" ]] || fail "Set either WENDY_BUILDER or WENDY_BUILD_HOST, not both."
"$cli" vm --help >/dev/null || fail "This Wendy CLI has no VM support. Update Wendy or set WENDY_CLI."

builder_ready() {
  python3 -c '
import subprocess, sys
try:
    result = subprocess.run(sys.argv[1:], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=5)
    raise SystemExit(result.returncode)
except (OSError, subprocess.TimeoutExpired):
    raise SystemExit(1)
' "$@"
}

# Check a local build engine before downloading or starting the VM. An explicit
# remote build host does not use a local container engine.
if [[ -z "$build_host" ]]; then
  case "$builder" in
    docker)
      command -v docker >/dev/null && builder_ready docker info || fail "Docker is not responding. Start Docker or use WENDY_BUILDER=apple-container on Apple Silicon."
      ;;
    apple-container)
      command -v container >/dev/null || fail "Apple Container is missing. Install it or use a running Docker engine."
      ;;
    buildkit)
      command -v buildctl >/dev/null || fail "BuildKit is missing. Install buildctl and start buildkitd, or choose another builder."
      ;;
    "")
      if command -v docker >/dev/null && builder_ready docker info; then
        builder=docker
      elif [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] && command -v container >/dev/null && builder_ready container system status; then
        # Specify the working engine so Wendy cannot fall back to inactive Docker.
        builder=apple-container
      else
        fail "No running local image builder. Start Docker, use WENDY_BUILDER=apple-container on Apple Silicon, or set WENDY_BUILD_HOST to a configured Linux Wendy build host."
      fi
      ;;
    *) fail "Unknown WENDY_BUILDER: $builder. Choose docker, apple-container or buildkit." ;;
  esac
fi

vm_json="$("$cli" --json vm list)" || fail "Could not list Wendy VMs."
vm_state="$(python3 -c '
import json, sys
rows = json.load(sys.stdin)
if not isinstance(rows, list):
    raise SystemExit("Wendy vm list did not return a JSON array")
matches = [row for row in rows if row.get("name") == sys.argv[1]]
if len(matches) > 1:
    raise SystemExit("Wendy returned duplicate VM names")
print(matches[0].get("state", "unknown") if matches else "missing")
' "$vm_name" <<< "$vm_json")" || fail "Could not read the VM list."

if [[ "$vm_state" == missing ]]; then
  say "Creating WendyOS VM '$vm_name'. The first run downloads the WendyOS disk image."
  create_args=(vm create "$vm_name" --yes --disk 16)
  if [[ -n "${WENDY_VM_IMAGE:-}" ]]; then
    [[ -z "${WENDY_VM_VERSION:-}" ]] || fail "Set either WENDY_VM_IMAGE or WENDY_VM_VERSION, not both."
    create_args+=(--image "$WENDY_VM_IMAGE")
  elif [[ -n "${WENDY_VM_VERSION:-}" ]]; then
    create_args+=(--version "$WENDY_VM_VERSION")
  fi
  # Omitting --profile keeps this compatible with older Wendy releases.
  # --yes selects the generic profile in newer releases.
  "$cli" "${create_args[@]}" >&2 || fail "Could not create '$vm_name'. Check download access and free disk space."
  vm_state=stopped
fi

case "$vm_state" in
  stopped)
    say "Starting '$vm_name' with 4 CPUs, 4096 MiB RAM and agent ports $vm_port/$((vm_port + 1))."
    "$cli" vm start "$vm_name" --detach --memory 4096 --cpus 4 --port "$vm_port" --yes >&2 || fail "Could not start '$vm_name'. Check QEMU, its ARM64 UEFI firmware and the selected ports."
    ;;
  running) say "Reusing running VM '$vm_name'." ;;
  *) fail "VM '$vm_name' is '$vm_state'. Inspect it with '$cli --json vm list' before retrying." ;;
esac

run_args=(--json run --prefix "$project_root" --device "vm:$vm_name" --detach --yes --build-type docker)
[[ -z "$builder" ]] || run_args+=(--builder "$builder")
[[ -z "$build_host" ]] || run_args+=(--build-host "$build_host")
say "Building and deploying the vacuum simulator to vm:$vm_name."
result="$("$cli" "${run_args[@]}")" || fail "Wendy deployment failed. The VM is preserved. Check the build output above and retry."
printf '%s\n' "$result"
if ! app_url="$(python3 -c '
import json, sys
result = json.load(sys.stdin)
url = result.get("url")
if not url:
    url = next((entry.get("url") for entry in result.get("endpoints", []) if entry.get("url")), None)
if not isinstance(url, str) or not url.startswith(("http://", "https://")):
    raise SystemExit("No HTTP URL in Wendy detached result")
print(url)
' <<< "$result" 2>/dev/null)"; then
  say "This Wendy release returned text. Reading this VM's actual HTTP forward from QEMU."
  app_url="$(python3 - "$vm_name" "$project_root" <<'PY'
import json
import os
from pathlib import Path
import socket
import sys
import time
import urllib.request

name, project = sys.argv[1:]
config = Path(os.environ.get("WENDY_CONFIG_DIR", str(Path.home() / ".wendy")))
directory = config / "vms" / name
try:
    state = json.loads((directory / "state.json").read_text())
    if state.get("name") != name or state.get("netMode") != "user" or not state.get("pid"):
        raise ValueError("the VM has no active user-networking record")
    os.kill(state["pid"], 0)
    manifest = json.loads((Path(project) / "wendy.json").read_text())
    guest_port = next(e["port"] for e in manifest["entitlements"] if e["type"] == "http")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(3)
        connection.connect(str(directory / "qmp.sock"))
        with connection.makefile("rwb") as stream:
            greeting = json.loads(stream.readline())
            if "QMP" not in greeting:
                raise ValueError("invalid QEMU monitor greeting")

            def execute(command, arguments=None):
                request = {"execute": command}
                if arguments is not None:
                    request["arguments"] = arguments
                stream.write(json.dumps(request).encode() + b"\n")
                stream.flush()
                while True:
                    reply = json.loads(stream.readline())
                    if "event" in reply:
                        continue
                    if "error" in reply:
                        raise ValueError(str(reply["error"]))
                    return reply["return"]

            execute("qmp_capabilities")
            listeners = execute("human-monitor-command", {"command-line": "info usernet"})
    host_port = None
    in_net0 = False
    for line in listeners.splitlines():
        if line.startswith("Hub "):
            in_net0 = line.strip().endswith("(net0):")
            continue
        fields = line.split()
        if (in_net0 and len(fields) >= 6 and fields[0] == "TCP[HOST_FORWARD]"
                and fields[2] == "127.0.0.1" and fields[4] == "10.0.2.15"
                and fields[5] == str(guest_port)):
            host_port = int(fields[3])
            break
    if not host_port:
        raise ValueError(f"this VM has no loopback forward for guest TCP port {guest_port}")
    url = f"http://127.0.0.1:{host_port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 30
    while True:
        try:
            with opener.open(url + "/api/health", timeout=2) as response:
                health = json.load(response)
            if health.get("app") != "robot-vacuum" or not health.get("ready"):
                raise ValueError("the forwarded HTTP service is not the ready vacuum simulator")
            break
        except (OSError, ValueError) as error:
            if time.monotonic() >= deadline:
                raise ValueError(f"vacuum readiness check failed at {url}: {error}")
            time.sleep(0.5)
    print(url)
except (OSError, ValueError, KeyError, StopIteration) as error:
    print(f"Could not verify the legacy VM app URL: {error}", file=sys.stderr)
    raise SystemExit(1)
PY
)" || fail "The app may have started, but its HTTP URL could not be verified. Inspect the VM or use a newer Wendy CLI."
fi
say "Vacuum viewer: $app_url"
say "Detached deployment does not check readiness. Verify $app_url/api/health before opening the viewer."

runtime_dir="$project_root/.runtime"
mkdir -p "$runtime_dir"
viewer_state="$runtime_dir/$vm_name-viewer.json"
viewer_url="$(python3 - "$viewer_state" "$app_url" <<'PY'
import json, pathlib, sys, urllib.request
try:
    state = json.loads(pathlib.Path(sys.argv[1]).read_text())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(state["url"] + "/viewer-health", timeout=2) as response:
        health = json.load(response)
    if health.get("app") == "robot-vacuum-viewer" and health.get("upstream") == sys.argv[2]:
        print(state["url"])
except (OSError, ValueError, KeyError):
    pass
PY
)"
if [[ -z "$viewer_url" ]]; then
  nohup python3 "$project_root/scripts/view-vm.py" --upstream "$app_url" \
    --port "${WENDY_VIEWER_PORT:-18884}" --state-file "$viewer_state" \
    >"$runtime_dir/$vm_name-viewer.log" 2>&1 </dev/null &
  viewer_pid=$!
  viewer_url="$(python3 - "$viewer_state" "$viewer_pid" <<'PY'
import json, pathlib, sys, time, urllib.request
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    try:
        state = json.loads(pathlib.Path(sys.argv[1]).read_text())
        if state["pid"] == int(sys.argv[2]):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(state["url"] + "/viewer-health", timeout=1) as response:
                if json.load(response).get("app") == "robot-vacuum-viewer":
                    print(state["url"])
                    raise SystemExit(0)
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(0.1)
raise SystemExit("The local VM viewer did not start. Check .runtime/*.log.")
PY
)" || fail "Could not start the local VM viewer. The VM app remains running."
fi
say "Open the VM viewer: $viewer_url"
