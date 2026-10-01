# Run in a WendyOS VM

This project runs the MuJoCo simulator, vacuum controller and HTTP server in one
Linux container inside a real WendyOS ARM64 virtual machine. The browser renders
the virtual home using the server's state. The VM disk and local Wendy settings
stay outside this repository.

You need the Wendy CLI, QEMU with ARM64 UEFI firmware, Python 3, and an image
builder. Docker with a running engine works on the supported development hosts.
Apple Container also works on Apple Silicon. QEMU uses HVF acceleration on Apple
Silicon, KVM on ARM64 Linux when available, and slower CPU emulation elsewhere.

On macOS, install QEMU with `brew install qemu`. Install Wendy using the official
installer or the Wendy Homebrew tap. Start Docker before using its builder.
The first VM creation downloads the published WendyOS image and creates a
16 GiB sparse disk. The container build also downloads its base image and Python
dependencies. Later runs reuse the VM and image caches.

From this repository:

```sh
./scripts/run-vm.sh
```

The script creates `house-vacuum` if needed. It starts a stopped VM with 4 CPUs,
4096 MiB RAM, and host agent ports 50061 and 50062. It preserves every other VM
and reuses `house-vacuum` if it is already running. The app listens on guest port
18881. Wendy forwards the app port to the host and prints its actual browser URL.
Use that URL because the guest's private address is not directly reachable with
default QEMU user networking.

The script also starts a loopback viewer relay, normally at port 18884. It serves
the repository's static assets and forwards API calls to the VM one at a time.
This avoids QEMU refusing simultaneous browser connections. The simulator and
controller still run entirely inside WendyOS. Open the final "VM viewer" URL
printed by the script. If port 18884 is occupied, the relay chooses a free port.
Rerunning the script reuses a healthy relay for the same VM.

You can start the relay separately without rebuilding:

```sh
python3 scripts/view-vm.py --upstream http://127.0.0.1:18881
```

The relay's process ID and URL are in `.runtime/<vm-name>-viewer.json`. To stop
the relay, stop that process. Stopping the VM makes its API unavailable until
you restart the VM. Set `WENDY_VIEWER_PORT` to choose a different host port.

The script then runs:

```sh
wendy --json run --prefix . --device vm:house-vacuum \
  --detach --yes --build-type docker
```

Recent Wendy releases print JSON with the app and forwarded HTTP URL.
`readiness: "not_checked"` means Wendy acknowledged the start. Check `/api/health`
at that URL to confirm the server is ready. Older releases can emit a text result
even with `--json`. In that case the script reads the named VM's actual loopback
port mapping from QEMU's monitor and verifies the vacuum's health endpoint before
printing its URL. It never assumes the guest and host HTTP ports match.

The CLI's `vm` command is hidden from its general help in some versions.
`wendy vm --help` shows it. The script omits `--profile` so older releases can
create a generic VM too.

## Choose the CLI or builder

`WENDY_CLI` is the name or path of one executable. For example:

```sh
WENDY_CLI=/path/to/wendy ./scripts/run-vm.sh
WENDY_BUILDER=docker ./scripts/run-vm.sh
WENDY_BUILDER=apple-container ./scripts/run-vm.sh
```

If Docker is unavailable and Apple Container is already running on Apple
Silicon, the script selects Apple Container. An explicit Apple Container choice
allows Wendy to start its system when needed. Building a Dockerfile with Apple
Container still uses `--build-type docker` because that flag selects the project
format.

A configured Linux Wendy build host can build the image remotely:

```sh
WENDY_BUILD_HOST=my-linux-builder ./scripts/run-vm.sh
```

The build host must have the builder role enabled, a running BuildKit daemon, and
support for `linux/arm64`. The target VM must be enrolled so the build host can
reach its registry through Wendy's device mesh. This route does not require a
local Docker engine. Choose either `WENDY_BUILD_HOST` or `WENDY_BUILDER`.

## VM options

Use a different VM name or free adjacent agent ports if needed:

```sh
WENDY_VM_NAME=vacuum-demo WENDY_VM_PORT=50063 ./scripts/run-vm.sh
```

Pin the downloaded WendyOS release for a new VM with `WENDY_VM_VERSION`, or supply
a local published disk image with `WENDY_VM_IMAGE`. These options only apply when
the named VM does not exist. They never replace an existing disk.

```sh
WENDY_VM_VERSION=0.19.4 ./scripts/run-vm.sh
WENDY_VM_IMAGE=/path/to/wendyos.wic ./scripts/run-vm.sh
```

To inspect or stop this project's VM:

```sh
wendy --json vm list
wendy vm logs house-vacuum
wendy --json device info --device vm:house-vacuum
wendy vm stop house-vacuum
```

Stopping the VM keeps its disk and project container. The script does not remove
VMs, alter the saved default device, or attach to physical robot hardware.
