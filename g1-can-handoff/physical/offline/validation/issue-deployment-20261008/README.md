# Native inference build recovery, without robot movement

The seven recorded failed deployment attempts represented four separate causes:
remote delivery attempted a 7,481,342-byte image assembly RPC against a
4,194,304-byte target limit; NX2 has no BuildKit; the Mac has no Docker runtime;
and the existing Apple build VM could not reach apt/pip indexes. Repeating these
same routes cannot repair an application. The installed remote-build branch
returns before the local `--chunking` handling, so `--build-host ... --chunking
off` does not provide the needed registry-only transfer workaround.

`prepare_offline_yolo_build.py` prepares a separate, network-free Stagefile
context from 13 pinned arm64 wheels and 34 pinned Ubuntu runtime packages. The
host downloads over HTTPS and verifies all 47 artifact sizes and SHA-256 hashes
against `offline_yolo_dependencies.json` before unpacking anything. Cached
corruption, missing offline artifacts, source pin drift, distribution version
drift, unapproved sources and escaping paths reject preparation. Torch and
torchvision remain supplied by the original CUDA base digest; no CPU torch wheel
or replacement CUDA wheel can enter the overlay. No apt maintainer scripts run,
and no host proxy, network configuration, apt configuration or package cache
is copied into the runtime.

The Ubuntu package records were resolved from official noble, noble-updates
and noble-security arm64 indexes. Their Packages.xz SHA-256 and length matched
the corresponding signed InRelease entries. A separate native build-only probe
verified all three InRelease signatures using gpgv and the Ubuntu keyring from
the original digest-pinned CUDA base. Signed Release snapshots are retained
here. The committed dependency lock fixes these reviewed artifact hashes;
routine preparation does not silently resolve new package versions.

Positive native Stagefile build: **2m5.24s**, with only COPY operations for app
dependencies, weights, fixtures and source. `native-build-positive.txt` records
the command output. A separate build-only probe then imported the real pinned
torch2.7/CUDA12.8, numpy2.2.6 and OpenCV4.11 and ran all **10 saved qualification
frames** through the actual service's packed-forward, letterbox and decoder on
CPU. That RUN passed in 4.4 seconds. This proves the composed runtime libraries
and saved-frame recognition, not a robot GPU qualification or physical pickup.
The CPU probe does not enable a CPU fallback in the deployed service.

Preparation example, from physical/:

```sh
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python \
  prepare_offline_yolo_build.py \
  --context /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-trial-fixes/.runtime/offline-yolo-context \
  --cache /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-trial-fixes/.runtime/offline-yolo-cache
```

Add `--offline` to require a complete existing cache. Always reprepare after
source changes; `offline-build-provenance.json` records the six exact deployed
Python source hashes. Binaries and expanded roots are kept in ignored .runtime,
not committed. Keep the original source Stagefile and descriptor untouched.

The controlled positive command is **build-only**, in the generated context:

```sh
DOCKER_CONFIG=/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-trial-fixes/.runtime/native-public-registry \
  wendy build --dockerfile build.stagefile.yaml --builder apple-container \
  --device 10.10.20.238:50052
```

The explicit Stagefile is required if build-only diagnostic Dockerfiles are
also present. For a separately authorized deployment, the audited native
`wendy-run-timed` wrapper must use that same explicit Stagefile, target,
`--builder apple-container --chunking off`, and isolated DOCKER_CONFIG; omit
`--build-host`. This selects the available local builder and regular registry
transfer rather than the failed remote assembly path. Verify six source hashes,
all 10 GPU qualifications, matching controller/YOLO budgets, and PASSIVE/no
ownership afterward. Transfer/readiness on NX2 remains **unverified** in this
build-only subtask, and no robot app was restarted or motion command sent.

`test_offline_yolo_build` passes 12 focused tests. The saved native failure
fixture retains the original oversized RPC symptom, while the positive build
and fixture proof demonstrate the app-level alternative. No Agent/OS patch,
public CLI release, Docker installation, BuildKit installation or shared service
restart was required.
