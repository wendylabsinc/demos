# Bounded 500 ms RGB-D exposure budget

The operator requested a longer exposure allowance after an externally started
trial stopped at 401.7 ms. The native records show frame 41858 had already been
processed at 288 ms age. At the stop, its RGB exposure was 401.64 ms old and its
depth exposure 401.76 ms old, with only 0.121 ms skew. Frame 41860 returned about
38 ms after the stop and was 304 ms old. This was a gap between accepted paired
observations, not evidence of a separate RGB or depth stream stopping. Raw
camera health later showed advancing frames and no restarts.

The native trace, five exact RGB-D frames, calibration and source snapshots are
preserved under `offline/evidence/exposure-failure-20261008-preserved/trial-01`:
2,537 events, 20 export files, archive SHA-256
`55f9673ed3450e491038f06271c38dfa051f76a293b3bba8cfa79539004cefac`.
The initiating Start request was not observed. No duplicate Start was issued.

Both controller and CUDA inference deployment descriptors now specify 500 ms.
The shared validator allows only finite values from 50 through 500 ms, retaining
the 400 ms library fallback for old deployments and offline comparisons. This
adds 100 ms to the accepted exposure age; it does not change frame timestamps,
RGB-depth skew, identity, semantic rechecks, uncertainty, geometry, collision,
robot feedback, cancellation or command-speed checks. A fresh late result still
cannot clear a latched freeze or automatically resume an interrupted trial.

The persisted controller budget also must be updated. The deployment verifier's
explicit `--configure-camera-max-age-ms 500` option does this only after source
hash verification, matching YOLO readiness, PASSIVE state, no ownership, no demo
and zero startup commands. The option must match the deployment descriptor.
Read-only verification remains the default. The supervised runner compares its
verified release's budget with both services before admitting a Start.

`test_latest_trial_replay` replays this exact gap through the real tracking
guard: 400 ms rejects it, 500 ms permits it, and exposures beyond 500 ms or
uncertain tracking still reject. The old 400 ms freeze-and-no-resume replay is
retained. Client and inference tests cover the 500 ms boundary and future
timestamps. A red full regression report exposed the new verifier CLI consuming
the test runner's arguments; explicit argument passing fixes that interface.

Rollback: restore both deployment descriptors to 400 ms and update the
persisted controller budget while idle. The previous deployed source is
`095ee9b`, controller `0.3.11-warm-perception`, inference `0.2.2`.
Offline timing/replay acceptance does not verify a physical pickup or prove
that every future perception delay will fit inside 500 ms.


## Deployment transport diagnosis

Native `wendy run --build-host=10.10.20.208:50052 --detach --yes`
built the inference image but delivery failed because the NX2 Agent rejected
an assembly RPC of 7,481,342 bytes against its 4,194,304-byte receive limit.
CLI: 2026.10.01-013419; NX2 Agent: 2026.10.05-134249; target:
10.10.20.238:50052, asset 501, Jetson Orin NX, arm64/sm_87. The original
inference service remained running. This is a deployment-platform limitation,
separate from the captured camera deadline failure.

The on-device build attempt refused before build because NX2 has no BuildKit
daemon. The default local builder also refused because this Mac has no Docker
runtime app. Neither refusal changed the running robot service. Failure ledgers
and sanitized command output are retained here. No Agent/OS changes or cache
pruning were made.

The native CLI supports the existing Apple Container runtime and
`--builder apple-container --chunking off` for regular registry transfer,
avoiding the oversized assembly RPC. Its build VM, however, timed out reaching
package indexes over HTTP, HTTPS and a signed Ubuntu mirror. The Mac itself
could reach the HTTPS indexes. Restarting only the local image builder was
attempted; robot services and the Agent/OS were untouched. Temporary package
source experiments were reverted. All command failures are retained here.

The 500 ms source change is tested and saved, but deployment is **not verified**.
The running controller 0.3.11 and inference 0.2.2 still use **400 ms**. This
report must not be read as evidence of a deployed limit increase.
