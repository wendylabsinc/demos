# G1 NX2 offline failure regression pack

Use this pack to continue developing after the robot is gone. It includes the
actual checkpoint, images, calibrated rays/depth, exposure-paired poses, and
failure traces from the October 7–8 Coke-can trials. No robot, Wendy CLI,
RealSense SDK, Unitree SDK, or CUDA GPU is needed for the main profile.

## Run at home

From a clone of `wendylabsinc/demos`, branch
`codex/g1-nx2-yolo-coke-detector`:

```sh
cd g1-can-handoff/physical
python3.12 -m venv .venv-offline
. .venv-offline/bin/activate
python -m pip install -r offline/requirements.txt
python offline/run.py
```

The default profile runs **184 regression tests**, verifies captured evidence
checksums, and writes `offline/reports/latest.json`. Each module runs in a fresh
process with temporary runtime storage, fixed random/hash seeds, a 400 ms
exposure limit, and bounded execution. Reports retain individual test IDs,
failures, stdout, and stderr. A nonzero exit means a real failure or timeout.

The runner blocks real DDS imports and network connections. The only permitted
HTTP connections are to ephemeral loopback servers created by the tests in that
same process. Spawned perception children inherit the same restrictions.

For the portable Linux environment, including headless simulation:

```sh
docker build -f offline/Dockerfile -t g1-offline .
mkdir -p offline/reports
docker run --rm --network none --read-only --tmpfs /tmp \
  -v "$PWD/offline/reports:/reports" \
  g1-offline --report /reports/regression.json
```

Build/install dependencies once while online. Runs then need no external network.
The container base is pinned by digest and Python packages are pinned. macOS
native execution is also tested; the portable reference uses Python 3.12/Linux.

## Profiles and focused debugging

```sh
python offline/run.py --list
python offline/run.py --test test_recorded_failures
python offline/run.py --test test_perception_timing
python offline/run.py --test test_target_tracker.TrackerTests
python offline/run.py --profile open --report offline/reports/open.json
python offline/run.py --profile simulation --report offline/reports/simulation.json
python offline/run.py --profile all --report offline/reports/all.json
python offline/inspect_trace.py --attempt 20261008T004101Z
python offline/inspect_trace.py --attempt 20261008T004012Z --events
```

`regression` covers fixed defects and required rejection behavior. `open` is an
intentionally red acceptance test: a cold SDK ownership change must happen before
the final measured pickup plan used for execution. It uses fake actuator
endpoints and proves the current runtime still uses a plan from before ownership.
It is neither skipped nor marked expected-failure. `simulation` runs 13 MuJoCo
control/dynamics tests; **five currently fail** with changed starting posture,
final-reach target movement, or an earlier unreachable-path error. `all` combines
these three profiles and currently exits nonzero. These failures are work to do,
not a reason to relax movement gates.

The latest hardware trial also has a real pipeline headroom problem: frame 45594
returned at 322.94 ms age and frame 45596 arrived after the watchdog stopped at
411.8 ms. The archived timing replay asserts that the next arm/finger write is
blocked. Passing that test proves correct rejection, **not a latency fix**. Use
the full trace, source timestamps, worker timings, and archived camera source to
work on capture/transport/processing latency; CPU wall time cannot qualify Jetson
CUDA timing.

An untested control-settle/replan proposal was saved in
`experiments/control-settle.patch`. It is not applied, not deployed, and not a
verified fix. Start with the red `open` profile before editing that behavior.

## Run the actual YOLO checkpoint on CPU

The default tests exercise semantic/image identity and geometry with frozen
responses or fake transport. This optional profile runs real neural inference on
the pinned checkpoint and all **ten** annotated positive/negative images, with
the same fixed-size preprocessing and one-class decoder used by the deployed
service:

```sh
python -m pip install -r offline/requirements-yolo.txt
python offline/run.py --profile model --report offline/reports/model.json
```

This passed on the existing Python 3.11/macOS environment with Torch 2.14.1 and
Ultralytics 8.4.123. It uses CPU float32; hardware uses CUDA FP16, so this checks
recognition, not bitwise GPU parity or runtime speed. Positive cases require one
unambiguous detection strictly above 10% confidence and IoU at least 0.5. Negative
crops require no accepted detection. Training images and same-scene correlated
views are labeled in `yolo/fixtures/qualification.json`; ten images are not an
independent estimate of general detection accuracy.

## Saved evidence and provenance

- `attempts/`: all ten original trial snapshots, including start response,
  before/after status, calibration, and reported error.
- `trial-events.json.gz`: 4,204 full ordered controller events from the same ten
  attempt intervals, including commands, first faults, poses, geometry,
  processing time, and exposure age. Absolute `/data` paths inside are provenance,
  not dependencies. `inspect_trace.py --events` prints these locally.
- `evidence/rgbd/`: five exact original JPEG/depth-PNG ZIPs, original timing NPZs,
  and a shared losslessly compressed calibrated ray map. Frames 44504/44725 cover
  cold acquisition; 45594/45596 bracket the last stale-frame stop. The camera
  client decodes the original bytes; the planning replay checks the original BGR
  image SHA against the recorded YOLO response before fitting geometry.
- `evidence/attitude-replay.npz` and `attitude-poses.json`: 153 measured point sets
  and nearest decoded lowstate poses, preserving original exposure timestamps
  and measured waist joints. The subset is 512 evenly spaced points per frame
  from the original calibrated table region. This is a calibration replay, not
  independent table metrology. Original full-frame hashes are retained.
- `evidence/table-plane-audit.json` and `../fixtures/table-plane.npz`: original
  tape-plane comparisons and disjoint broad-table train/held-out depth pixels.
- `../fixtures/cylinder-local-minimum.npz`: measured partial cylinder arc that
  formerly converged to the mirrored local minimum.
- `reference/camera/`: exact camera-service, lease, and settings source snapshots
  collected from the running camera app. These are diagnostic reference files,
  not the controller's deployed camera service or an automatically runnable SDK.
- `../yolo/model/` and `../yolo/fixtures/`: tracked checkpoint, model provenance,
  manually annotated recognition fixtures, and recorded qualification reports.
- `evidence/latest-planner-stop.json.gz`: additional later planning failure,
  including rejected continuous-lift IK/collision candidates and its first fault. Frame 51053 is also saved for geometric replay.
- `manifest.json`: SHA-256 of all captured artifacts; source collection used
  controller commit `1538fd1`. Trials span several earlier fixes; this does not
  claim every trial ran the same commit.

The first 00:12:46 trial is an important historical case: its raw primary fault
was **stale exposure at 478.3 ms**, but final status overwrote it with “Operator
cancelled trial.” Both are preserved. `test_runtime_lifecycle` prevents that
masking behavior from returning.

## Coverage and unresolved work

See [CASES.md](CASES.md) for the failure-to-test mapping, exact trial IDs, and the
remaining open cases. GitHub Actions runs the regression profile for relevant
PR/push changes and uploads the JSON report. Manual workflow dispatch can run
`open`, `simulation`, or `all` to expose the outstanding failures.

A home-development loop is: choose a captured failure, run its focused test,
change the production code, rerun that test, then run the regression profile.
For handover work, the open acceptance test must go red → green while stale-frame
and target-movement rejection tests continue to pass. Never rewrite captured
expectations/checksums merely to make a code change pass. Add a new labeled
fixture when new evidence changes the expectation.

No trial in this archive verified a physical lift. Simulation, fake transport,
and saved-image inference do not prove a live grasp, thermal behavior, balance,
release response, or pickup success; those require later hardware validation.
