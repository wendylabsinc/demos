# G1 NX2 offline failure regression pack

## Today's demo simulation and failure lab

Run the current color/depth tracker in MuJoCo and replay the October 8 failures:

```sh
python offline/run.py --profile today --report offline/reports/today/latest.json
```

Open `offline/reports/today/index.html` for the outcome matrix, expandable
individual test results and animated MuJoCo replays. The run is fully offline:
the existing sandbox blocks real DDS imports and external network connections.
Each run writes its own animation directory, so older renders cannot be confused
with new results. Use the environment installation below on a fresh computer.

The lab covers initial/periodic recognition, stale RGB-D, table-height drift,
pickup bounds, ownership settling, collision planning, target movement,
start/stop lifecycle and placement overlay behavior. A passing suite includes
**expected safe rejections**; it does not claim every scene can be picked up.
Actual saved frames and native timing/feedback traces test the recorded errors.
Synthetic fault injection also removes the target or makes its exposure 600 ms
old immediately before alignment, requiring a stop before alignment commands.
The measured-position hold command on a stop is counted separately.

The new contact runs use `TargetTracker(identity_mode='color_depth')`, the same
geometry, planner, arm reference, finger control and pickup sequence as hardware.
They assert zero YOLO calls, and cover both external and approximate head-camera
views. The previous synthetic-red tests remain as a comparison baseline.

This is not a calibrated digital twin: the base is supported, the can is an
untextured cylinder, hand friction/gains are modeled, feedback is synthetic, and
camera geometry is approximate unless supplied explicitly. Simulation now adds
recorded, correlated XY estimate drift after RGB-D geometry; see the position-noise
section below. It does not model USB/HTTP queues, CUDA inference latency, realistic
lighting or depth-pixel noise. Those failures use recorded-data replays rather than
invented physics. Neither kind of offline test proves a physical pickup will succeed.

**Current known failure:** the external-view color/depth pickup passes, while
the approximate head-camera view stops at lift verification with
`Color/depth can target absent/ambiguous`. This remains a failing test rather
than an expected failure or a skipped test. It reproduces the target-loss
symptom with YOLO disabled; matching the real failure's cause still requires
comparison with the physical recording. The `today`, `simulation` and `all`
profiles return nonzero until this case is repaired.

Replay its saved synthetic frame in under a second without running dynamics:

```sh
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python offline/replay_sim_frame.py fixtures/sim-head-color-depth-20261008.npz
```

This narrower repro pins the original exposure clock and exercises color/depth
reacquisition; it does not reconstruct the earlier tracker history. The saved
frame and report are listed in `offline/validation/simulation-lab-20261008`.

To render just one demo or one fault (under the same offline sandbox):

```sh
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python sim_runtime.py --height .762 --camera head --capture-animation --output offline/reports/head-demo.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python sim_runtime.py --height .762 --fault stale-align --capture-animation --output offline/reports/stale-demo.json
```

Create the output directory first for standalone commands. A fault demo returns
nonzero because its pickup was stopped; the `today` suite instead checks that
this stop was the expected outcome. GIF playback samples simulated motion at
5 fps and is not a benchmark of the real robot's speed.

Use this pack to continue developing after the robot is gone. It includes the
actual checkpoint, images, calibrated rays/depth, exposure-paired poses, and
failure traces from the October 7–8 Coke-can trials. No robot, Wendy CLI,
RealSense SDK, Unitree SDK, or CUDA GPU is needed for the main profile.

## Trajectory parity against real recordings

```sh
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python offline/trajectory_parity.py --output offline/reports/trajectory-parity.json
```

The current interpolation code exactly reproduces 1,647 logged left-arm target
samples across eight staging segments, given each recorded execution start and
IK goal. This isolates command generation; collision admission is not repeated
by that comparison. It does not claim independent regeneration of those goals.

Two approach IK goals recomputed with current code repeat exactly on this
computer, but differ from recorded goals by at most 0.0000421 and 0.0009585 rad.
Those inputs use the captured planning start and calibration with other joints
from the before-trial snapshot. Complete exact-time model/feedback state is
missing, and code/dependency versions may differ. Therefore full replanning and
the physical trajectory have not been shown to be identical. The generic demo
animation is a synthetic scene, not a replay of these real command streams.

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

The default profile discovers regression tests, verifies captured evidence
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

The later supervised batch is preserved in `evidence/ten-demos-20261008`:
three numbered trials, 130 native events, 15 exact RGB-D frames, accepted
calibration, terminal feedback and deployed source copies. Each member is
checksum-verified by `manifest.json` and `test_clearance_planning`. The batch's
raw DDS segments remain on the robot; the local settling analysis contains
derived window statistics rather than a full raw-feedback export. An additional
attempt's recent events are saved separately under
`validation/ten-demos-20261008/additional-attempt-observed.json`.

To replay planning without the robot:

```sh
python offline/run.py --test test_clearance_planning
python -m offline.planner_replay --trial 1
python -m offline.planner_replay --trial 2
python -m offline.planner_replay --trial 3
```

The replay rebuilds the accepted geometry and replaces only model state with
recorded calibration/hold/staging poses. It rebases receipt timestamps for
offline checks and never constructs command publishers. It does not reproduce
live timing, feedback dynamics or physical grasp success. Returned arm samples
are independently checked with MuJoCo geometry-distance queries; all three
scenes passed with at least 32.5 mm modeled table clearance at a required 10 mm.
The earlier planner also eventually found a path on the faster Mac (7.14 s),
after two rejected candidates. The final replay, including independent distance
auditing, took 4.84–5.41 s while other validation jobs ran concurrently. These
are Mac observations, not a controlled speed comparison or Jetson benchmarks.
Full trajectory/finger collision checks remain active. Staging search is deferred
until alignment, reach, finger closing and lift have passed geometric checks,
so infeasible grasps do not spend the expensive staging budget.

The high-first grasp experiment passed the static geometry replay but caused
a simulated head-camera lift verification failure. Its exact RGB-D frame,
events, source hashes and truth-only diagnostic pose are preserved in
`evidence/head-camera-offset-20261008`. The can shifted 39.7 mm while the camera
fit measured 40.3 mm. `test_clearance_planning` still requires this old frame to
fail the unchanged 40 mm bound. The final planner prefers the can-centre grasp
and checks higher candidates only if needed. The head-camera dynamics test
now passes without changing the tracking gate. Original red and final green
reports remain under `validation/ten-demos-20261008`.

```sh
python offline/run.py --list
python offline/run.py --test test_recorded_failures
python offline/run.py --test test_perception_timing
python offline/run.py --test test_latest_trial_replay
python offline/run.py --test test_record_demo_attempt
python offline/run.py --test test_target_tracker.TrackerTests
python offline/run.py --profile open --report offline/reports/open.json
python offline/run.py --profile simulation --report offline/reports/simulation.json
python offline/run.py --profile all --report offline/reports/all.json
python offline/inspect_trace.py --attempt 20261008T004101Z
python offline/inspect_trace.py --attempt 20261008T004012Z --events
python offline/inspect_trace.py --attempt 20261008T005915Z --events
```

`regression` covers fixed defects and required rejection behavior, including the
four ownership acceptance cases. `open` runs those four cases alone with fake
actuator endpoints: final planning follows measured ownership settling; failed
perception, cancellation, or unstable control cannot execute the preview plan.
`simulation` runs both the current color/depth and legacy MuJoCo control/dynamics
tests. `all` runs regression and simulation without duplicating ownership cases.
The head-camera color/depth case above is currently red. No tests are skipped
or marked expected-failure to conceal it or the original failures.

The latest hardware trial also has a real pipeline headroom problem: frame 45594
returned at 322.94 ms age and frame 45596 arrived after the watchdog stopped at
411.8 ms. The archived timing replay asserts that the next arm/finger write is
blocked. Passing that test proves correct rejection, **not a latency fix**. Use
the full trace, source timestamps, worker timings, and archived camera source to
work on capture/transport/processing latency; CPU wall time cannot qualify Jetson
CUDA timing.

The runtime now holds the existing arm reference with bounded gravity updates
during finger operations and settling. Cold ownership must produce at least
500 ms of advancing, stable measured feedback within five seconds, then refresh
calibration and build a new pickup plan. Recreating the calibrated model retains
the hold reference and torque ramp. Cancellation is checked before entering IK.
These changes pass offline acceptance and simulation; they are not deployed or
hardware-qualified. The original untested proposal remains in Git history.

Before/after reports, individual test outcomes, source hashes, and historical
red proofs are retained in `validation/home-fixes-20261008.json` and
`validation/home-fixes-20261008/`. The original five simulation failures and
the ownership failure are preserved alongside the passing results. Collision
rejections remain visible when orientation search exhausts its candidates.
The staging test compares executed waypoints with the validated plan and requires
zero pregrasp contacts instead of forcing a particular number of segments.

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
  before/after status, calibration, and reported error, plus later labeled trial snapshots.
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
- `evidence/20261008T005915Z/`: the next live attempt's complete 1,023-event
  timeline, five exact RGB-D/timing pairs and ray map. A periodic YOLO result
  arrived about 9 ms after the 401.7 ms exposure stop. The replay checks the
  actual motor write gate and cancellation latch; latency remains unresolved.
- `evidence/20261008T010351Z-85369a28/`: the following attempt was cancelled
  by a Release request during planning, before any motion. Its full timeline,
  RGB-D samples, source snapshot, and recorder status are preserved.
- `manifest.json`: SHA-256 of all captured artifacts; source collection used
  controller commit `1538fd1`. Trials span several earlier fixes; this does not
  claim every trial ran the same commit.

The first 00:12:46 trial is an important historical case: its raw primary fault
was **stale exposure at 478.3 ms**, but final status overwrote it with “Operator
cancelled trial.” Both are preserved. `test_runtime_lifecycle` prevents that
masking behavior from returning.

## Coverage and unresolved work

See [CASES.md](CASES.md) for the failure-to-test mapping, exact trial IDs, and the
remaining open cases. GitHub Actions runs the complete `all` profile for relevant
PR/push changes and uploads the JSON report. Manual workflow dispatch can select
`regression`, `open`, `simulation`, or `all` for focused checks.
CI gives each module a 300-second wall-clock budget: the full simulation module
took 178 seconds on a GitHub runner, close to the local runner's default 180.
This test-process budget does not change controller planning or motion deadlines.

A home-development loop is: choose a captured failure, run its focused test,
change the production code, rerun that test, then run the regression profile.
For handover work, retain the ownership acceptance cases while stale-frame and
target-movement rejection tests continue to pass. Never rewrite captured
expectations/checksums merely to make a code change pass. Add a new labeled
fixture when new evidence changes the expectation.

No trial in this archive verified a physical lift. Simulation, fake transport,
and saved-image inference do not prove a live grasp, thermal behavior, balance,
release response, or pickup success; those require later hardware validation.

## Recording further hardware trials

On the operator machine, while connected to the authorized NX2, run one attempt:

```sh
python record_demo_attempt.py --start --output ../../.runtime/demo-attempts
```

`--start` issues real motion through the normal controller. Each invocation checks
independent recording before Start, polls one bounded attempt, and writes a
uniquely named status JSON and summary row. It exports the full controller event
timeline and up to five exact camera source/timing pairs, calibrated rays, and
deployed source files into a local tar archive. Both the archive and each member
must pass SHA-256 verification. Transfer failure is saved explicitly and returns
nonzero. First faults and per-attempt command counts remain separate from later
cancellation and cumulative controller counters. A failed pickup also returns
nonzero even when its evidence was successfully archived.

Recorder before/after health, persisted sample counters, session identity, and
sealed raw DDS segment hashes are retained in the JSON. The independent recorder
keeps all delivered body/hand feedback and commands on the robot; the tar archive
contains controller events and a bounded image subset, not every raw DDS message
or every camera frame. Upstream DDS loss cannot be proven absent by recorder
counters. Review each terminal failure before another invocation; the runner
never releases arms, restarts the controller, or repeats an uncertain Start.

To copy a trial's raw DDS records after its recorder segments have sealed:

```sh
python archive_trial_recording.py ../../.runtime/demo-attempts/TRIAL.json
```

This read-only command verifies sealed segment hashes, preserves the original
base64 CDR records from the trial interval and adjacent rotation boundaries, and
verifies the local archive and members. It requires the original recorder
session and refuses an unsealed tail. The recorder's exact `export_cdr.py` is
included for later decoding with its SDK dependencies. The export code passes
offline tests for interval boundaries, all topic retention, checksum failures,
and reordered callbacks. The attempted live raw export failed because the NX2
became unreachable; raw DDS transfer is **not yet hardware-verified**. The
controller event/image archives were already transferred and verified locally.
Connection failures also produce local error records, with Start marked as
unattempted when Wendy initialization fails.

## Recorded position noise in simulation

`sim_runtime.py` now defaults to `--position-noise recorded`. A seeded whole-trace
selection replays the XY offsets from sixteen phase observations in two October 8
trials, preserving millimetre jitter, the 10.4 mm control-handover shift, and the
11.6 mm approach drift. Noise is applied to the geometry result before the
production tracker accepts it; the planner and all later checkpoints see the same
noisy estimate. The physical can in MuJoCo stays at its original pose until
physics moves it. Observation timestamps, Z, table height, depth-fit residuals and
lift height are not fabricated.

```sh
mkdir -p offline/reports/position-noise
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python sim_runtime.py --identity-mode color_depth --setup-json offline/validation/position-noise-20261008/setup.json --noise-trace approach-drift --output offline/reports/position-noise/drift.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python sim_runtime.py --identity-mode color_depth --setup-json offline/validation/position-noise-20261008/setup.json --position-noise none --output offline/reports/position-noise/ideal.json
python offline/run.py --test test_simulation_position_noise
python offline/run.py --test test_sim_position_noise
```

The drift run should stop at reach with `Can or table moved during approach;
replanning required`; it returns nonzero intentionally. The regression tests assert
that stop before a reach/close command. Existing ideal physics tests explicitly
set `position_noise='none'`; noisy cases have their own lab coverage.

`--noise-seed 7` repeats recording selection; `--noise-trace handover` selects the
other pattern. Selection weights are equal for test coverage, **not measured
probabilities**. `--noise-scale 1` preserves recorded offsets; other scales are
stress tests. Missing phases retain the final offset and mark it `extrapolated`
in each observation. A second preflight reuses the last recorded planning/refresh
occurrence when the source already held control.

This is an empirical failure model, not calibrated sensor accuracy: no independent
can ground truth was recorded, so some observed drift may be actual movement.
Two traces do not establish a general noise distribution, pixel noise, temporal
spectrum, or postgrasp variability. Sampling is by movement phase, not elapsed
seconds. The result records seed, selected trace, scale, profile checksum, per-frame
offsets, unperturbed estimates and source frame IDs separately from physical truth.

`g1_can_handoff/position_noise_profile.json` preserves the selected native samples
and source archive/event checksums. To regenerate it from the original archives:

```sh
python offline/fit_position_noise.py --real-directory offline/validation/sim-real-20261008 --output g1_can_handoff/position_noise_profile.json
```

See [the paired simulation report](validation/position-noise-20261008/README.md)
for current ideal/noisy runs and the comparison with the real estimates.
