# NX2 demo diagnosis, 2026-10-07

Target: `unitree-g1-nx-2`, `10.10.20.238:50052`, camera `338622073335`.
Physical pickup attempts are operator authorized. The operator changed the
semantic confidence requirement from strictly above 30% to strictly above 10%.
The 400 ms exposure limit, independent measured geometry, calibration,
collision, telemetry, recorder and ownership guards remain enabled.

The native controller preserves exact RGB-D archives and source bundles in
`/data`. Each perception result now records the exact image identity, model
checksum, raw detections, inference/processing/exposure timings and any error
in its event journal, including observations that fail before planning.

## Reproduced failures

The latest start with the previous 30% setting failed at initial planning with
`Coke YOLO target absent/ambiguous`, zero motor commands and no arm adapter.
Saved frames 95265 and 95373 reproduce this deterministically offline: the
repaired checkpoint scores one can at 29.411% and 28.653%, respectively. Both
source ZIPs match the archived RGB and depth pixels exactly, with 0.121 ms
RGB/depth timestamp skew. No camera pairing defect was found in these frames.

At the requested 10% setting, frame 95373 has a 2.08 mm cylinder residual and
2.20 mm table-plane median error; its estimated center is inside the left-hand
workspace. This is an offline replay, not a current observation or proof of
pickup. Frame 94670 is outside the left-hand workspace at y=0.0527 m and must
continue to be rejected.

The previous checkpoint also scores an empty-table crop at 11.89%. Therefore
the new 10% negative recognition qualification initially fails. Retain that
failure in the ledger; do not remove the negative fixture to make deployment
pass. Any training use of a qualification image changes its provenance to a
training fixture and requires separate held-out evaluation.

Compact evidence and archive checksums are retained in
`yolo/validation-demo-attempts-20261007.json`. Full local copies are under
`.runtime/demo-debug`; robot-local originals remain under `/data`.

## Repeatable operator tools

From the repository root, capture status without starting motion:

```sh
python3 g1-can-handoff/physical/record_demo_attempt.py
```

Explicitly run one already-authorized pickup attempt and record its result:

```sh
python3 g1-can-handoff/physical/record_demo_attempt.py --start
```

The runner records before/after status and native events locally, requests Stop
and hold after a timeout or uncertain command acceptance, and never retries
Start automatically, releases held joints, or restarts a controller. Review
and resolve the recorded failure before invoking another attempt.

Offline replay preserves original archived timestamps and never calls hardware:

```sh
.venv/bin/python g1-can-handoff/physical/yolo/diagnose_archived.py \
  .runtime/demo-debug/camera-95373-planning.npz \
  --events .runtime/demo-debug/events-20261007T224339Z.jsonl \
  --confidence .10 --output .runtime/demo-debug/replay.json
```

## Visual/depth tracking version

Initial recognition uses YOLO at strictly above 10%. Between semantic checks,
a fresh RGB-D tracker selects the previously identified object using its
measured depth, bounded world position and visual appearance. YOLO rechecks
identity at least once per second and immediately when the tracker has
insufficient support, ambiguous components or uncertain visual agreement.
Every proposed track still passes the existing cylinder/table geometry and
400 ms exposure checks before its state is accepted.

During movement a background worker updates the track continuously. Every
arm/finger command tick checks the latest accepted exposure; stale or failed
tracking freezes the measured pose. The worker cannot issue motion commands.
The pre-close path retains the 10 mm object-motion limit, and post-close
tracking retains the 40 mm center limit. The final lift still requires five
fresh observations showing the measured rise and hold.

The 10% classifier update rehearses the two known empty crops while retaining
the original training images. All feature and box-regression tensors remain
unchanged. Its original validation recall changes from 81.25% to 79.17%, and
test recall from 95.77% to 94.37%; precision improves on both splits. The
negative crop qualification and four additional correlated same-scene crops
pass. These are bounded qualification results, not broad scene generalization.

86 focused tests pass, including tracker identity/reacquisition, geometry,
stale-frame command blocking, archive completion, cancellation and release.
The prior deployed version and the pre-tracker source snapshot are preserved
on GitHub; see `ROLLBACK.md`.

The first deployed tracker attempt completed the geometric plan but stopped
before arm ownership because the old 180 ms acquisition budget rejected a
frame. Measured acquisition age varied from 163 to 213 ms. The revised
acquisition budget reserves one quarter of the configured final deadline
(300 ms acquisition at a 400 ms final limit); all completed observations and
every movement tick still require exposure age at or below 400 ms.

The final continuous passive probe accepted 45 observations: 41 visual/depth
tracks and four semantic checks. The maximum exposure age sampled every
10 ms was 382.19 ms. This includes exact-frame archive writes and one-frame
prefetch; no motor adapter or camera lease was acquired by the probe.


## Deployed 0.3.1 physical attempts

Commit `a863c066670957859f244cdf91e1dc69bc08fed4` was deployed to NX2.
The 23:07:10 UTC attempt passed planning, but a tracking observation reached
488 ms during controller initialization. It froze before any `target` or
`hand_target` event. SDK measured-hold packets were emitted; their count is
not a count of physical movements. Heavy SDK initialization is a suspected
source of the startup stall, not a proved sole cause.

The 23:08:26 UTC explicit pregrasp retry reused the initialized controllers
and reran fresh recognition and the complete path check. It emitted five
opening-hand target events, then froze before arm travel or closing. The
last accepted frame was 281 ms old, and its replacement completed about
160 ms later at 375 ms old. Thus the accepted observation could cross the
400 ms deadline while its replacement was still processing. The watchdog
currently overwrites its initial fault with `Operator cancelled trial` in
the execution loop, so the exact triggering guard was not retained. This is
a diagnostic limitation; no user cancellation or successful pickup is inferred.

The controller retained ownership with `grasp_started=false` in both failure
journals. Do not restart/redeploy while joints remain held. Another attempt
requires reviewed evidence and a new explicit invocation; the local runner's
`--start --resume-pregrasp` permits a FROZEN hold while the native controller
still refuses any active trial or previously begun grasp.

Full attempt files are `.runtime/demo-attempts/20261007T230710Z.json` and
`20261007T230826Z.json`; compact evidence is included in the tracker validation
report. Next work is reducing the under-control processing tail and recording
the first watchdog fault. No completed physical pickup has been verified.


## Isolated perception trial, 2026-10-08 00:12 UTC

Controller `3e3c771` (0.3.3-isolated-perception) ran RGB-D processing and
exact-frame archiving in a spawned process without motor adapters. The
passive native probe accepted 45 observations, with a maximum independently
sampled exposure age of 380.62 ms. The 24 focused tests passed before the
physical trial; deployed runtime, worker and physical-core hashes matched.

The trial passed planning, opened the hand and entered arm staging. It
stopped before grasping: the first watchdog fault was
`Visual/depth tracking exposure stale (478.3 ms)`. The controller retained
ownership in FROZEN state. Its 1,146 recorded command writes are not a count
of movements and do not establish pickup. No lift was verified.

The worker's last pre-fault computation took 135.83 ms, but parent round-trip
processing took 211.50 ms. Parent event replay delayed publication of that
observation; its full frame records were also synchronously duplicated to
container stdout. This identifies an avoidable delay, but does not establish
that logging explains every latency spike. Candidate `71989c2` separates file
logging from the controller-state lock, suppresses duplicated per-frame
console output while retaining JSONL evidence, and propagates the first
failure into demo status/attempt summaries. Its 25 focused tests pass; this
candidate still requires deployment and another measured physical trial.

See `yolo/validation-isolated-worker-trial-20261008.json` for compact native
evidence. The full local attempt is
`.runtime/demo-attempts/20261008T001246Z.json`.

## Cylinder initialization repair, 2026-10-08 00:20 UTC

The logging candidate was deployed, but the next trial stopped before grasping
at 407.1 ms. Its first fault is now preserved in both controller status and
the attempt summary. No successful pickup is claimed. The robot retained its
measured hold. A subsequent retry was stopped while investigating perception.

Passive profiling reproduced intermittent 6 mm cylinder-fit failures with an
upright, unobstructed can. The captured partial arc converges to a mirrored
local minimum when initialized at the points' mean: center (0.2767, 0.1073),
RMSE 6.68 mm, 14 optimizer evaluations. An algebraic circle initializer finds
the correct basin: center (0.3290, 0.1338), RMSE 2.92 mm, eight evaluations.
The repaired implementation uses that initializer and an analytic Jacobian.
The measured point fixture is retained in `fixtures/cylinder-local-minimum.npz`.
It changes no acceptance threshold or calibration.

If a proposed visual track still fails the cylinder fit, the tracker now
reacquires YOLO identity on that exact frame and repeats the full geometric
checks. Failed reacquisition never reuses an old mask or commits tracker state.

61 focused tests pass. A 20-second native passive probe accepted 267
observations with 21 YOLO checks, no geometry errors, no age violations, and
a maximum continuously sampled exposure age of 367.87 ms. Median processing
was 58.95 ms versus 64.41 ms in the preceding 12-second passive baseline.
This is a bounded timing result, not proof of freshness throughout motion.
See `yolo/validation-cylinder-initializer-20261008.json`.

Disabling prefetch, a continuously refreshed one-slot reader, and the existing
raw RGB-D transport were tested without actuators. They did not consistently
improve latency, so the candidate retains the existing camera transport and
one-frame prefetch. Controller replacement and physical verification of the
cylinder repair remain pending supported release.


### Deployed cylinder repair trial

After supported release, native Wendy deployed `e4893b1`; runtime, camera,
tracker and physical-core hashes were verified. The bounded 00:21:31 UTC
trial entered arm staging with valid cylinder fits and no first-fault
freshness failure. It stopped before grasping on
`Table plane differs by over 5 mm`. No physical lift was verified.

Replay of the archived table neighborhood gives an initial median offset of
-1.46 mm and a failed-frame offset of -5.02 mm. Body pitch changed from
0.03515 to 0.02256 rad between initial and final telemetry. This is consistent
with camera/base attitude drift, but endpoint samples do not prove exact
exposure-time motion. The fixed camera-to-world transform needs validation
under arm load before another calibration change or physical trial. The table
guard remains unchanged and the controller remains in a measured failure hold.
See `yolo/validation-cylinder-trial-20261008.json`.

### Exposure-synchronized attitude correction

The 00:28:09 UTC warm retry refreshed calibration while retaining the existing
arm controller. It still stopped during staging on table-height drift, so a
startup-only refresh was insufficient. Across 153 recorded frames, matching
each exposure to recorded body attitude (maximum skew 1.87 ms) and rotating
the camera about the calibrated pelvis reduced the table offset range from
[-0.63, 5.17] mm to [-0.90, 0.46] mm. The shared production helper reproduced
this result; the 5 mm table guard is unchanged.

The candidate retains a bounded history from the existing body-state reader.
Perception requests the pose nearest its actual exposure, rejects skew above
20 ms, and applies the same rotation to camera and pelvis transforms. Current
attitude also updates the robot collision/gravity model; live arm and finger
targets are checked together. Attitude changes above 0.04 rad and unmodeled
waist movement above 0.002 rad stop the trial. Base translation remains
unmodeled, and this is still the operator-accepted factory/table estimate,
not an independently measured extrinsic calibration.

An initial passive probe stopped before observations because body motor 11
reported 78 C. No motor adapters were created by the probe. After the operator
reset, the controller reported PASSIVE, no arm adapter, healthy telemetry,
and a maximum body temperature of 64 C (then 60 C). Live validation of the
candidate and a physical pickup remain pending.
