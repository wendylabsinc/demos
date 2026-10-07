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
