# Tuned planner: real trials and offline replay

NX2 asset501 at `10.10.20.238:50052`, 2026-10-09 UTC / October 8 Pacific.
Native controller deployment `f93a2c0e-f1be-446f-8333-8b629e623d03`, source
`ffc06ef1bcdcc947506bffaa4574b31ace32d3e6`, version `0.3.19-noise-aware-ik`.
Readiness passed with exact source/profile hashes, fresh decoded advancing
camera frames and healthy recorder. Tuned profile: three fully checked grasp
candidates, 0.1 rad orientation spacing, 5 mm endpoint tolerance, fitted
anisotropic covariance; P150/D3 shoulder/elbow and P40/D1.5 wrists.
Phase-boundary color/depth perception and existing guards remain enabled.

| Trial | Native terminal error | Demo commands | Ownership |
| --- | --- | ---: | --- |
| 1 | Waist moved beyond the 0.002 rad unmodeled alignment bound | 0 | Never acquired |
| 2, after operator stationary confirmation | Grasp orientation search time limit reached | 0 | Never acquired |

The initial read-only setup probe also reported insufficient yellow tape depth
points. The first actual trial subsequently measured the table successfully;
its archived primary error is waist alignment, not table support.

Both native archives are checksum-verified and preserve all trial events, exact
boundary RGB-D payloads, planner/profile sources, and (trial 2) 267 numerical
decision snapshots plus the 117 MB compiled model. The large compiled model is
kept inside the compressed archive; extracted directories are ignored.

## Reproduced failures

`error-replay.json` reproduces both original error strings without hardware.
Trial 1 uses its saved terminal status: waist displacement was 0.01612384 rad
from the calibration reference. The exact constructor feedback read was not
recorded, so this is a replay of the same guard with recorded terminal feedback,
not a claim of exact failing-sample replay.

Trial 2 uses the exact three candidate inputs and recorded successful validation
outcomes. Injecting their measured durations into the real `task.plan` deadline
logic reproduces its error: 11.18682 + 11.10559 + 11.17966 = **33.47207 s**, over
the existing 30 s budget. No candidate collision or IK failure caused this stop.
The deadline fired before final selection/revalidation.

## Numerical and timing gap

All 267 individual decisions replay without errors on the Mac. Strict exact
comparison passes 135 joint-interpolation decisions; 129 IK decisions and three
whole-candidate preflights differ numerically. The largest IK difference is
**1.737924e-6 rad**. Do not relabel this as bitwise equality.

Whole-candidate Mac replay takes 2.476 / 2.225 / 2.028 s, compared with about
11.1 s per candidate during the native trial. A separate **motion-free NX2
replay** of candidate 1 is bitwise identical and takes 11.122 s, versus 11.187 s
originally. This isolates a meaningful platform/runtime performance gap;
snapshot recording alone does not explain it. The replay compares identical
versions of MuJoCo 3.14.0, NumPy 2.2.6 and SciPy 1.15.3.

These are planner/state/timing replays. Neither trial moved the arm, so there is
no new physical joint-tracking or pickup-success comparison. Final controller
state is BLOCKED, demo idle, no arm ownership, zero demo commands.

## Repeat locally

From the physical directory, with the pinned project Python environment:

```sh
python -m offline.replay_real_stops offline/validation/tuned-real-replay-20261008 --extract --output offline/validation/tuned-real-replay-20261008/error-replay.json
python -m unittest test_real_stops_replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m offline.replay_planning offline/validation/tuned-real-replay-20261008/trial-02/planning --kind preflight --kind ik --kind continuous_ik --kind joint_poses --output offline/validation/tuned-real-replay-20261008/numerical-replay.json
```

The last command deliberately returns nonzero for the observed cross-platform
bitwise differences; its report retains all errors and numerical deltas.
The regression test passes when both historical stops reproduce as expected.
It does not turn a stopped demo into a successful pickup.

Next work: budget ranking/validation for measured NX2 execution time and include
that time distribution in simulation. Preserve full collision validation and a
fresh observation before motion. Also persist the initial constructor read when
alignment rejects it, closing trial 1's exact-feedback capture gap.
