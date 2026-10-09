# NX2 arm characterization

The controller starts passive. This lane reuses the commissioned arm SDK and
independent DDS recorder. It commands arms with PD gains and gravity feedforward;
there is no integral term in this interface. Finger commands are disabled.

`harness_calibration.py` implements one bounded trial. It stores the request,
compiled MuJoCo model, collision preflight, chronological encoder samples, latest
SDK reference, gains, torque estimates, casing/winding temperatures and result
under `/data/calibration-trials/<trial-id>/`. It holds measured positions and
releases arm SDK ownership on success or failure. Stop remains available.

`offline/run_calibration_trial.py` runs one trial through native Wendy. Its
read-only default sends no start command. Actuation requires the harness, clear
workspace and empty-hand attestations; the operator must establish those facts.
Output filenames cannot be reused. Deployed Python must match local source.
Evidence transfer is checksum verified; large compiled models use a separate
streamed read-only endpoint and local content-addressed model directory.

## Bounds and signals

| Setting | Bound / default |
| --- | --- |
| Shoulder/elbow P | 20–100, default 50 |
| Shoulder/elbow D | 1–6, default 3 |
| Wrist gains | min(P, 40), min(D, 3) |
| Small joint excursion | ±0.02–0.12 rad, default ±0.06 |
| Small test maximum travel from start | 0.3 rad |
| Small test reference velocity / acceleration | 0.12 rad/s / 0.5 rad/s² |
| IK displacement | at most 40 mm |
| Recorded replay maximum excursion | 1.5 rad from measured start |
| Recorded replay reference velocity / acceleration | 0.5 rad/s / 2 rad/s² |
| Admission temperature | both arms at most 55°C on both sensors |
| Absolute temperature stops | casing 60°C; winding 65°C |
| Temperature-rise stop | 8°C for small tests; explicit 8–12°C for longer demo replay |
| Torque estimate stops | 12 Nm shoulder/elbow; 4 Nm wrist |
| Measured speed stop | 0.7 rad/s |
| Joint tracking stop | 0.15 rad |
| Deadline | 35 s small tests; 110 s recorded replay |

These experimental ceilings are independent of pickup environment overrides.
Normal feedback/fault, joint-range, base-state, collision and recorder checks
remain active. The 200 ms hand feedback gate is unchanged; its reader now polls
at 20 ms instead of 100 ms. A missing DDS sample does not refresh its timestamp.

Encoder FK is derived from the saved robot model, not an independent measurement
of the physical wrist. Reported torque is `tau_est`, not a calibrated current or
external force measurement. Temperature samples have integer-degree resolution.
A short trial does not establish steady-state thermal behavior.

## Reproduce a recorded demo trajectory

The preserved fixture at
`offline/validation/harness-calibration-20261009/demo-staging-reach-fixture.json`
contains 1,700 chronological targets from the 9 October staging/approach/align/
reach failure. Its forward duration is 39.613833673 s. The original timestamps
are preserved and targets interpolated at the 20 ms experiment cadence. The
entry to the recorded start and reverse-path return are separately labelled;
the final recorded target is held for five seconds.

From this `physical` directory, after establishing the stated setup:

```sh
python3 offline/run_calibration_trial.py \
  --output offline/validation/my-new-replay.json \
  --kind demo_replay \
  --replay-file offline/validation/harness-calibration-20261009/demo-staging-reach-fixture.json \
  --kp 75 --kd 3 --max-temperature-rise 12 \
  --actuate --harness-confirmed --workspace-clear --hands-empty
```

Omit `--actuate` to inspect the request without starting motion. A new output
filename is required every time. Source changes require deployment via the
existing `g1_can_handoff/redeploy.py`, which invokes the native timing wrapper.

Offline plotting and raw DDS export:

```sh
python3 offline/analyze_calibration_trials.py offline/validation/my-new-replay.json
python3 offline/export_calibration_dds.py offline/validation/my-new-replay.json
```

Raw export waits for sealed recorder segments and preserves all callback-delivered
DDS records in contiguous timestamp intervals. Upstream packet loss is unknown.
A transfer error is recorded separately and does not replace the physical outcome.

The replay changes the contact/loading environment: empty hands, harness support,
no table contact, current base orientation, and measured right-arm hold. Gains
and feedforward are recorded explicitly. It reproduces the left joint path and
requested chronology; it is not an identical physical pickup or thermal-contact
experiment. Comparing it to a table-contact failure cannot isolate gain effects
from the changed environment.

## IK profiles

Use `--kind ik --solver baseline|jac_scaled|nearby_regularized|damped_jacobian`
with `--offset x y z` in metres. The first three compare SciPy least-squares
scaling/reference regularization; the last uses an independent analytic damped
Jacobian method. Every target and the intervening motion are collision checked.
Solver preflight is distinct from measured physical execution.
