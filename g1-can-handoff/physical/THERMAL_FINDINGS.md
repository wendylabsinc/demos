# NX2 Coke demo: shoulder heating and completion status

Evidence reviewed on 2026-10-09. **Overheating under sustained reach load is the
main demonstrated motion blocker in the analyzed run.** Solving that load is
the next controller priority and may let this setup complete the demo, but a
complete physical pickup/lift has not been established by these findings.
It is not yet supported to say overheating is the only remaining problem:
the later saved retry stopped before motion on table-depth perception.

## What the recording establishes

Robot: Unitree G1 Nx 2. Recorder session: `20261009T191749Z-f6bb3fab`.
Start accepted: 2026-10-09 19:24:31.294168 UTC. Primary failure:
`ValueError: body temperature 85 C reaches the 85 C guard or is invalid`,
100.115756 seconds after Start. The run reached **reach**, then **close**;
it stopped about 205.5 ms after close began, without a verified lift.

The affected body joint was **index 16, left shoulder roll**. Its reported
casing/winding pair at the first threshold crossing was **[53, 85] C**;
every other body joint was at or below 54 C in that sample. This was localized
shoulder winding heating, not every joint or the shoulder casing reaching 85 C.

The independent recorder captured both `rt/arm_sdk` commands and `rt/lowstate`
feedback. A checksum-verified replay of sealed segment 008 gives:

| Seconds after Start | Mean target (rad) | Mean measured position (rad) | Mean reported torque magnitude (Nm) | Last sampled winding (C) |
| --- | ---: | ---: | ---: | ---: |
| 94 | -0.0421 | -0.0368 | 0.96 | 55 |
| 96 | -0.1413 | -0.0841 | 7.33 | 56 |
| 98 | -0.2074 | -0.0880 | 17.01 | 73 |
| 99 | -0.2074 | -0.0881 | 16.97 | 83 |
| 101, after the stop | -0.0881 | -0.0791 | 1.44 | 81 |

The complete temperature trace first reaches 85 C at **100.098453 seconds**.
The table uses one-second summaries of independently sampled streams; the
temperatures in the table are last selected samples, not bucket maxima.
At 98 seconds, the greatest selected speed magnitude is only **0.0245 rad/s**.
The recorded shoulder gains are **Kp = 150, Kd = 3**. The feedforward term is
about **+1.06 Nm** at 98-99 seconds, substantially smaller than the approximately
17 Nm reported output magnitude. `tau_est` is firmware-reported torque, not
an independent calibrated torque measurement or a measured current trace.

The failure hold reduces the position target error and the reported load.
Cooling was observed **after the stop**, not while continuing the reach.

## Is this a PID problem?

**Yes, gain selection is a plausible contributor, but this is a contact/control
interaction rather than evidence that changing P alone fixes the task.**
The exposed command interface has position/velocity targets, proportional and
derivative gains, and feedforward torque. This run supplies no integral gain
through that interface; it is more precise to describe our commanded loop as PD
plus feedforward. These records do not establish the firmware's internal loops.

The approximate torque demand is:

`tau = Kp * (q_target - q_measured) + Kd * (dq_target - dq_measured) + tau_ff`

At 98-99 seconds the position error is about **-0.119 rad**, so the proportional
term alone is approximately **-17.9 Nm**. Adding the recorded feedforward and
small velocity term is consistent with the approximately **-17 Nm** reported
torque. The controller continues to demand a position while measured motion
has almost stopped. Sustained motor current generates winding heat even when
the joint is not moving; copper loss grows with current squared.

The operator reported table contact during the motion. That is consistent with
the blocked movement, growing error and large sustained torque, but the archive
contains no independent contact-force measurement proving the table is the
constraint. Other mechanical constraints and model mismatch remain possible.
Repeated execution of the same constrained path with the same stiffness would
be expected to reproduce the heating; this analysis is one recorded load
episode, not an exhaustive attribution across every historical session.

## Casing, winding and stop thresholds

Unitree's G1 SDK labels `temperature[0]` casing and `temperature[1]` winding.
Its protection helper defaults are 85 C casing and 120 C winding, with a
strict greater-than comparison. These are helper protection defaults, **not
verified continuous-operation recommendations or guaranteed firmware cutouts**.
The analyzed controller stopped at a global 85 C guard, so the winding reading
triggered it while the casing was 53 C.

A later saved controller build, `0.3.25-winding-120`, distinguishes 85/120 C
casing/winding guards. Changing a stop threshold changes when execution is
interrupted; it does not remove the recorded high-load heating mechanism.
That later build was produced from commit
`f78dfab5feea30c90c03a34c37afd78c0d447391` **with local edits**, source digest
`ee038c3838c9e15774883bd73be562d85170ab3105d8e6ffb07a3eb0606b0343`.
This documentation branch does not contain or deploy those local code edits.

## What to solve and how to know the demo is complete

1. Resolve the contact/target mismatch. Verify the measured table geometry and
   hand path; if contact is intentional, use a defined contact-control phase
   with bounded torque and appropriate stiffness rather than continually
   demanding an obstructed position.
2. Compare appropriate contact gains against the recorded baseline. Evaluate
   peak and sustained reported torque, position error, temperature rise and
   successful progression. Lower P can reduce obstruction load but may worsen
   free-space tracking; D mainly affects motion and oscillation and cannot
   remove a persistent static position error by itself.
3. Restore reliable table perception for the retry setup. The later retry
   stopped with `ValueError: Too few yellow tape depth points`, with **zero
   motion commands**, no arm ownership and no physical lift. Raising the winding
   guard therefore did not by itself demonstrate completion.
4. Record a full physical close/grasp/lift and stable hold with healthy feedback
   and controlled thermal behavior. Repeat on the same previously failing
   setup to establish consistency before marking the demo complete. Simulation
   and accepted Start requests are not substitutes for measured pickup success.

**Working expectation:** eliminating the sustained reach load is likely to
remove this recorded thermal stop and is the main controller work item. Whether
it also completes pickup remains a hypothesis until the remaining phases run
successfully on hardware. No gain change, deployment or robot movement was
performed to produce this report.

## Evidence and reproduction

- [Temperature report, graphs, CSV and phase events](offline/validation/body-temperature-demo-20261009T192431Z/README.md).
- [Full temperature summary](offline/validation/body-temperature-demo-20261009T192431Z/summary.json).
- [Sampled command/feedback/torque replay](offline/validation/body-temperature-demo-20261009T192431Z/shoulder-load-summary.json).
- [Read-only load extraction script](offline/validation/body-temperature-demo-20261009T192431Z/extract_shoulder_load.py).
- [Later retry summary](offline/validation/winding-120-20261009/retry-summary.json) and [native trial record](offline/validation/winding-120-20261009/retry-trial/trial-01.json).

The temperature archive includes 150,423 decoded body records with no empty
250 ms display buckets. Callback timestamps align phases; hardware exposure
timestamps and loss-free upstream DDS delivery are not proven by this archive.
The original temperature `artifact-manifest.json` verifies its original 11
members. `analysis-manifest.json` additionally verifies this report's evidence
files and replay additions. Raw multi-megabyte sealed DDS segments remain on
the recorder; their SHA-256 identifiers are preserved, not the raw segments
themselves. The later retry's source archive identity is also preserved in its
JSON, while the archive is not bundled here.

To reproduce the load summary in an environment containing Unitree SDK2 Python:

```sh
python3 extract_shoulder_load.py /records/20261009T191749Z-f6bb3fab/samples-000008.jsonl.gz
```

The script checks the exact segment SHA-256 before decoding and reads archived
bytes only. It creates no DDS publisher or live subscription.

Primary references:

- [Unitree G1 temperature protection helpers](https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/common/terminations.hpp).
- [Unitree G1 arm command example](https://github.com/unitreerobotics/unitree_sdk2_python/blob/master/example/g1/high_level/g1_arm7_sdk_dds_example.py).
- [Maxon motor fundamentals: current, torque and thermal loading](https://www.maxongroup.com/medias/sys_master/8815460712478.pdf).
