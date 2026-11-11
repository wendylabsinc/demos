# NX2 recorded reach and physical gain comparison

Nine harness replays were completed using identical 418 left-arm joint requests. The operator confirmed harness support and a clear arm sweep. There were no finger commands or physical pickup attempts.

| Shoulder/elbow P | Trajectory RMS error (rad) | Final settling-window maximum (rad) | Outcome |
| --- | ---: | ---: | --- |
| 80 | 0.012292 | 0.041821 | Settling failure reproduced |
| 100 | 0.011085 | 0.041221 | Settling failure |
| 110 | 0.010648 | 0.040874 | Settling failure |
| 120 | 0.010341 | 0.040598 | Settling failure |
| 130 | 0.010147 | 0.040323 | Settling failure |
| 140 | 0.009992 | 0.040047 | Settling failure |
| 150, run 1 | 0.009757 | 0.039783 | Passed, returned and released |
| 150, run 2 | 0.009670 | 0.039795 | Passed, returned and released |
| 150, run 3 | 0.009696 | 0.039795 | Passed after default deployment, returned and released |

P150 improved mean trajectory RMS error by 21.0% relative to P80. The final settling margin is still small: about 0.000205 rad below the unchanged 0.04 rad limit. This is repeatable evidence for the recorded empty-hand reach on a harness, not proof of an entire loaded Coke pickup.

Selected setting: shoulder/elbow P150, D3; wrist P40, D1.5; no integral term. Deployment `0.3.18-arm-kp150` was verified against all 52 runtime source hashes, then physically replayed successfully. Final independent status confirmed PASSIVE, no adapter ownership, advancing healthy feedback and no further published commands. Peak measured motor temperature across the sweep was 55 C.

[Measured comparison plot](physical-gain-comparison.png) and [machine-readable comparison](physical-gain-comparison.json) include all nine attempts. Each `physical-p*-*.json` has before/after state, gains, error and metrics; its matching `.events.jsonl.gz` contains commanded/measured joints, velocity, feedback age, torque and SDK reference. Every local event archive checksum was verified, and identical left joint sequences were checked before comparison. The independent passive recorder retained raw DDS throughout.

The final published SDK reference matched the requested target to floating-point precision; increasing P improved physical following without altering the requested trajectory. Six failures were retained rather than hidden by changing the settling limit.

70 focused tests passed: 65 guard/gain/packaging/feedback tests, two new recorded-feedback regression tests and three offline-report checks. `test_harness_regression.py` replays P80's recorded failure and P150's recorded success through production `PhysicalPlant._execute_poses` with a deterministic clock, without camera/DDS/robot hardware. These are recorded-feedback tests, not a dynamics simulation. The offline lab includes these cases under “Arm tracking and gain replay.”

Simulation still predicts substantially better absolute following than hardware. `replay-p150-comparison.json` gives P80/P150 final errors of 0.004211/0.002397 rad, compared with physical 0.041821/0.0398 rad. Do not use the simulator's absolute servo error as a physical guarantee.

The bounded harness endpoint requires explicit support and clear-workspace confirmation. It preserves joint limits, self collision, fresh feedback, faults, thermal guards, base/waist stability, tracking error, cancellation, exclusive ownership, recorder health and deadlines. The requested arm reference is limited to 0.25 rad/s and 1 rad/s2. Fingers are not commanded. A verified return releases control; failure holds measured position.

To roll back the selected proportional gain, set `HANDOFF_ARM_KP` in the deployment descriptor to 100 or 80 and redeploy while released using the native timed Wendy deployment path. Set the same variable for simulation, or pass the gain explicitly. Saved fixture SHA256 is `29a6ec9df2be0b5778fe51d1764f3dec8012602f092413a98b059450cce6583f`.
