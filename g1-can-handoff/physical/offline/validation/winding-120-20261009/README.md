# Requested winding guard 120 C

Prepared the Nx2 controller descriptor with
`HANDOFF_BODY_WINDING_TEMPERATURE_LIMIT_C=120`, version
`0.3.25-winding-120`. Casing stays 85 C and hand guard stays 75 C.
The library fallback remains 95 C when no winding override is supplied.
Unitree's SDK protection helper uses 120 C winding and 85 C casing; these are
protection thresholds rather than continuous-operation ratings:
https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/g1/common/terminations.hpp

18 focused temperature/passive-deployment/redeployment tests passed. The new
test exercises the failed shoulder reading of 95 C under the 120 C override,
the retained stop at 120 C, and the independent casing stop at 85 C.
Unrelated pre-existing physical_core.py edits were preserved.

Read-only preflight on G1 Nx2 `10.10.20.238:50052` rejected unhealthy feedback.
Two subsequent fresh samples showed left-hand fault array
`[0,0,512,0,0,0,0,0,0]` and `telemetry_error` equal to
`left motor fault or missing fault feedback`. Shoulder casing/winding had cooled
to `[55,56]` C. Controller remained PASSIVE, arm ownership released, no demo
running; its command counter stayed 17167 across both reads. No physical retry
or deployment was issued. The live winding guard remains 95 C until the idle
healthy preflight permits applying the new descriptor.

Baseline deployed source matched the pre-edit local digest
`613c38d09094f2e93b8bf3c1eafe90131b613162d85fe1238354856d21c6f662`,
build version `0.3.24-sensor-temperature-guards`, commit
`f78dfab5feea30c90c03a34c37afd78c0d447391` plus existing local edits.
The operator was asked to check the left Dex3 hand and report when the fault
has cleared before the requested deployment/retry.

## Verified deployment and retry after restart

Fresh readback confirmed `0.3.25-winding-120`, source ee038c3838c9e15774883bd73be562d85170ab3105d8e6ffb07a3eb0606b0343 and casing/winding guards 85/120 C. `startup-verified.json` matched deployed source hashes and build stamp, healthy recorder, advancing camera frames, PASSIVE state and zero commands. 61 focused regression tests passed, including the 95 C acceptance, 120 C winding stop and retained 85 C casing stop.

One requested retry was accepted and stopped before motion with `ValueError: Too few yellow tape depth points`. The controller ended BLOCKED, with no arm ownership and zero motion commands. Native trial export and every member checksum were verified locally; trial JSON and archive are under `retry-trial/`. The local attention beep completed. No second retry was issued. The earlier left-hand fault preflight above remains as historical evidence.
