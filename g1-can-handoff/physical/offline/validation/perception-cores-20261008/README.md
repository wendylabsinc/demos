# NX2 CPU and perception measurements

The NX2 exposes eight online CPUs and all services already have access to all
eight. The existing body listener uses CPU 6 and motor publisher uses CPU 7.
CUDA executes the pinned YOLO network. Every probe here was PASSIVE, acquired
no camera lease, created no actuator, and left the command counter unchanged.

`world-transform-benchmark.json` compares 32 transforms of one immutable
recorded frame at 1/2/4 workers. Four workers reduced median 12.05 ms to 6.32 ms
and p95 12.97 ms to 9.27 ms, with bit-exact output. Concurrent encoding reduced
median time but increased p95; that prototype was not shipped.

`comparison.json` summarizes the full-path 1/4/4/1 comparison (32 observations
per run). Four workers did not improve full-path latency: accepted round-trip
p50 was 63.84/63.97 ms and p95 was 79.95/87.66 ms for one/four workers.
`HANDOFF_GEOMETRY_WORKERS` supports 1, 2, or 4; the deployed default remains 1.
Setting it to 1 restores serial geometry without changing any pixels or guards.

The initial profiler did not preload `Observation` in its parent. First IPC
unpickling therefore imported SciPy during a timed receive, adding approximately
100-160 ms absent from the real controller. Initial `baseline`, `abba`, and
`warm-ready` reports preserve these failures for audit, but their first-sample
receipt failures must not be interpreted as production controller failures.
Warm-sample timings remain useful. `profile_perception.py` now preloads the type
as the real controller does and no longer sleeps before the first request.

`startup-comparison.json` contains the corrected deployed/candidate/candidate/
deployed comparison. All 128 observations passed. The worker now waits for an
actual decoded frame no older than 75% of the configured exposure budget before
announcing readiness. The first request takes 94-95 ms versus 569-616 ms on the
old worker; readiness moves about 0.5 s of camera initialization before that
request. Total startup time is approximately unchanged. This avoids cold
calibration/decode work inside the bounded observation path; it does not claim
an improvement to steady-state median throughput.

Native Wendy deployed `0.3.11-warm-perception`, source `095ee9b`. All 48 deployed
source hashes match (`controller-readiness.json`). The final deployed passive
profile accepted all 32 observations: exposure age p50 215 ms, p95 247 ms,
maximum 268 ms. Startup was 2.33 s and the first request 104 ms. Command counts
were zero before and after this profile. The normal Start button was restored
after verification; this optimization session did not start a physical trial.

These are passive comparisons, not physical pickup or motion-watchdog
qualification. Demo 4's recorded watchdog failure occurred between accepted
observations, so individually fresh passive observations do not prove the
400 ms bound will hold during motion. The exposure limit, exact frame identity,
tracking uncertainty, geometry, collision and robot-feedback gates are retained.
