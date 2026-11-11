# Nine native stale-perception failures: same-input motor-gate replay

This replay covers all nine stale stops in the preserved error ranking. It
uses original successful observation metadata, worker events and primary
failure times; it does not replace their source timestamps with the current
clock. Source file hashes and available exact RGB-D ZIP/timing paths appear in
each saved report. Missing original image inputs remain timeline-only cases.

Run from `g1-can-handoff/physical` in the project's Python environment:

```sh
python -m offline.stale_replay --max-age-ms 400 --publication legacy --report offline/validation/stale-perception-20261008/red400.json
python -m offline.stale_replay --max-age-ms 400 --publication receipt --report offline/validation/stale-perception-20261008/receipt400.json
python -m offline.stale_replay --max-age-ms 500 --publication receipt --repeats 10 --report offline/validation/stale-perception-20261008/green500.json
python -m unittest test_stale_trial_replay test_perception_worker test_perception_timing test_motion_tracking -q
```

Verified using
`/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python`:

| Same recorded observations | Accepted scenarios | Exit |
| --- | ---: | ---: |
| Original background-only publication, 400 ms | 0/9 | 1 |
| Receipt-first shared publication, 400 ms | 3/9 | 1 |
| Receipt-first shared publication, 500 ms, ten repeats | 9/9 | 0 |

The 18 focused regression tests pass. Each scenario replays the interval from
its original failure through the next successful worker publication. It checks
every 20 ms command tick and the instant immediately before each intervening
publication. Since exposure age increases monotonically between updates, those
boundary checks cover the entire recorded interval's age envelope. The worst
corrected exposure age is 460.51 ms, leaving 39.49 ms below the tested limit.

The real `PhysicalPlant._execute_poses` calls its real `check` and the real
`Runtime.tracking_guard`; fresh synthetic body/hand feedback isolates the
perception issue. Arm/hand sinks only record calls. All original 400 ms stops
block those sinks. Future timestamps, exposures beyond 500 ms, tracking
uncertainty and latched cancellation also block both sinks. This exercises the
command-write admission seam; it does not issue DDS commands or reproduce the
robot's mechanics, collisions, temperature or original body telemetry.

The receipt-first result is a bounded counterfactual of the existing handoff
fix. Availability is the maximum of the successful `perception_result` event
timestamp and the recorded child geometry-completion timestamp, calculated
from exposure plus reported exposure age. It never precedes either. The
packet is eligible only after pairing its successful geometry with the
subsequent successful `perception_worker_result`. Original behavior
waits until the later worker-result event and publishes only background
observations. Neither mode changes image exposure, inference duration, frame
identity or metadata. The current worker's receipt-before-event publication
also has independent IPC regressions in `test_perception_worker` and the
contention regression in `test_perception_timing`. A new regression proves
archived frame evidence with failed geometry never publishes a motor-guard
observation.

Timestamp provenance was explicitly checked against worker source: the
child's `event()` buffers `(phase, fields)` without creating `monotonic_ns`.
After `_observe_frame` completes geometry, joins the archive and accepts the
tracker state, the child sends the entire result and event list. Parent
`PerceptionWorker.observe` receives that packet before relaying its events;
parent `Runtime.event` creates `monotonic_ns` then. Consequently buffered
`phase_camera_evidence` timestamps in these paired packets are parent relay
times even though the archive event was originally buffered before fitting.
The replay nevertheless uses the later successful `perception_result` time,
rather than relying on the camera-evidence event name or that inferred earlier
receipt bound. The conservative rerun retains the same 0/9, 3/9 and 9/9 verdicts.

The evidence separates two causes: publication/phase-observation handoff is
sufficient for three scenes; six retain real processing/camera gaps beyond
400 ms. The bounded 500 ms setting covers these nine recorded gaps. It is not
evidence that the native pipeline is faster, that every future gap will fit,
or that a physical pickup succeeds. No production source was changed by this
replay task. The tested 500 ms descriptors remain distinct from a verified
deployment, and native timing validation remains outstanding.
