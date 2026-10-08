# Six native planner failures: offline acceptance

No DDS imports, publishers, controller lease, network calls or physical commands
are used by this replay. `physical_lift_verified` stays false.

The previous test only checked that a collision was rejected, or froze the
planner's clock while trying three old scenes. Neither proves that the recorded
failed placement can produce a complete path within the real deadline.

## Reproduction and change

`clearance-original-task-and-core.json` uses the captured trial-03 physical-core
source and the `11a582c` task planner on the same recorded calibration, can centre
and staging start. It fails with `No reachable grasp and offer path for this can
position; preflight rejected: Interpolated physical trajectory intersects
robot/table` (13.94 seconds on this Mac; the native Jetson originally exhausted
30 seconds). This is a real IK/collision rejection, not an injected exception.
The historical task source is saved as `source_task_11a582c.py` so the test does
not need Git or a network connection.

The physical caller previously solved four endpoints independently with three
IK seeds each, then solved and validated the entire continuous path again.
Those disconnected endpoint branches could discard a valid continuous branch
before preflight. `task.plan` now delegates directly to the existing full-path
validator when supplied. Backends without a validator still run endpoint IK.
`PhysicalPlant.preflight` still checks every arm segment, both finger paths,
continuous Cartesian align/reach/lift, joint limits and table/robot collisions.
The 30-second total planner budget and 10-mm positive table clearance remain.
No penetration tolerance was added.

The current physical-core source also contains the previously prepared elbow /
torso null-space avoidance. The captured failed clearance scenes predate that
change. `clearance-original-core-current-task.json` isolates the task change:
that old core can find a path in 12.65 seconds with the new task, while the full
current task+core finds a path in approximately 2.7 seconds. These are local CPU
measurements, not deployed Jetson performance or successful pickup claims.

## Commands

Run from the physical checkout with the project's NumPy/SciPy/MuJoCo environment:

```
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python -m offline.planning_failures_replay --repeat 3
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python -m unittest test_planning_acceptance test_clearance_planning test_planning_deadline test_pickup test_physical_core
```

`test_planning_acceptance` proves the actual captured-source rejection and current
acceptance. It also locks down no-validator endpoint rejection and negative
full-path-validator rejection. `endpoint-red.txt` separately captures the old
planner's redundant disconnected solver call; this is the smaller call-contract
regression, not the primary native failure reproduction.

## Evidence limits

All six native failed *scenes* have been enumerated and replayed. Four have a
preserved measured feedback snapshot. Two (`owned-staging-collision` and
`additional-owned-search-timeout`) lack full native feedback; their replay uses
explicit fallback feedback and is labelled scene-only in every result. The
additional-attempt record also lacks its exact RGB-D planning frame and source
snapshot. They cannot count as fully captured input acceptance.

Even the four measured snapshots are terminal records, reconstructed with the
recorded calibration's base joints/IMU and recorded staging start where present.
Receipt times are rebased solely for the offline harness. This does not recreate
the hardware's dynamic hand/arm tracking state at every execution sample.
The returned paths pass the native collision checker plus an independent
MuJoCo distance audit. This is planning evidence; it does not prove execution
settling, finger accuracy, grasp or lift. A historical staging collision during
actual hand motion can still require a measured-state execution replay that the
current export lacks.

`six-scenes-current-repeat03.json` records three consecutive real-clock passes
of every scene, and includes input SHA256, minimum table and elbow/torso gap,
all planner events and explicit missing fields. No clock freezing is used for
orientation search or staging search.
