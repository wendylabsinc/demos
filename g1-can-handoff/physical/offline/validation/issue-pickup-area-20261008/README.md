# Three captured pickup-area failures: bounded offline acceptance

The physical admission rectangle now permits can-base centers 25.5–45 cm forward
and 6.5–29 cm robot-left (previously 27–45 / 9–29). The 1.5 cm forward and 2.5 cm
left extensions admit two captured placements only after the unchanged full
`PhysicalPlant.preflight` checks IK, commissioned joint limits, every trajectory
segment, opening and closing finger paths, and a positive 10 mm table margin.
Robot collision checks remain unchanged. Admission alone is not reachability.

The controls overlay reads the literal production bounds from the deployed task
AST. Its label now preserves half-centimeter values instead of rounding them to
whole centimeters. The integration test executes that actual read/projection
script against fake IO and verifies the resulting polygon and label.

## Native input coverage and results

| Native failure | Captured center | Current result | Independent returned-path distances |
|---|---|---|---|
| `20261008T003403Z` | forward 50.245 cm, left 12.574 cm | Retained area rejection; no checked path found by bounded diagnostic searches | No accepted trajectory |
| `20261008T003718Z` | forward 25.736 cm, left 15.088 cm | Accepted in three consecutive real-clock replays | table >=17.003 mm, elbow/torso >=18.953 mm |
| clearance trial 02 | forward 36.8 cm, left 6.8 cm | Accepted in three consecutive real-clock replays | table >=17.089 mm, elbow/torso >=2.993 mm |

The input catalog is `../exposure-500ms-20261008/error-ranking.json`.
`old-bounds-red.json` reproduces all three exact reported area errors. The saved
`source_task_old_bounds.py` isolates the old rectangle with the current planner
contract. `three-scenes-current-repeat03.json` retains each input SHA256, original
failure, exact center/calibration, all returned samples/events and timings.
`acceptance_passed=true` means two safe checked plans plus the retained far
rejection. `passed=false` honestly records that all three placements do not pass.
No captured evidence or manifest was rewritten.

## Commands

From `g1-can-handoff/physical`, with the existing native virtualenv:

```sh
PYTHONPATH=offline:. HANDOFF_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m offline.pickup_area_replay --historical-bounds
PYTHONPATH=offline:. HANDOFF_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m offline.pickup_area_replay --repeat 3
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python offline/run.py --test test_pickup_area_acceptance --report offline/validation/issue-pickup-area-20261008/acceptance-tests.json
PYTHONPATH=offline:. HANDOFF_OFFLINE=1 OPENBLAS_NUM_THREADS=1 /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m unittest test_pickup_workspace test_placement_overlay test_planning_acceptance test_clearance_planning test_pickup test_physical_core
```

The acceptance module passed 7 tests (including 3 imported projection checks);
the final workspace/acceptance run passed 10 tests; the related planner,
collision, pickup and physical-core run passed 41 tests. Their exact stdout
and reports are retained in this directory.

Historical-bounds replay intentionally exits 1. Current acceptance exits 0 while
retaining the far rejection. The offline sitecustomize blocks all real DDS imports
and all external network IO. No sender/adapter, publishers, ownership, deployment,
service restart or robot motion is involved.

## Far-placement diagnostics and limits

`diagnostic-expanded.json` temporarily bypasses only the rectangle in an isolated
planner module: all 30 normal rotation/height candidates fail genuine full-path
validation in 21.69 seconds. Rejection counts are 11 required-orientation IK
failures, 11 continuous Cartesian lift IK failures, and 8 positive table-clearance
trajectory failures. Therefore this is not a claim that the 50.2 cm grasp point
alone is geometrically beyond the arm. It is a complete-pickup path failure under
the bounded planner search. Its production forward bound stays at 45 cm.

Two additional bounded diagnostics preserved genuine rejection: lifting the
grasp by 30/40 mm within the cylinder gave 15 continuous-lift IK and 5
orientation IK failures in 24.16 seconds; lowering only the approach offset from
160 to 120 mm gave 9 continuous-lift IK, 13 orientation IK and 8 table-clearance
failures in 23.19 seconds. These diagnostic variants are never installed in the
production planner. No higher grasp or alternate approach was admitted.

The accepted paths retain positive table gaps above the required 10 mm and
positive elbow/torso separation throughout the independent sample audit. The
existing robot collision contract does not impose a new 10 mm torso gap. An
extra diagnostic stricter 10 mm elbow/torso wrapper rejected the midline scene;
that exploratory report is preserved separately and is not the production
contract or acceptance requirement.

These inputs replay the captured fit, accepted calibration and terminal measured
feedback, with base/IMU replaced by the recorded exposure calibration and any
recorded staging start. Receipt timestamps are rebased only offline. Exact native
feedback at the planning instant and independent calibration metrology are
missing. The first two captures also lack their exact planning RGB-D frame in the
local pack, so this proves planning acceptance on their recorded geometry, not
an image-to-grasp or live mechanical success. `physical_lift_verified=false`
and `actuation=false` remain explicit in every result. No live pickup is claimed.
