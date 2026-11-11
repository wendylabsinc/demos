# Identical-input comparison of the latest planner tuning

The baseline before stochastic-model tuning and the tuned candidate both passed
all six historical planning-scene replays. Input hashes match within each pair:
robot/hand feedback, model poses, joint state, recorded can center/table and
calibration/configuration are held fixed. Only the bounded IK/planner settings
change. No new noise is sampled, no command adapter is constructed, and no
physical deployment or motion occurs.

An additional original-source control (`original-planner-control.json`) reproduced
the older recorded planner's collision rejection on the saved trial-3 scene and
verified that the current clearance planner passes it. That check attributes a
historical improvement to the earlier repair, not to the latest noise tuning.

| Recorded original failure | Baseline | Tuned |
| --- | --- | --- |
| Orientation search timeout | Plan passes | Plan passes |
| Owned staging/table collision | Plan passes | Plan passes |
| Owned search timeout | Plan passes | Plan passes |
| Additional owned search timeout | Plan passes | Plan passes |
| Clearance search timeout, trial 1 | Plan passes | Plan passes |
| Clearance search timeout, trial 3 | Plan passes | Plan passes |

The latest tuning therefore did not rescue an additional historical planner
failure in this set. These scenes already passed after the earlier continuous-
path and clearance-replanning changes. The tuned search took roughly 10-14
seconds per scene here versus 2.5-3.8 seconds baseline, with two concurrent
offline workers. All returned plans passed the existing independent clearance
audit and left the planning model joint state unchanged.

These are identical inputs *between the two current policies*, not a claim of
perfect reconstruction of the original physical planning state. Four cases use
captured terminal feedback amended with recorded calibration/base and staging
start values. Two cases lack full captured body/hand feedback and use the
previously documented fallback; one also lacks the exact RGB-D frame and
captured controller source. Each result preserves those limitations and the
original evidence checksum. The comparison validates plan preparation, not full
trajectory execution or pickup.

The larger campaign's three tuned-only versus one baseline-only completions
occurred under fresh stochastic noise. Identical seeds do not give identical
exposures when action timing and viewing differ, so those cannot establish that
the tuned solver fixes the same historical observations. The same fixed stale,
missing-target or over-10-mm pregrasp displacement inputs still face the existing
independent guards; solver tuning does not remove those failures. Initial can-
to-wrist/finger clearance stops were also shared by both policies in the larger
campaign.

`summary.json` records all pair hashes, outcomes and missing fields. Individual
case JSON files retain checked waypoints, rotations, clearance audits and events.
`provenance.json` records the sources used. Reproduce each policy for a selected
case (0-5) from the physical directory:

```sh
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python offline/compare_planner_inputs.py --case 0 --policy legacy --output offline/reports/identical-legacy.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python offline/compare_planner_inputs.py --case 0 --policy tuned --output offline/reports/identical-tuned.json
```
