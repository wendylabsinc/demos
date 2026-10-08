# G1 collision error review pack

This branch is stacked on PR #2 (`codex/g1-nx2-yolo-coke-detector`) at commit `9ced867`. It snapshots selected collision-repair files and saved evidence into a separate checkout; it does not change the main demo checkout, deployment version, operator controls, or robot state.

`collision-errors.json` indexes all collision-related events found in the saved compressed evidence: 99 unique native/captured events, with overlapping copies deduplicated by event identity and payload. Categories are 53 table-clearance/intersection entries, 38 elbow-to-torso entries, and eight grasp-search deadlines. Counts are event records, not a count of separate physical attempts. Every source archive has its SHA-256 recorded; entries retain phase, body/geometry names, signed distance, joint target and path fraction when available.

The latest observed attempt is preserved under `offline/evidence/clearance-demos-20261008-preserved/trial-03`: 29 events and two exact RGB-D frames with timing/rays, plus terminal measured posture and source copies. It ended before actuator ownership with zero motion commands. Its initiating request and before-start snapshot were not captured. Raw DDS streams remain on the robot; this pack does not claim a full raw telemetry export.

The exact near-centre scene fails pose-only Cartesian continuation because elbow redundancy curls into the torso during reach/lift. The repair first tries the original continuation; only solutions within 3 mm of the torso use a signed-distance objective and weak null-space damping. Existing local joint bounds, pose tolerances, table clearance, full interpolated-path collision checks, and stop/feedback guards still determine acceptance. Applying the extra objective to every solution caused a dynamics regression; that failed report is retained as `elbow-clearance-simulation-red.json` and motivated the selective fallback.

The red replay and regression test, green replay, hypothesis probe, and historical test reports are under `offline/validation/ten-demos-20261008/new-placement-*.json` and `elbow-clearance-*.json`. `source-snapshot.json` records copied input hashes. Tests rerun for this isolated PR are `regression.json` and `simulation.json`; those are authoritative for this snapshot rather than the imported intermediate reports.

Reproduction, from `g1-can-handoff/physical` after setting up the dependencies described in `offline/README.md`:

```sh
python -m offline.collision_error_catalog --output /tmp/g1-collision-errors.json
python offline/run.py --report /tmp/g1-collision-regression.json
python offline/run.py --profile simulation --report /tmp/g1-collision-simulation.json
python -m offline.planner_replay --trial 3 --root offline/evidence/clearance-demos-20261008-preserved
```

No deployment or physical pickup validation is performed by this PR preparation.
