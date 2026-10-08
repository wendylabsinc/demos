# Table clearance IK repair

The original Cartesian continuation can reach the requested hand pose while a
finger violates the 10 mm table clearance. The synthetic regression keeps the
preserved trial-03 posture and table geometry and moves the can to 44 cm forward,
18 cm robot-left. Its rejected alignment solution has 8.33 mm endpoint clearance.
The new solution has 10.98 mm minimum clearance over all 43 returned arm samples,
2.36 mm position error, and at most 0.0621 rad local joint change. This is an offline
planning result, not a new physical trial.

Physical IK keeps the original solution when clear. A table proximity violation
triggers signed-distance residuals over the endpoint and three intermediate joint
poses, together with the existing torso objective and weak damping. The target
table gap is the configured clearance plus a 1 mm planning buffer. Every returned
trajectory still passes the existing dense collision checks; intermediate objective
samples do not replace those checks. Endpoint IK also retries a table/torso rejection
with the same commissioned bounds and unchanged pose acceptance tolerances.
Cancellation and measured-state checks remain active during the new optimization.

`segment-red.json` reproduces rejection before the change; `segment-green.json`
records its passing focused test. `segment-proof.json` records independently queried
distances and pose errors. `regression.json` has 262 passing tests, including both
new table regressions, and `simulation.json` has all 13 dynamics tests passing.
The local runtime is Python 3.10.20 with MuJoCo 3.14.0, NumPy 2.2.6, and SciPy
1.15.3. The repository's portable reference remains Python 3.12/Linux.

`placement-sweep.json` records 27 synthetic positions using captured base/arm
posture: three forward positions, three lateral positions, and table heights
65, 76.2, and 95 cm. All 18 cases at 76.2/95 cm pass. Four of nine cases at 65 cm
pass. Two lower-table cases fail continuation/pose preservation; three farthest
cases fail endpoint reach/orientation search. The parent also rejects the two
lower-table edge cases replayed in `baseline-low-edges.json`. These are finite
offline probes, not a qualified workspace map. Wall times overlap other local
validation work and do not establish a robot-speed improvement.

From `g1-can-handoff/physical`, using the offline dependencies:

```sh
python offline/run.py --test test_table_clearance_ik --report /tmp/table-ik-focused.json
python offline/run.py --report /tmp/table-ik-regression.json
python offline/run.py --profile simulation --report /tmp/table-ik-simulation.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m offline.table_clearance_sweep --output /tmp/table-ik-76cm.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m offline.table_clearance_sweep --height .95 --output /tmp/table-ik-95cm.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m offline.table_clearance_sweep --height .65 --output /tmp/table-ik-65cm.json
```

The sweep exits nonzero if any placement is rejected, retaining every case in its
report. The 65 cm sweep currently exits nonzero. Source hashes are recorded in
`source-snapshot.json`. No deployment or physical pickup is qualified by these
results.
