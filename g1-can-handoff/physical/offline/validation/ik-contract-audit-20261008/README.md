# IK and collision contract audit

The earlier green results were insufficient: the application ignored contacts
between nonadjacent left links, accepted an endpoint using looser tolerances than
its candidate filter, and retried arbitrary ValueErrors from feedback. The new
four-test baseline in red.json fails on e9afa92. The final suite has 284 passing
tests, including six IK contract tests and 14 dynamics tests.

The collision checker now rejects nonadjacent left-side contacts. A hand at the
recorded arms-down posture with zero finger angles starts with 16.65 mm of modeled
thumb/hip penetration. It must abort before commands. The old successful dynamics
fixture assumed that open resting hand was clear. Its positive staging test now
uses the preserved trial-03 folded finger measurements. Separate tests retain the
open, intersecting start and require zero commands with staging both enabled and
disabled. A node-budget test now uses an analytic wall with clear endpoints.

For a clear folded resting hand, preflight and execution stage/approach with the
measured fingers, then open at the checked approach pose. Opening beside the thigh
was the sequence defect exposed by the stricter collision policy. Full hand
opening and closing trajectories remain checked.

Endpoint acceptance consistently requires <=8 mm position error and <=0.02
weighted matrix rotation residual (0.15 times the matrix difference norm).
A typed IKError permits another geometric candidate; stale feedback and other
operational errors abort immediately. Table/torso recovery also considers rejected
left-side self-contact pairs. Cartesian continuation includes detected forbidden
robot contacts in its clearance residuals at the sampled connecting poses, then
validates every returned joint sample with the collision guard. Soft objectives
never grant a collision exemption. The high-table wrist case in wrist-recovery.json
was blocked after enabling the stricter guard and now has a checked complete path.

The independent primitive-distance test computes box/sphere distances from sizes
and center coordinates, including penetration, tabletop clearance and edge/corner
cases. MuJoCo 3.14.0 agrees to 10 decimal places on these cases. This supports the
distance calculation for those primitives; it does not independently validate
Unitree mesh hulls, the camera transform, or real robot/table dimensions. MuJoCo
uses convex hulls for mesh collision. The factory/table calibration remains an
operator-accepted estimate with a conservative table footprint, not measured edges.
No geometry was shrunk and the 10 mm table margin remains in force.

The final synthetic grid uses the preserved trial-03 base/arm posture and can XY
positions 28/35/44 cm forward by 10/18/28 cm left at heights 65/76.2/95 cm.
All 18 cases at 76.2/95 cm pass with the stricter policy. Four of nine at 65 cm
pass. Five low-table placements still exhaust the bounded pose/orientation search;
this is not proof that those positions are physically unreachable. The finite
input bounds are not a qualified workspace map. These are planner probes, not
full-task pickup trials. The dynamics suite separately checks lift/hold and
zero-command rejection. Physical motion, calibration accuracy and pickup remain
unverified. Timings overlap local test runs and are not robot benchmarks.

Local dependencies: Python 3.10.20, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3.
The portable CI reference remains pinned Python 3.12/Linux. RAM scratch storage was
used for test temporaries after the host disk filled; published source and reports
are retained in this repository.

Reproduce from g1-can-handoff/physical:

~~~sh
python offline/run.py --profile all --report /tmp/ik-contract-all.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m offline.table_clearance_sweep --height .762 --output /tmp/table-762.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m offline.table_clearance_sweep --height .95 --output /tmp/table-950.json
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m offline.table_clearance_sweep --height .65 --output /tmp/table-650.json
~~~

The low-table sweep intentionally exits nonzero and retains each rejection.
