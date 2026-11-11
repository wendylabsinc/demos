# October 8 offline simulation lab

The new `today` profile joins recorded failure replays with contact simulation
using the deployed color/depth tracking path. No robot calls or deployment were
made. See `today-suite.json` for every outcome and the individual scenario JSONs
for simulated events, command counts and source hashes.

The external-camera color/depth scene completes its simulated lift and hold
without YOLO. Injected missing-target and 600 ms-old alignment observations stop
before any alignment command; one measured-position hold is issued on failure.

The approximate head-camera scene is deliberately left as a failing positive
pickup test. It stops at lift verification with `Color/depth can target
absent/ambiguous`, even with YOLO disabled. The same frame reproduces the error
through single-frame color/depth reacquisition. This establishes an offline
reproducer of the symptom, not proof of the real camera failure's cause.

`head-camera-failure.png` is the synthetic camera image at the stop.
`frame-provenance.json` identifies the archived NPZ by SHA-256. Replay it with:

```sh
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. python offline/replay_sim_frame.py fixtures/sim-head-color-depth-20261008.npz
```

The repro currently exits 1. Repairing the condition should make it exit 0;
the full head-camera pickup test must then also pass. The replay pins the old
exposure timestamp and does not reconstruct prior tracker history.

Regenerate the interactive viewer and sampled animations with:

```sh
python offline/run.py --profile today --report offline/reports/today/latest.json
```

Open `offline/reports/today/index.html`. GIFs are generated locally, not stored
in Git. Original real-world recordings and their immutable hashes are untouched.
The simulator uses a supported base, an untextured cylinder, approximate camera
geometry and modeled contacts/feedback; live transport latency and physical
success are not validated by these runs.

The final lab result is 100/101 passing, with the head-camera pickup left red.
The initial report also contained a test-harness budget mismatch: the historical
400 ms stop test inherited the lab's 500 ms budget. It now pins 400 ms explicitly;
its focused rerun passes under the 500 ms lab profile. Both the original report
and targeted rerun are preserved, with rerun metadata in `today-suite.json`.
Three report tests, two hardware-isolation tests and the trajectory-parity test
also pass.

Trajectory comparison (`trajectory-parity.json`): 1,647 recorded command samples
across eight segments reproduce exactly through the actual interpolation
function. Current repeated IK endpoints are identical to themselves; differences
from the two historical endpoints are 0.0000421 and 0.0009585 rad. This is not
proof of full independent replanning parity: complete planning-time state and
source-version equivalence are not established.
