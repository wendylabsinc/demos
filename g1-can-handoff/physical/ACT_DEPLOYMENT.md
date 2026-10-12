# ACT residual on NX2

This isolated checkout starts the physical controller released. The controls
page's **Start ACT demo** explicitly requests one trial; deployment starts none.

The selected simulation checkpoint is `step-02000.pt`, SHA-256
`a92445135a9360499d2c9aa6f1123da9335c3d8b5bb2aa727d81b058adcd208f`.
It achieved 125/128 successes in the fixed-reference ManiSkill evaluation.
The NumPy implementation includes the exact inference decoder and temporal
ensemble. Its parity fixtures contain 1,000 recorded PyTorch inputs/outputs.
NX2 maximum chunk-action difference is 5.7e-7; median inference 1.49 ms,
95th percentile 1.56 ms. No Torch installation or GPU inference is required.

The per-inference deadline is 40 ms (previously 20 ms); ACT still updates at
10 Hz. `HANDOFF_ACT_INFERENCE_BUDGET_MS` configures this deadline in milliseconds,
default 40, valid range greater than zero through 100. An invalid value or an
overrun fails before the corrected target is admitted. The policy's correction,
motor speed, feedback freshness and placement limits are unchanged.

Camera/pose pairing now permits 40 ms skew (previously 20), consistently in
the pose history, perception worker and scene refresh. Configure with
`HANDOFF_POSE_ALIGNMENT_MAX_SKEW_MS`, default 40, valid range (0, 100] ms.
The ACT inference deadline stays at 40 ms. Pose failures now report the measured
skew and both timestamps; accepted pose evidence records its skew and limit.

The optimized NumPy decoder caches its observation-independent first
self-attention/norm sublayer and temporal coefficients. Five alternating
1,000-observation rounds on NX2 measured median 1.520 -> 1.243 ms and p95
1.622 -> 1.336 ms. All 1,000 actions matched the previous implementation
bit for bit; maximum PyTorch chunk difference stayed 5.66e-7. This was a
read-only CPU benchmark with no hardware commands, not trial latency evidence.

ACT runs at 10 Hz from 94 measured/model-state features. Its five outputs
correct XYZ by at most 1 cm and rotation about world Y/Z by at most 3 degrees.
The same damped IK Jacobian maps them to left-arm corrections capped at 0.1 rad.
The existing SDK motor-clock reference remains the sole arm filter: 0.25 rad/s,
1 rad/s2, measured-position leash and commissioned joint limits. Nominal online
IK, contact geometry, stop, feedback/thermal checks and the gentle finger grasp
remain in the physical controller. The policy never commands fingers or feet.

This is a hardware adapter around the existing online IK path. Simulation
training used a fixed reference path; the physical adapter and perception are
new integration surfaces. The 97.7% simulation result is not a measured robot
success rate. First physical execution remains to be tested by the operator.

At the operator's explicit request, ACT's training-region XY and height
rejections are removed. The green guide and planner use the original IK area:
can center **25.5–45 cm forward and 6.5–29 cm robot-left**. Accepted calibration,
fresh observations, measured table/can geometry, full-path IK/collision checks,
joint limits and bounded residual corrections still apply. The checkpoint was
trained only around **28.8–30.8 cm forward, 15.5–17.5 cm robot-left, 82.3 cm
center height**; removing admission restrictions does not establish wider ACT
training coverage or physical success. The separate wider-table training work
is unchanged. Runtime reports `act_training_placement_enforced=false`.

Before grasp admission ACT requires fresh continuous color/depth tracking.
This adds a continuous pregrasp input requirement compared with the original
phase-boundary IK demo. No periodic YOLO identity classification is added.
After the measured gentle grasp, the last can estimate is retained and its age
advances. Independent static-table pose frames continue through lift/hold when
`HANDOFF_SCENE_REFERENCE=1`; they do not estimate the held can or claim visual
lift verification. ACT stops after
the bounded lift/hold; the existing controller retains the final gentle hold.

Deploy this copy with `python3 g1_can_handoff/redeploy.py`. The script preserves
the Wendy timing wrapper and verifies PASSIVE startup and the actual build.

## Rollback

The original deployed source digest is
`69e62188a2f4684701d2c38076f66fb13f7721496d3117f6c0b2b373de7988a3`.
GitHub tag: `g1-first-working-recorded-v0.3.29`, commit `e3119fae4a55`.
Its detached local copy is `/Users/olivertaylor/Documents/Wendy/demos-g1-ik-rollback`.
An additional robot archive is
`/data/rollback/IK-first-working-v0.3.29-pre-ACT/controller-source.tar.gz`.

After supporting and releasing the arms/hands, run `python3 rollback_to_ik.py`.
It verifies the pinned IK source and uses the same checked deployment path.
Neither deployment nor rollback starts a demo or returns arms home.


## Static scene reference

Version 0.3.34 enables `HANDOFF_SCENE_REFERENCE=1` (boolean 0/1, default 0).
An immutable first planning exposure anchors independent tabletop landmarks;
pixels within 90 mm of the can are excluded. Advancing exact exposures retain
camera serial, stream and calibration identity. At least 30 correspondences,
60% inliers, broad two-dimensional support and bounded residuals are required.
The fitted proper rigid correction is bounded to 0.04 rad rotation and 20 mm
translation. It transforms BOTH camera and pelvis coordinates after the same
IMU correction; every arm command still rechecks current collision geometry.
Missing/uncertain references and exposure age beyond the existing 500 ms gate
fail closed. Tracking starts during planning to keep the FK reference fresh.
Post-grasp pose-only observations continue without can fitting. Native trial
exports include the immutable anchor image, correction metrics and source.

Local validation: 83 affected tests pass. Full runtime replay of five exact
recorded exposures reduces the failing frame's apparent XY drift from 10.037 mm
to 2.312 mm. Shared camera/FK, invalid/stale pose rejection and 25 mm independent
can displacement guard regressions pass. This is not physical pickup proof.


Version 0.3.35 uses ORB/Hamming tabletop features to reduce registration cost.
All rigid-fit, identity, spatial support, transform and freshness bounds remain
unchanged. Three native motion-zero readiness observations finished at 391.5,
336.5 and 379.4 ms exposure age, all below 500 ms. Exact saved-frame runtime replay
reduces the failed estimate to 1.864 mm. These are readiness/replay measurements,
not a physical pickup outcome.


Version 0.3.36 pins the latest accepted tracking sample before initial path
planning. The 0.3.35 trial's first observation aged to 568 ms while synchronous
tracking startup obtained a second sample (301 ms old at plan admission). The
planner now receives that fresh accepted sample and fixes its path baseline
before any path is checked. It never relabels timestamps or rebases an already
checked path. The 500 ms planning/freshness guards remain unchanged; 85 affected
tests pass, including the recorded handoff and stale/future/error rejections.


Version 0.3.37 projects measured depth only at candidate table feature pixels;
it avoids a redundant dense 480x640 world cloud on the scene reference path.
The can still receives its full independent geometric fit. Table support,
target exclusion, robust fit, pose/identity/freshness and collision bounds are
unchanged. Local runtime replay changes median steady-frame processing from
approximately 45 ms to 30 ms; the recorded false-shift frame remains within 2.43 mm
of its checked center. 86 affected tests pass. A native passive candidate check
accepted three observations at 345.0, 368.3 and 405.7 ms without motion commands.
