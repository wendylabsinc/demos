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
After the measured gentle grasp, the last can estimate is retained, its age
advances, and no new camera/depth observation is requested. ACT stops after
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
