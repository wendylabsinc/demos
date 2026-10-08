# Phase-boundary perception

The default controller policy is now `phase_boundary`: fresh, exact-frame
perception before opening, staging, approach, alignment, reach, closing and
lifting, plus the existing five lift/hold verification observations. Initial
YOLO identity and uncertain-track reacquisition remain; elapsed time never
requests another Coke classification. The 10% threshold, 500 ms deployment
budget, geometry bounds and robot command checks are unchanged.

The first tracker and pickup runs failed (`red-tracker.json`, `red-pickup.json`)
because periodic rechecks could not be disabled and opening/staging/lift lacked
immediate checkpoints. Focused green reports exercise the new policy, all
movement boundaries, failed-boundary rejection, actual Runtime.run entry and
the spawned worker's policy selection.

The first full run also caught four simulated pickup failures at the new
pre-lift check. It had incorrectly retained the pregrasp 10 mm/table contract
after finger closure. The diagnostic measured a 29.4 mm XY shift during closing.
The corrected pre-lift check uses the existing postgrasp 40 mm XY contract and
does not require table contact. It still requires a fresh, geometrically valid
observation; a new negative test rejects 41 mm. The original full-run failures
and diagnostic values remain in `before-prelift-fix-full-tests.json` and
`prelift-diagnosis.json`.

`test_perception_policy` verifies the immutable original event archive and
injects frame 97662's captured `Coke YOLO target absent/ambiguous` error at the
production command check. Continuous mode rejects it; phase-boundary mode has
no background producer or image guard to interrupt that movement. Cancellation
and stale body feedback still reject commands. This is a scheduling replay,
not a replay of all native pixels or proof that the next boundary will have a
valid target. A separate tracker test supplies a stable measured visual/depth
track, advances semantic age by 90 seconds, and proves no timed YOLO call occurs;
uncertain visual evidence still requires successful reacquisition.

The historical tracking-position replay explicitly selects `continuous` so
its immutable historical asynchronous failure remains reproducible. Deployment
verification checks the source hashes, the configured policy, and disabled
periodic rechecks before reporting a passive startup. No physical motion is
part of these tests.

Rollback: deploy the previous controller commit `af58078`, or set
`HANDOFF_PERCEPTION_POLICY=continuous` and restart after supported release.
The model and inference service need no change for this policy update.

Final clean Git-index export: **331 tests pass**, including all 13 simulation tests, with all 140 original evidence hashes verified. The earlier focused index reports precede the pre-lift correction; `clean-index-full-tests.json` is the final combined result. No robot motion was used for these tests.

NX2 deployment completed with controller `0.3.14-phase-perception`. The saved passive readiness result verifies all 49 source hashes, policy `phase_boundary`, no periodic YOLO recheck, healthy paired camera/inference and recorder, PASSIVE state, no arm ownership and zero startup commands. Advancing camera images were 47.3 and 57.3 ms old. `deployment-timing.json` records native build/deploy/readiness; no physical demo was started.
