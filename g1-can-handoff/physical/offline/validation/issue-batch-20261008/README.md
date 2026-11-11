# Historical failure acceptance batch

The repair/acceptance batch used read-only robot references and passive deployment.
`reference-before.json` verifies PASSIVE, no adapter and no active demo, with
advancing decoded RGB-D frames. After passive verification the operator explicitly
requested one demo; its separately recorded outcome is under
`offline/evidence/issue-batch-demo-20261008`. Offline acceptance, deployed readiness
and actual physical outcome remain distinct.

The ranked corpus contains 23 first technical failures: nine stale exposures,
six planning failures, three pickup-area failures, two table-height failures,
two control-settling failures, and one pregrasp position-shift failure. Seven
deployment failures are counted separately. One operator cancellation is excluded.
This is the preserved corpus, not a lifetime count of every problem.

| Category | Repeated acceptance | Evidence boundary |
|---|---|---|
| Stale exposure | 9/9 timelines, ten repeats, at 500 ms with receipt-first publication | Counterfactual publication replay through the real motor gate; fresh synthetic body feedback isolates perception. Worst age 460.51 ms; native new latency unmeasured. |
| Planning | 6/6 scenes, three real-clock repeats | Four reconstructed measured snapshots; two explicitly scene-only fallback cases. Actual full-path validation and independent distances, no physical execution. |
| Pickup area | Two placements accepted three times | Bounds 25.5–45 cm forward and 6.5–29 cm left. Full IK, fingers and collision checks retained. The 50.245 cm case remains rejected after three bounded diagnostic searches found no checked complete pickup path. |
| Table geometry | Both original fault RGB-D frames accepted five times, including all tracker frames since their semantic seed | Exact original RGB-D, exposure-matched native poses; residuals −1.970/+0.392 mm, unchanged 5 mm limit. Earlier tracker history and physical execution unavailable. |
| Control settling | Both native feedback traces pass three times at strict <1 rad/s; 678 negative controls reject | Real production loop with recorder-selected body and both hands, including raw hand-device fault fields. Original controller sample schedule and physical settling unavailable. |
| Tracking position | Historical failure reproduces three times; owned replanning reaches an inert checked command three times; later 25 mm movement still blocks | Actual Runtime.run and full checked paths with counterfactual settle and terminal scene held stationary. Original post-settle inputs unavailable. Reference is copied/immutable and invalid centers reject; 10/40 mm bounds unchanged. |
| Deployment | Native build, transfer, source verification and ten GPU recognition fixtures pass | Hash-locked offline dependency overlay preserves pinned CUDA torch. Controller 0.3.13 and YOLO 0.2.3 verified at 500 ms; controller PASSIVE/no ownership/zero startup commands before the separately authorized demo. |

Dedicated directories contain red reproductions, accepted repeat reports, input
hashes, negative regressions and explicit missing-input fields. Rejection of a
genuinely invalid placement is not counted as an accepted historical placement.

From `physical/`, use the project environment:

```sh
python offline/run.py --profile all --report offline/validation/issue-batch-20261008/full-tests.json
python -m offline.stale_replay --max-age-ms 500 --publication receipt --repeats 10
python -m offline.planning_failures_replay --repeat 3
python -m offline.pickup_area_replay --repeat 3
python -m offline.table_geometry_replay --repeat 5
python -m offline.tracking_position_replay --repeat 3
python -m offline.control_settling_replay --repeats 3
```

The isolated runner verifies the original 140 captured evidence hashes before
loading tests and blocks hardware/network access in child processes. Local tests
for `pickup_area_viewer.py` are user-owned and are not part of the PR.

Final clean Git-index export: **320 tests pass**, including **13 simulated
demonstrations**, with all original evidence hashes verified. The export excludes
the user-owned viewer files and contains the newly captured ZIPs and settling
feedback, so this run does not depend on untracked local fixtures.

`controller-readiness.json` verifies all 48 deployed source hashes; inference
verification checks six hashes and the pinned checkpoint. A 32-sample passive
profile has 32 successful observations, median exposure age 217.65 ms,
95th percentile 253.17 ms and maximum 255.64 ms. Its before/after command counters
are both zero. This static-scene profile does not establish timing under motion.

The separately authorized live trial reached approach and alignment, then froze
on `Coke YOLO target absent/ambiguous`. Its artifact SHA-256 is
`53c0352b18d33ebf2499a4016855083301426e9373cd765404988634ebb095b1`;
2,488 events and five RGB-D frames are verified. Frame 97662 has zero detections
returned above 10%, while its exposure is only 196.88 ms old. The last fresh
YOLO score was 10.4858% approximately one second earlier. Exact subthreshold
confidence was not logged; it must not be represented as a measured 0%.
The displayed JPEG decodes to the exact failed inference input hash.
`error-confidence/` retains the native time series, plot and frame provenance.
This new moving-scene recognition failure is unresolved. No grasp/lift success
or automatic second attempt is claimed; operator release/reset was requested.
