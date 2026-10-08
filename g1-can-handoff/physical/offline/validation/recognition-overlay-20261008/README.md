# Low-YOLO perception and placement visibility — 2026-10-08

The final runtime change uses `HANDOFF_TARGET_IDENTITY=color_depth`: initial
color/depth acquisition, then appearance/depth tracking with phase-boundary
checks. YOLO is not a startup, movement, or reacquisition dependency. This is
red can-shaped object detection, not brand verification. Existing depth,
geometry, freshness, workspace and robot guards remain enabled.

The two archived initial-failure frames 115503 and 115616 pass the new geometry
path without YOLO (cylinder residuals 2.36/2.42 mm). The permanent test exercises
both in sequence, missing-target rejection, stale-exposure rejection and exact
source timestamps. Raw sources and the saved scene are also in the hashed
`offline/evidence/low-yolo-20261008` fixture pack.

The earlier exploratory model repair is preserved as diagnostic reports only.
No checkpoint change was promoted or deployed. The candidate files remain in
`.runtime/recognition-candidate-20261008`; candidate-comparison and blended-
comparison reports are not claims about the deployed model. Their 14-fixture
red model report refers to an experimental qualification manifest, subsequently
reverted. The deployed model remains the original `fab625...01e9` checkpoint.

The placement overlay is retained during demos and transient feedback outages.
Only current valid projections are green. A changed pose or stale projection
becomes an amber dashed reference, never movement authorization.

The old None recheck-interval TypeError is covered at the real geometry call
site. The separate `yolo_once` option performs initial recognition only and
rejects uncertain tracking without issuing another YOLO request. The previous
semantic behavior remains available as `yolo_reacquire`.

Offline tests and passive deployment/probe results are recorded alongside this
file. No demo start or robot movement is authorized by this validation.

Deployment 0.3.15-color-depth-tracker passed passive readiness with 49 source hashes matched and zero motion commands. The five-frame live probe used zero YOLO calls, with completed exposure ages 166–264 ms and cylinder residuals 2.70–3.12 mm. Its first attempt used old stored calibration and failed table agreement; that failure is retained. The corrected probe mirrors the production in-memory scene refresh with exposure-aligned poses; it does not persist calibration or authorize movement. The full offline suite passed 357 tests including 13 simulations and 143 evidence hashes.
