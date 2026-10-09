Gentle grasp implementation: LOCAL ONLY

Prepared version 0.3.28-gentle-grasp. Not deployed; no hardware commands.

56 focused tests passed, including chronological hardware-failure replays, partial-contact hold, moving/empty/repeated-feedback rejection, cancellation, physical hand-loop transition, finger offset logging, arm/fault/feedback guards and packaging.

The thumb reduces to 0.5 Nm nominal proportional effort after a 0.2-second measured stop. First thumb reduction occurs at 1.59 s/30 C in the earlier trace and 1.43 s/36 C in the latest trace. All closing fingers stop and at least two groups show obstruction before completed-grasp admission (2.70 s and 2.66 s). Temperatures are unchanged recorded values, not predicted results of sending the new commands.

Matched MuJoCo scene: same arm P75, finger P2/D0.2, pose and planner. Old closing behavior passes lift verification. New gentle grip does not: can center z 0.91405 m vs 0.92871 m with old closing, about 14.65 mm lower. Camera lift verification correctly rejects the new case. Full pickup with the 0.5 Nm grip is not validated, and this is a physical-validation blocker before claiming success. No thresholds were weakened to hide this result.

Evidence: focused-tests.log, new-grip-replay.json, gentle-physics.json, old-closing-matched-P2-physics.json, physics-comparison.log, implementation-source-sha256.json. Run compare_grip_physics.py --output-dir NEW_DIRECTORY to repeat the offline matched comparison.

Subsequent deployment: 0.3.28-gentle-grasp is now running on NX2, with source checks and PASSIVE/zero-motion startup verified. The native timed deployment's functional readiness failed because the configured RealSense is absent (HTTP503). Latest hand fault feedback is clear. No demo was started, and the simulation lift-verification limitation remains unresolved. See ../gentle-grasp-deploy-20261009/RESULTS.md for source digest and live evidence.
