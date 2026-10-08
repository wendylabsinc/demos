First supervised trial after clearance-planner deployment

The regular Start endpoint accepted one bounded attempt. Perception passed, but the grasp orientation search exhausted its deadline while rejecting table-clearance and torso/elbow collisions. The trial ended BLOCKED with zero motion commands and no actuator ownership. No pickup occurred. Three attention sounds returned exit code 0. The operator was asked to reset the placement before a second attempt.

The verified archive preserves 37 native events and two exact RGB-D frames, frame timing, calibration/posture in events, and before/after controller snapshots. Archive and member hashes are checked by offline/preserve_demo_batch.py. Raw recorder DDS streams remain on the robot; this is not a full raw telemetry export.

Use supervised_demo_trials.py --verified-release offline/validation/ten-demos-20261008/clearance-controller-readiness.json for this release; the default verifier remains the earlier release for historical trials.

Trial 2: stopped before ownership/motion because the measured can was 36.8 cm forward, 6.8 cm robot-left (required left 9–29 cm). Ten native events and two exact RGB-D frames were exported and checksum verified. Three alert sounds completed. Operator asked to move the can farther robot-left before trial 3.

Trial 3 was already running when the agent observed it. The supervisor issued no new Start (its fresh-frame preflight received camera HTTP 423 Locked). Its eventual planning failure was exported separately: 29 events and two exact RGB-D frames, zero commands/ownership. The initiating request and before-start snapshot were not captured. It reproduced offline with terminal measured posture and accepted scene calibration; the pose-only continuation curled the redundant elbow into the torso. A signed-distance clearance objective and weak null-space damping yield a fully checked path for the exact scene without exempting any collision. Red/green reports are saved under offline/validation/ten-demos-20261008/new-placement-*.json.
