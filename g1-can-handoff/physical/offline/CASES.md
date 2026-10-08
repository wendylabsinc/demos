# Historical failures and repeatable tests

Every path below is relative to `physical/`. Run a listed module with
`python offline/run.py --test MODULE`. All execute without a robot.

| Failure / behavior | Repeatable reproduction | State |
| --- | --- | --- |
| Coke missed on NX2 / rotated can; empty-scene false positives | `--profile model`: actual checkpoint, 4 positive + 6 negative annotated images; original role/hash kept | Ten saved images pass on CPU; generalization still limited |
| Red object mistaken for Coke; non-red Coke label discarded | `test_yolo_coke`, `test_replay_perception`: semantic box→measured RGB-D geometry, absent identity rejection | Covered without color fallback |
| Wrong frame/image/model result, malformed class or ambiguous boxes, strict >10% boundary | `test_yolo_coke`, `test_target_tracker`, `test_frame_age_limit` | Covered |
| Partial arc fit flips to wrong cylinder center and >6 mm residual | `test_cylinder_fit`: original measured points, old mean-seeded solver fails, repaired fit <3 mm | Fixed regression; old behavior demonstrated |
| Tape strip gives low residual but wrong broad table plane / biased pickup-area coordinates | `test_table_plane`: captured frames 20903/38920/41836, disjoint held-out tabletop points and narrow/disagreeing surfaces | Fixed geometric regression; no independent metrology claim |
| Table appears to move as pelvis pitches during staging | `test_recorded_failures`: 153 measured frames, old transform >5 mm drift, exposure-pose correction <1.5 mm; `test_pose_alignment`, `test_attitude_plant` | Corrected rotation replay passes; translation remains open |
| Old/missing exposure pose, heading wrap, unmodeled waist motion | `test_pose_alignment`, `test_scene_refresh` | Covered |
| Visual tracking becomes uncertain, merges/clips, loses support; cylinder reacquisition fails | `test_target_tracker`, `test_replay_perception`, `test_geometry_parallel` | Same-frame YOLO reacquisition required; failed reacquisition blocks |
| GIL / worker stall / camera prefetch and archive contention | `test_perception_worker`, `test_perception_timing`, `test_observation_archive`, `test_geometry_parallel` | Spawn IPC and controlled concurrency covered; actual Jetson headroom remains open |
| Fresh phase observation fails to refresh watchdog | `test_perception_timing`: actual worker IPC + phase/background callers + controlled clock; old `a217a77` stops at 460 ms | Fixed, demonstrated red on historical runtime |
| Genuine freshness stop at 411.8 ms, trial 00:41:01 | `test_recorded_failures`: exact frame 45594 decoded, recorded gap injected into real command guard, arm/finger writes stay zero | Guard passes; latency unresolved |
| Invalid/stale/future timestamp, >12 ms skew, frame/capture-time mismatch, changed rays/dimensions | `test_camera_pipeline`, `test_replay_perception` with encoded fake HTTP JPEG/PNG+NPZ bundles | Covered |
| Wrong compact camera serial, inconsistent/zero/negative/NaN depth scale, NaN color exposure | `test_camera_pipeline` | Three production defects found by this pack and fixed; old source demonstrably fails |
| Real out-of-area 50.2 cm, false tape-calibration 25.7 cm; confusing error text | `test_pickup_workspace`, original attempts 00:34:03/00:37:18, table-plane replay | Bounds/message covered; no relaxed workspace |
| Startup handover shifts perceived can XY >10 mm, trial 00:40:12 | `test_recorded_failures`: .38627→.39629 m measured shift rejected; `--profile open`: final measured plan must follow cold ownership | **Open**, acceptance test intentionally red; untested proposal saved separately |
| Evidence logging holds state lock / stdout stalls | `test_runtime_lifecycle`: locked state + independent event persistence; no per-frame stdout | Covered |
| Primary error overwritten by cancellation | Original 00:12:46 raw fault 478.3 ms vs cancelled final status; `test_runtime_lifecycle` latches first fault | Fixed regression, historical discrepancy retained |
| Release unavailable, accidental release during active trial, double demo start, cancellation race | `test_operator_panel`, `test_runtime_lifecycle`, `test_dex3_packaging` with fake HTTP/DDS transport | Endpoint/lifecycle/finger mapping covered; physical release unverified |
| Stale body/hand feedback, nonfinite samples, faults, ≥75°C | `test_physical_core`, command-stop cases in `test_sim_runtime` | Required rejection covered; real cooling cannot be simulated |
| Starting posture shifts, joint limits, pregrasp/arm/table collisions, changed target | `test_pregrasp_collision`, `test_staging`, `test_cartesian_lift`, `test_pickup`, `test_model_calibration` | Guard and planner coverage; full simulated pickup has five failures |
| Fingers report unsupported obstruction / lift evidence not fresh or high enough | `test_pickup`, `test_camera_input`, `test_target_tracker` | Covered; no successful physical lift archived |
| Missing recorder / prior ownership / lost start response / unhealthy evidence / cancelled retry | `test_record_demo_attempt`, `test_observation_archive`, `test_runtime_lifecycle` | Covered |
| Grasp orientation search exceeds its 30 s budget | `test_planning_deadline`: recorded failure plus virtual-clock expensive IK candidate; no writes | Timeout rejection covered; finding a valid fast path remains open |
| Offline pack accidentally uses robot/network or corrupt fixtures | `test_offline_pack` | Captured hash corruption rejected; network/DDS denied in child |

## Open simulation scenarios

`python offline/run.py --profile simulation` retains the original 13 tests,
including five failures verified on macOS and Linux/MuJoCo 3.14.0:

1. `test_shared_physical_sequence_lifts_with_measured_table_height`: measured
   starting arm posture changes after planning.
2. `test_75cm_scene_tracks_finger_split_label_during_hold`: same posture failure.
3. `test_head_camera_tracks_finger_split_label_during_hold`: same posture failure.
4. `test_measured_arms_down_stages_then_lifts_and_holds`: can/table moves during
   final reach.
5. `test_measured_arms_down_start_rejects_interpolated_collision`: rejects earlier
   with no reachable path instead of reaching the expected collision check.

No test was skipped or weakened to turn these failures green. The first four
are currently unsuccessful simulated pickup scenarios. The fifth is an error
classification/coverage issue: the scenario still sends no commands.
