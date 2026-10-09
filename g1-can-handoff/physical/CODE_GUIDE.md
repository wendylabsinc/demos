# Reading and editing the Coke-can demo

Start with `pickup.py` for the pickup sequence, then `runtime.py` for how a
button press runs it. `physical_core.py` connects that sequence to the real
robot. `g1_can_handoff/sim.py` supplies the shared MuJoCo model and IK solver;
the simulator is also used for planning without publishing robot commands.

The `physical/` folder is the deployment source. Editing an archived copy under
`offline/validation/` does not change the robot. The package at the demo root is
the original standalone simulation; physical deployment uses the package inside
`physical/g1_can_handoff/`.

## From Start to pickup

1. The local controls page calls `operator_panel.py`'s `/start` endpoint.
   Its `Wendy.request()` sends a fixed request to the NX2 controller through
   Wendy's native transport. Loading the page does not start motion.
2. `runtime.py`: `Runtime.start_demo()` starts one background attempt.
   `Runtime.run()` checks readiness, acquires or reuses arm ownership, waits for
   control to settle, and captures the measured planning state.
3. `pickup.py`: `prepare()` observes the can and asks `task.plan()` to select
   a grasp. `PhysicalPlant.preflight()` validates the complete arm/finger path.
4. `pickup.py`: `execute()` opens the hand, stages the arm, approaches, aligns,
   reaches, closes, verifies contact, lifts, and verifies the lift in five new
   camera observations. Phase-boundary perception checks precede movement.
5. Completion holds the can for the operator. Stop freezes the measured pose;
   Release returns ownership to Unitree after the operator confirms support.

## Where to look

| Topic | Main file | Useful entry points |
| --- | --- | --- |
| Real trial, ownership, cancellation, watchdog | `runtime.py` | `run`, `hold_check`, `watchdog`, `stop`, `release` |
| Pickup order and outcome checks | `pickup.py` | `prepare`, `execute`, `checkpoint` |
| Joint commands and measured feedback checks | `physical_core.py` | `_execute_poses`, `command_arm`, `_enforce_arm_tracking`, `check` |
| Grasp candidates and target waypoints | `g1_can_handoff/task.py` | `plan` |
| IK residuals, seeds, joint bounds, collisions | `g1_can_handoff/sim.py` | `Plant._solve`, `collision_details` |
| Collision-checked staging route | `staging.py` | `joint_path` |
| Tuned planner settings | `planner_tuning.py`, `g1_can_handoff/noise_aware_planner.json` | `configure`, `grasp_noise_score` |
| Exact-frame RGB-D acquisition and geometry | `camera_input.py` | `Camera.frame`, `world_points`, `geometry` |
| Color/depth or YOLO identity and tracking | `target_tracker.py` | `select`, `_color_depth`, `_semantic`, `accept` |
| Isolated perception process | `perception_worker.py` | `PerceptionWorker.observe`, `worker_main` |
| Camera owner | `camera_owner/src/coke_detector/camera_service.py` | camera capture and frame endpoints |
| GPU Coke detector | `yolo/service.py` | detector service and inference |
| Camera/identity policy | `perception_policy.py`, `yolo/frame_limits.py`, `yolo/confidence.py` | environment parsing |
| Planning-time state and reproducible replay | `planning_snapshot.py`, `offline/` | snapshot capture, saved-case replay |
| Simulated feedback and perception noise | `sim_runtime.py`, `simulation_position_noise.py`, `stochastic_position_noise.py` | simulated adapters and noise |
| Controls-page layout and browser behavior | `operator_ui.html` | HTML, CSS, JavaScript |
| Controls transport and request validation | `operator_panel.py` | `Wendy`, `Handler` |
| Human-readable state/errors | `operator_state.py` | `ERROR_CASES`, `describe_state` |
| Deployment | `g1_can_handoff/redeploy.py`, `build.stagefile.yaml`, `wendy.json` | local source, copied files, launch environment |
| Read-only deployed-source/startup verification | `verify_passive_deployment.py` | source hashes and passive readiness |
| Complete passive recordings | `../recorder/recorder.py` | recording, disk floor, recorder health |

## Understanding a stopped attempt

`Runtime.record_failure()` retains the first failure and its source. A later
watchdog error does not replace that primary cause. `Runtime.event()` appends
timestamped events to `/data/events-<UTC timestamp>.jsonl` on the controller;
the status response identifies that exact path. Camera archives and planning
snapshots preserve the inputs. The independent recorder retains raw telemetry
in its own persistent volume.

Common error paths:

| Error | Where it is raised or checked | Recorded evidence |
| --- | --- | --- |
| Color/depth target absent/ambiguous | `TargetTracker._color_depth` | exact camera frame and perception failure |
| Exposure stale | camera/result freshness checks | frame ID, exposure timestamp, timing fields |
| Passive recorder health gate failed | `Runtime.archive_guard` | recorder health/session and primary failure |
| Can outside pickup area | `task.plan` | measured forward/left position in the error |
| Measured arm tracking error | `PhysicalPlant._enforce_arm_tracking` | target/feedback, then primary failure |
| Physical arm did not settle | `PhysicalPlant._execute_poses` | timeout event/raise site and joint errors; currently commented out in the saved local version |
| Measured FK target tolerance | `PhysicalPlant.move` | measured grasp-site position versus target |
| Body/hand feedback invalid, stale, hot, or faulted | `snapshot_guard` | measured telemetry and primary failure |

The latest color/depth failure is still reproducible: the initial detector
requires exactly one qualifying red component. A can's rim and body can become
two components. This readability cleanup does not change that rule.

## Settings and units

Launch environment values live in `wendy.json`. Documented ranges and defaults
are in `README.md`; parser functions enforce them. Joint positions/errors use
radians, velocities use radians per second, distances use metres, and camera
age is reported in milliseconds. Preserve the distinction between raw measured
table height and the planner's height offset.

Tracking and settling limits are class attributes on `PhysicalPlant`.
`simulation_tracking_limits.py` supplies **simulation-only** multipliers; it
does not raise the physical limits. Changing a launch environment requires a
controller restart unless the specific setting has a documented runtime endpoint.

## Keeping edits readable and checking them

Install the formatter in your development Python environment, then run it from
the demo root:

```sh
python3 -m pip install black==26.10.0
python3 -m black --check .
```

Use `python3 -m black .` to format active Python. The configuration excludes
archived evidence, validation snapshots, robot assets and native SDK files.
The controls template is formatted with Prettier; keep its JavaScript functions
expanded and its DOM IDs stable because the browser tests exercise them.

```sh
npx --yes prettier@3.9.9 --write physical/operator_ui.html
```

Run offline tests with the project's Python environment:

```sh
cd physical
HANDOFF_DATA=/tmp/g1-handoff-offline \
  python3 -m unittest discover -s . -p 'test_*.py'
```

These tests do not establish physical pickup success. The saved failure cases
remain under `offline/`. Read `offline/CASES.md` and `offline/README.md` for
replay commands and the evidence each case can establish.

Redeploy from the demo root with `python3 redeploy.py`. It deploys this local
checkout through the workspace timing wrapper, checks source hashes and passive
readiness, and never starts a trial or releases owned joints.
