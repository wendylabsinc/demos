# Physical handoff integration

The service starts with passive body/hand readers and HTTP status on port 8102.
It creates no command publisher at startup. It uses the downloaded handoff's IK
model and planner, with the existing arm adapter and a Dex3 adapter that receives live motor temperatures.
The simulated torque loop, simulated contacts, and simulated can position are
never used as physical evidence.

`POST /capture` stores an exact RGB-D NPZ with camera intrinsics, distortion-
corrected rays, exposure timestamps, frame identity and camera serial.
`POST /plan` performs a read-only calibrated plan. `POST /trial` performs one
bounded grasp and 15 cm lift, then holds. These endpoints are loopback-only.
Every trial requires healthy passive recording, fresh fault-free telemetry,
an accepted camera/root transform and recorded table/can dimensions.
The old onboard service must be stopped before the trial; this runtime refuses
a reachable old service or fresh arm SDK traffic. Do not stop the old service
until the read-only plan and operator setup are qualified.

The calibration example intentionally contains no guessed transforms. Supply
independently measured camera/world correspondence points, including at least
two held-out validation points; `fit_camera_calibration.py` fits a candidate and
rejects validation above 10 mm RMS or 15 mm maximum. The world frame must be
robot-forward X, robot-left Y and floor-up Z. Separately measure the pelvis
transform, tabletop height/center/extent, can diameter/height and arm limits.
The candidate remains uncommissioned until those measurements are reviewed.
Persist the resulting configuration at `/data/calibration.json` in the physical
service's named volume. Camera serial and G1 identity must match exactly.

An explicit `operator_accepted_factory_table` mode also accepts the factory
mount position, current IMU/joint posture and the horizontal tabletop depth
plane. This mode records `independent_validation: false`; its plane residual
is not an independently measured camera-position error. The operator must
accept the estimate and confirm the stationary base and factory mount position.
Acceptance expires after 30 minutes. The runtime rejects changed base/waist
posture or attitude, missing source evidence, plane residual above 3 mm, or
tilt correction above 12 degrees. Table geometry extends at least 2 m in each
modeled half-dimension, with a conservative near edge, instead of requiring
physical width and depth. `factory_table_candidate.py` generates a draft and
runs passive planning; it never approves or enables motion automatically.

A trial holds the right arm and hand at their measured initial positions,
because the arm SDK weight affects both arms. The left hand converts the
simulator's thumb/index/middle order into the adapter's thumb/middle/index
order. Arm references are capped at 0.25 rad/s and 0.5 rad/s². Interpolated
poses are collision checked before publication; feedback age, tracking error,
body/waist posture drift, temperature and raw motor faults stop progression.
Finger obstruction is only a contact candidate. Camera-measured vertical
movement independently verifies the lift; stationary raw Dex3 torque values
are not treated as Newton-metres or grasp evidence.

Ownership settling uses `HANDOFF_CONTROL_SETTLE_MAX_SPEED_RAD_S` (rad/s,
default `1`, finite valid range `0 < value <= 1`). Every body joint's absolute
reported velocity must remain strictly below that value over the settling
window. Set `0.03` to restore the earlier, more restrictive limit. The value
is read at each handover, so the process environment must be updated before
the next trial; changing a launch environment requires a controller restart.
The half-second position-stability requirements remain: IMU span <=0.002 rad,
base/leg/waist joint span <=0.005 rad, and arm joint span <=0.015 rad, with a
five-second timeout. This limit changes handover admission, not commanded
motion speed. It does not exempt table collisions or robot self-collisions.

The physical FK model now requires positive table clearance, configured with
`HANDOFF_TABLE_CLEARANCE_M` (metres, default `0.01`, finite range
`0.005..0.03`). Proximity contacts enforce the margin without moving the
measured table plane or exempting penetrations. The planner prefers a grasp
at the measured can centre, testing 10/20 mm higher candidates if needed.
Alignment and final reach use
continuous Cartesian IK from the preceding arm pose, checking every connecting
joint segment with the same collision and velocity limits as execution.
Staging and the ascending lift retain their existing bounded planners.

The current planner checks nonadjacent left-side contacts as well as table/body
contacts. It rejects an already-intersecting measured starting pose before IK.
Staging and approach retain the measured finger shape; the hand opens at the
checked approach pose, which avoids opening beside the thigh. Endpoint IK uses
one 8 mm/0.02 weighted-rotation acceptance contract. Geometric IK rejections may
retry; feedback and operator faults abort. Collision-aware recovery adds table,
torso and detected forbidden robot-contact residuals while the dense path guard
still decides acceptance. The independent audit and remaining low-table failures
are recorded in [IK contract validation](offline/validation/ik-contract-audit-20261008/README.md).

The arm adapter's second `ctrl_dual_arm` argument is feedforward torque.
The physical plant supplies gravity torque from the measured joint posture,
with zero model velocity. Per-arm caps are 6/6/4/4/1.5/1.5/1.5 Nm and the
torque changes by at most 5 Nm/s. PD gains and the 0.04 rad settling limit stay
unchanged. The simulator now consumes that same torque argument and uses the
adapter's 80/3 shoulder/elbow and 40/1.5 wrist gains and reference filter.
Earlier simulation results used implicit gravity compensation and different
arm gains; they did not expose the zero-feedforward hardware failure.
`actual-servo-zero-feedforward.json` reproduces a settling failure;
`actual-servo-bounded-gravity.json` passes lift verification in the same setup.

The FK collision model places the can at its camera-observed position.
Staging, approach and alignment require 10 mm of clearance from the can;
alignment descends 10 cm to the side before the final reach. Only left-finger
contact is allowed during the intentional reach/grasp/lift phases. The former
blanket can-contact exemption could allow a finger to push the can during
alignment. A physical retry showed an approximately 35 mm shift after that
descent; `premature-contact-camera-evidence.json` preserves the observations.

`POST /stop` freezes at measured arm positions; it does not home or open the
hand. A watchdog also freezes on stale feedback, archive failure or a 90-second
trial limit. `POST /release` requires `{"can_supported": true}` and ramps arm
SDK ownership to zero. Shutdown and emergency-stop behavior still require
physical qualification before a pickup; do not call software verification
physical qualification.

After a successful lift the runtime waits for the operator to take and support
the can, then `/release` ends hand/arm holding effort and returns arm ownership
to factory control. It does not automatically drop the can or drive a scripted
home path through the table. A failed trial before the close phase automatically
releases SDK ownership because it has not grasped the can. Failures after close
retain support until the operator confirms the can is supported.

Phase/target/error JSONL stays in `/data`, and each ownership event records the
independent recorder session. Raw complete telemetry stays in the recorder's
separate persistent volume. Controller publications and actual measured motion
remain distinct in both evidence streams.

## Coke-trained YOLO on G1 NX2

The physical detector adapts the Coke-trained YOLOv8n checkpoint from
`wendylabsinc/g1-bartending-demo`, with parent and adapted SHA-256 recorded in
[yolo/model/SOURCE.json](yolo/model/SOURCE.json). It runs in the separate
`DEV-g1-handoff-coke-yolo` Wendy app on JetPack 6.2 / Orin NX. CUDA PyTorch FP16
is required; missing CUDA, an incorrect checkpoint, or failed warm-up prevents
startup. This version uses PyTorch CUDA, not the old app's TensorRT engine.
The detector reuses the older detector's fixed-square letterbox and bounded
one-class CPU NMS around its CUDA model forward. Inference stays on the
same warmed CUDA thread, with a fixed 640-pixel FP16 CUDA graph by default.
Four saved Coke views and two empty regions must pass semantic/box qualification
before readiness. The detector has no
USB/camera/DDS access or command publisher.

The handoff runtime keeps the existing `g1-can-camera` owner and renewable
camera lease. It sends **the exact decoded BGR image paired with its depth** to
loopback `POST /detect` on port 8112. The response must match camera serial,
frame ID, exposure time, dimensions, image checksum, model checksum and CUDA
backend. The 400 ms exposure-age limit is enforced before and after inference
and after geometric fitting. It never consumes a latest-box feed or silently
falls back to red pixels when YOLO is unavailable.

Timing and RGB-D payloads are fetched concurrently for the same frame ID.
The physical runtime reserves a 180 ms acquisition budget, trying at most three
complete acquisitions before rejecting the observation. Inference overlaps the
ray transform; normalized NPZ generation and evidence writes overlap fitting.
The archive worker must finish successfully before any observation returns,
and the final 400 ms freshness check still applies after that join. There is
one archive worker and no pending evidence backlog between observations.

`HANDOFF_CAMERA_MAX_AGE_MS` defaults to 400 milliseconds and accepts only finite
values from 50 through 400. The physical controller exposes a loopback-only
`POST /perception-config` with `{"camera_max_age_ms":400}` for future tuning
without rebuilding or restarting an owned arm. It requires no active operation,
persists the accepted value in `/data/perception-config.json`, and never resumes
motion. The inference service retains its independent 400 ms hard ceiling.

On the operator Mac, `python3 physical/operator_panel.py` opens a local control
server at `http://127.0.0.1:8197/`. Its enabled Release button returns this demo's
upper-body override to Unitree and relaxes both Dex3 hands via native Wendy MCP.
Pressing it confirms that the can and raised arms are supported. Loading the page
only reads status; it cannot issue a release or start a pickup. The color image
refreshes directly from the camera without the legacy aiming guide or detector
dashboard. The server binds only to loopback with a per-run operator token.

Start demo issues one asynchronous `POST /demo/start` on the controller. It
retains the full planning, confidence, depth, freshness, ownership and recorder
gates. A second start is refused while the job runs, and Stop remains reachable
throughout preparation and motion. Cancellation is latched before the worker
starts, so an early Stop cannot be cleared by the queued demo. Stop holds the
measured pose; Release cancels first and requires the operation to finish before
relaxing both hands and relinquishing the upper-body override. If cancellation
is still pending, it keeps the hold and asks the operator to retry release.

Measured camera/decode/geometry changes and remaining latency limits are recorded
in `PERCEPTION_PERFORMANCE.md`. Inside the physical container,
`python3 /app/benchmark_perception.py --samples 8` runs a bounded passive check
without acquiring a camera lease or creating motor adapters.

A Coke detection strictly above 30% confidence (or a higher configured
threshold) selects a box; exactly 30% is rejected. Measured depth
inside its inset region supplies the cylinder points, with finite-depth,
foreground-depth, upright/table, 6 mm fit, and 6-pixel image-boundary guards.
The cylinder fit uses sidewall points: the upper 10 mm below the measured lid
is excluded before estimating foreground depth. Initial lid height is bounded
by the commissioned can/table height; tracked height follows measured depth.
There must be exactly one qualifying initial target. During lift/hold, only
boxes with measured points in the existing expected-cylinder region can match;
multiple matches remain ambiguous, and the fitted center must stay within
40 mm. Finger occlusion may cause YOLO/depth rejection; color cannot rescue a
missing semantic detection. Repeated observations retain the existing distinct
frame and vertical-rise checks. YOLO confidence does not authorize motion.

| Runtime variable | Default | Units / allowed range |
| --- | --- | --- |
| `HANDOFF_YOLO_URL` | `http://127.0.0.1:8112` | Same-host inference URL; timestamps use the NX2 monotonic clock |
| `HANDOFF_YOLO_MIN_CONFIDENCE` | `0.30` | Probability, `0.30..0.99`; geometry and motion gates remain required |
| `YOLO_MODEL_PATH` (detector app) | `/models/coke-can.pt` | Checksum-pinned checkpoint file |
| `YOLO_CUDA_GRAPH` (detector app) | `1` | `1`: captured FP16 forward; `0`: eager FP16 comparison |

Build/deploy the isolated detector first from `physical/yolo`, then deploy the
physical app from `physical` using native Wendy with an explicit NX2 selector.
On this workspace, operator deployments use
`python3 /Users/olivertaylor/Documents/Wendy/scripts/wendy-run-timed --device
10.10.20.238:50052 --builder docker --dockerfile build.stagefile.yaml --detach
--yes --chunking off`. Each app has its own build context; a detector change cannot replace
the physical controller. No existing camera service is deployed by either
project. Check `http://127.0.0.1:8112/health` **on the NX2** for the checkpoint,
CUDA backend and GPU; TCP readiness is available only after model warm-up.
The physical app still starts PASSIVE and requires the existing commissioned
calibration, recorder, ownership, telemetry and explicit trial gates.

From the repository root, using the project's Python environment:

```sh
PYTHONPATH=g1-can-handoff/physical python -m unittest discover -s g1-can-handoff/physical -p test_yolo_coke.py -v
python g1-can-handoff/physical/yolo/replay.py
```

The offline replay requires `ultralytics==8.4.123`. It intentionally runs on CPU
and requires one Coke detection at confidence >0.30 with IoU >=0.50 against
the archived real-camera annotation. CPU replay is not the physical service's
backend or a hardware qualification. Model/runtime licensing and source
provenance are recorded in the manifest.

### Model repair and measured timing, 7 October 2026

The parent checkpoint recognized the archived can but scored the initial NX2
view at only 12.7%. The repair changes only `model.22.cv3.*` (the class branch);
all feature and box-regression state remains exactly equal to the parent.
Training adds one manually annotated NX2 view and two empty table/cable crops
to the original 453 training images. Weighted copies are sampling weights,
not additional independent views. A rotated view and two archived G1 views
remain held out. The unchanged >30% gate reports validation precision/recall
86.8%/77.8% and test precision/recall 97.1%/94.4%; parent values were
87.1%/79.9% and 86.1%/95.8%. Scene generalization remains narrowly qualified.

Use `adapt_model.py --model <parent.pt>` for the first 12-epoch rehearsal
phase. Continue from its last checkpoint with `--epochs 16 --stop-after 5
--positive-weight 120 --fixed-view` and a fresh output directory. Blend those
two last checkpoints using `blend_model.py --adapted-weight .5`, then run
`check_dataset.py` against the original untouched validation/test splits and
`replay.py` against every fixture. The manifest records the recipe; MPS
training is not claimed to be bitwise reproducible.

On the NX2, 16 consecutive live frames produced one Coke box above 30%:
52.4–57.0% confidence, inference median 25.1 ms / p95 26.1 ms. Excluding the
first ray-cache warm-up, sequential camera acquisition plus detection ran
at 9.27 Hz; this is separate from the 15 Hz camera and ~40 FPS model capacity.
A controlled comparison on the parent checkpoint reduced median prediction
time from 27.7 to 20.3 ms with identical boxes using the CUDA graph.

The final passive plan passed. Trial requests still stopped before creating
motor adapters: camera acquisition/final freshness and cylinder residual
checks rejected some frames. The last rejection was a cylinder residual above
6 mm. Motor commands remained zero, and the runtime was returned to PASSIVE.
This is not a verified physical pickup. See the dated `yolo/validation-*.json`
files for exact artifacts, samples, dataset metrics, and simulation failures.

## Shared simulation regression

`runtime.py` and `sim_runtime.py` call the same `pickup.prepare` and
`pickup.execute`. Both use `PhysicalPlant` for measured-state IK, collision
checks, trajectory limits, finger ordering, tracking, obstruction candidates
and cancellation. MuJoCo replaces device commands, telemetry and sleeping.
A separate model handles FK so planning cannot alter the dynamics state.
Rendered RGB-D passes through the same `camera_input.geometry` depth fitter,
with an explicit `synthetic_red=True` mask for the untextured MuJoCo can. The
simulation camera marks these frames `synthetic_rgbd: true`; physical frames
cannot use that path. Simulation does not qualify the new YOLO perception.

Run the pinned Linux regression from the repository root:

```sh
docker build -f physical/Dockerfile.sim -t g1-handoff-shared-sim:test .
docker run --rm g1-handoff-shared-sim:test
```

The image uses Python 3.12, MuJoCo 3.14.0, NumPy 2.2.6, SciPy 1.15.3 and
Pillow 11.3.0. It renders through Mesa EGL. The pinned SciPy macOS wheel fails
to load on this development machine, so Linux is the reproducible path.

The scenarios cover 75cm and operator-measured 30 inch tabletop heights,
external and approximate head views, and the measured arms-down initial
posture. Use `--height .762` and `--initial-arm-q` followed by fourteen
left-then-right joint angles to reproduce the measured-start case. Can XY
remains synthetic at [.35, .18]m, as do table XY and extent. The reported
belly-to-table gap is not a pelvis-frame measurement and does not place these
objects in the model.

Before acquiring control, preflight checks opening, staging, reach, closing
and lift trajectories. If the first direct approach intersects the table,
`staging.py` searches within the configured joint limits using a fixed seed,
a 1,800-node cap and a ten-second deadline. The physical executor rechecks
every staging segment with its normal smoothstep interpolation and tracking
limits. A starting posture change above .04rad requires replanning.
`--disable-staging` preserves the direct-path collision regression, which
aborts with zero arm or hand commands from the measured arms-down pose.

After closing, the sequence requires three consecutive samples with at least
two stationary finger obstruction candidates, allowing up to two additional
seconds to settle. It then requires five distinct camera frames over one
second to retain at least .11m vertical rise and at most .04m center error.
Replayed frames, immediate slips and off-target red objects fail.

In the synthetic simulation mask, during lift and hold only, the detector can
merge red label fragments split
by fingers inside the expected cylinder's candidate region. That search
radius includes the can radius, the existing .04m center tolerance and a
.01m depth/pixel margin. Acceptance still requires a cylinder fit within
.006m and its center within .04m. Initial detection continues to reject
multiple red targets, and the image-boundary crop guard is preserved.

The supported base, torque servo, friction, perfect feedback timestamps and
synthetic camera calibration remain assumptions. The simulated camera FOV
and intrinsics do not reproduce the actual D435i. The observed real 640x480
intrinsics are fx=603.598, fy=603.704, cx=318.059, cy=261.1; the roughly
35.55 degree table-derived tilt is not a commissioned extrinsic transform.
Shared controller success therefore does not establish physical readiness.
Results retain `physical_pickup_verified: false`, dependency versions and
controller source hashes. Simulation can position is diagnostic truth only.

`/stop` cancels planning or running actions. `/release` refuses to relax the
hand until any cancelled operation finishes cleanup. It still requires
confirmed can support. Status preserves per-motor temperatures even when a
guard fails.

## Calibration without printed markers

Yellow tape can make a table edge easier to identify. It does not by itself
locate the table in the robot frame. Keep the robot stationary and use six or
more identifiable points with independently measured positions. Four points
are used to fit the transform and at least two different points are held out
for validation. Spread the points across the reachable tabletop; points on a
rigid box at a second height also help check the vertical alignment.

Use the archived exact-frame RGB-D NPZ from `/capture`. For each point, record
its image pixel `[u, v]`, its measured `[x, y, z]` in metres, and whether it is
held out. Use robot-forward X, robot-left Y, and floor-up Z with the same
measured origin as `pelvis_to_world`. The measurement JSON has `camera_serial`,
`frame_id`, and `points`. Each point has `pixel_uv`, `world_xyz_m`, and
`validation`. Select a point on the measured surface, not a pixel straddling
foreground and background. Missing depth and depth discontinuities are refused.

```sh
python calibration_points.py capture.npz measurements.json candidate.json
```

The helper extracts distortion-corrected camera points and runs the existing
held-out fit. Its output always has `commissioned: false`. Table dimensions,
can dimensions, robot base pose, arm limits, operator setup, and the independent
validation still need review before creating `/data/calibration.json`. Never
fill missing measurements from the simulator or treat the historical head
camera estimate as a verified transform.

The runtime preserves per-motor temperature readings in status even if a guard
fails. The operator-selected guard allows temperatures below 75 C for all body and
hand motors, and refuses values at or above 75 C. This is a commissioning setting,
not a manufacturer-qualified limit. The packaged Dex3 thermal
module receives fresh hand temperatures from the same telemetry snapshots used
by the motion guards, without adding blocking reads to the command loop.

## Offline regression pack

See [offline/README.md](offline/README.md) for recorded hardware failure replays,
184 CPU regressions, optional real-checkpoint tests, and the open handover and
simulation acceptance cases. Run `python offline/run.py` from this directory;
no robot or Wendy connection is required.
