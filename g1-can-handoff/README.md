# G1 Dex3 can pickup and supervised handoff

A Wendy application that detects an upright can and its tabletop from
calibrated RGB-D images, chooses a reachable left-hand grasp, closes the Dex3,
checks that the can remains held during lifting, and offers it above the table.
The release operation requires explicit confirmation that the recipient supports
the can. The demo stops in `waiting_for_acceptance` with the hand closed.

Start with the [code guide](physical/CODE_GUIDE.md) to find the pickup phases,
planner, camera path, joint checks and error logging. The G1 NX2
[physical runtime](physical/README.md) currently selects color/depth tracking
in `physical/wendy.json`; the trained Coca-Cola YOLOv8n model remains available
through the documented identity modes. The original simulation commands below
retain their synthetic red-cylinder detector.

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from
this directory:

```sh
uv sync --frozen
uv run g1_can_handoff demo --height .85 --x .315 --y .19 --record
uv run g1_can_handoff demo --record --view head --output results/head-perspective
uv run g1_can_handoff evaluate --random-cases 12 --output results/sweep
uv run python -m unittest discover -s tests -v
```

The demo saves `results/demo.json`, `results/camera.png`, and, with `--record`,
`results/pickup.gif`. The evaluation saves a machine-readable `evaluation.json`
after each case. It returns a failure exit code if any requested pickup fails.
`--boundaries` additionally tests lower and farther targets beyond the rectangular
pickup area. Those cases may correctly reject a target before motion.

`--view head` records from an approximate head-mounted camera fixed to the
robot's upper body and tilted down toward the table. It changes the recording
viewpoint only. The controller continues to use the fixed `observe` RGB-D
camera. This viewpoint is not a calibrated reproduction of the physical G1
camera's mounting or lens.

The standard sweep uses tabletop heights of 75, 85, and 95 cm, forward positions
of 28, 31.5, and 35 cm, and lateral positions of 10, 19, and 28 cm to the robot's
left. The random cases sample between those coordinates and also vary can mass
between 330 and 400 g, sliding friction between 0.7 and 1.2, and add 1 mm Gaussian
noise to rendered depth. Coordinates are in metres in the simulation's ground
frame, with x forward, y left, and z up. This is a tested area, not a guarantee
that every point elsewhere on a table is reachable.

## Open in Wendy Sandbox

Open this folder in Wendy, install its Python dependencies in the sandbox, then
run in the sandbox terminal:

```sh
TABLE_HEIGHT=.85 CAN_X=.315 CAN_Y=.19 wendy-simrun wendy_demo.py
```

The script uses Wendy's `wendymujoco` viewer and displays the current task phase
in the Sim tab. The launcher is provided for that environment; the local
evaluation and recorded replay do not need Wendy's viewer integration.

## What the controller measures

`perception.detect` accepts aligned colour and depth images, a 3×3 intrinsic
matrix, and a 4×4 rigid transform from the optical camera frame to the ground
frame. Depth is in metres. It isolates a red connected component, fits the known
66 mm can diameter, and finds the nearby horizontal tabletop. It rejects missing
or ambiguous targets, poor cylinder fits, invalid calibration, and stale captures.
The can is assumed to be upright and 122 mm tall. This is a red-cylinder detector,
not a general object recognition model or a Coke-brand classifier.

Only the evaluator reads the simulator's true can pose to measure detection
error. The planner receives the camera estimate. It refreshes the observation
after planning and rejects a can or table that moved during planning. It does
not yet track someone moving the can during the reach.

The planner tries different hand yaw and pitch angles, checks joint limits and
modelled body/table collisions at target poses, and solves every waypoint before
starting the episode. Joint trajectories use smooth interpolation. Actual arm
tracking and grasp/lift failures stop the episode. The core pickup simulation uses
the original seven-joint Dex3 geometry and torque limits. It does not weld,
attach, teleport, or otherwise constrain the can to the hand.

## Physical G1 status

Physical motion is **not implemented or verified in this version**. No robot was
connected during development. The shipped hardware tool is a read-only
commissioning probe and never creates a command publisher:

```sh
python -m g1_can_handoff.hardware --interface <robot-NIC> --domain <measured-DDS-domain>
```

Run the probe in an environment with Unitree's Python SDK and CycloneDDS installed.
The SDK reference checked for this integration is
`814556d15970dd2ecf1c9984e845ca02ab07e206` from
https://github.com/unitreerobotics/unitree_sdk2_python.
The probe reads `rt/lowstate` and `rt/dex3/left/state`, checks finite/fresh joint
positions, and reports receive rates. Body packet length alone does not confirm
the G1 variant; the robot must be verified as a 29-DOF G1 with seven-joint arms.

To transfer this to the robot, the remaining work is:

1. Connect to the specific G1's Wendy Agent and verify variant, firmware, network
   interface, DDS domain, and actual sensors.
2. Calibrate the aligned depth camera and its transform to the robot's frame.
3. Add and commission an arm-only `rt/arm_sdk` adapter and a left
   `rt/dex3/left/cmd` adapter against the observed SDK/firmware. Arm ownership,
   CRC, mode, joint ordering, bounded command rates, and the supported stop
   procedure need checking on that robot. The simulation's finger array orders
   index before middle; the physical left Dex3 protocol orders middle before
   index. An explicit mapping is required.
4. Replace simulated contact and object-state verification with calibrated
   physical hand/camera feedback. Simulation torque/gain settings are not
   commissioned hardware settings.
5. Verify stationary balance during the arm motion, then test pickup into a
   fixture before a supervised human handoff.

On G1 PC2, retain vendor Ubuntu and install the Wendy Agent. Deploying this app
currently runs a **simulation**, including when the target is a physical device.
It does not control that device's actuators. The staged deployment command is:

```sh
wendy run --device <G1-agent-address> --deploy --yes
```

## Simulation limits

The pelvis is fixed to a support in this manipulation test. It does not validate
standing balance or locomotion. The rendered camera has exact calibration and
synthetic lighting. Real camera noise, occlusion, reflective metal, can deformation,
condensation, and gripper friction require physical tests. No human interaction
physics or automatic recipient acceptance detector is included.

The Unitree mesh model uses convex collision hulls. The scene explicitly excludes
the torso from the neighboring shoulder roll/yaw housing hulls, which overlap
at the shoulder mount. Other arm/body and hand/table contacts remain enabled.
These model choices are documented in `sim.py` and need physical commissioning.

The model and its original license/source reference are retained under
`g1_can_handoff/assets/unitree/`. The application is isolated in this folder and
does not change the existing Wendy simulator or G1 remote-control template.

## Render the bartender scene

```sh
uv run python -m g1_can_handoff.bar_film --preview
uv run python -m g1_can_handoff.bar_film
```

The bar film is rendered directly from MuJoCo joint states with the original G1
and Dex3 model, a furnished bar, textured Coke cans, customer actors, multiple
camera views, and Wendy marketing assets. Output is saved in `results/bar-sim/`.
The controller uses the selected pickup observation layer and verifies the
physical finger contacts and lift. Decorative cans are excluded from that
observation layer. Customer arms follow scripted trajectories. A mocap weld
supplies recipient support after the actor reaches the held can, allowing the
Dex3 to open and the customer to withdraw it. This staging does not validate
autonomous person tracking, human grasp physics, or recipient detection.

`simulation.json` records each pickup, release, recipient support mechanism,
and final can position. The rendered film is labelled as a MuJoCo simulation.
The short edit is 29 seconds, with cuts at the original simulation speed.
The full recording is approximately 52 seconds. Recorded joint and actor states
are retained as compressed NPZ files. Use `--resume` to keep completed clips or
`--reuse-states` to render saved motion again.

## Archived scripted preview

```sh
uv run python -m g1_can_handoff.bar_delivery --scripted-preview
uv run python -m g1_can_handoff.bar_delivery --scripted-preview --reuse-states
```

This revision removes the people, starts both arms beside the hips, picks up
the can from an 85 cm counter, travels around the counter end and along the
open side of a separate 75 cm table, then sets the Coke upright on that table.
The movie, recorded states, and report are in `results/bar-delivery/`.

The movie is labelled **scripted motion preview**. A mocap weld supports the
pelvis along the route, with a scripted leg gait and turns. A temporary hand/can
weld preserves the hold during travel and lowering; it is inactive during
pickup and disabled before the fingers open. Final placement is checked after
the can settles under contact dynamics. The film uses can-contact friction 2.5
and finger gains 24/1, which are simulation staging settings, not hardware gains.
It does not validate autonomous navigation, unsupported walking, or a physical
carry. Recorded equality activation flags preserve the staging lifecycle.

An experimental free-base Unitree velocity-policy adapter and its pinned,
hash-verified ONNX weights are retained under `g1_can_handoff/assets/locomotion/`.
Its initial standing/lateral-motion test worked, but integrated reaching drifted
outside the arm's workspace. An unsupported long carry also lost grip.
Those trials are not the controller shown in that archived preview. The current
physics-driven controller and its remaining limitations are described below.

## Physics-driven bar delivery

```sh
uv run python -m g1_can_handoff.bar_delivery_physics
uv run python -m g1_can_handoff.bar_delivery_physics --reuse-states
```

`bar_delivery` now defaults to this version. The supported preview above is
available only through `--scripted-preview`.

The current controller uses a contact-force quadratic program for balance and
slow alternating steps, with bounded joint torque feedback for the arms and
Dex3 fingers. Its pelvis and can are free bodies. There are no equality
constraints, mocap bodies, attachments, root-pose animation, or applied external
forces. Foot and hand targets are controller setpoints; joint torques and
MuJoCo's contact dynamics produce the movement. The full trajectory is retained
in `results/bar-delivery-physics/states.npz`.

Both arms begin beside the hips. The hand first clears the counter edge before
reaching. After pickup, the wrist turns the can sideways to prevent the axial
slip seen during long upright carries. The G1 takes slow balance steps around
the counter, approaches the table from its open side, adjusts its stance,
returns the can upright, lowers it, opens the fingers, and withdraws. Robot
contacts with the counter top, counter front, delivery tabletop, or table legs
stop the simulation. Contact force is monitored at every physics step.

The recorded full run lasted approximately eight minutes and passed with zero
robot/furniture contact force, zero support constraints or attachments, and
the can upright at approximately `[1.624, -0.429, 0.811]` metres after release.
The minimum pelvis height was 0.714 m and minimum upright cosine 0.993. The
video uses cuts to shorten the walking segments, retaining simulation speed.

The floor is flat and navigation uses the known scene geometry and simulator
state. The physical G1 remains uncommissioned; these
controller gains and exact simulator observations must not be sent directly to
hardware. The seven existing perception/release fault tests still pass.

## Free-standing generalization evaluation

```sh
uv run python -m g1_can_handoff.generalization --workers 4
uv run python -m g1_can_handoff.generalization_full full75
uv run python -m g1_can_handoff.generalization_full full95
uv run python -m g1_can_handoff.generalization_video full75 g13 full95 g02
uv run python -m g1_can_handoff.generalization_report
```

These are tests of the free-standing torque controller, separately from the
older supported-pelvis manipulation evaluation. Every case starts fresh with
the arms beside the hips. Only the scene height and initial can position change;
controller gains and planning rules stay fixed. Pickup uses the rendered RGB-D
estimate. The balance controller reads simulator state; the route uses known
furniture geometry. This is not inside-out visual tracking or SLAM.

The predeclared grid covers counter heights 75, 85, and 95 cm, x positions 28,
31.5, and 35 cm forward, and y positions 10, 19, and 28 cm left, relative to the
initial robot origin. Six random cases use seed 20261001 within those ranges.
Six boundary cases test a 65/105 cm counter or positions outside the tested
rectangle. The pass condition includes pickup, lift, wrist rotation, two
unsupported physical steps, a two-second hold, continued finger contacts,
standing balance, and zero robot/furniture contact force.

Measured results: **24/27 grid cases and 6/6 random cases passed**. At 75 cm,
x=28/y=28 cm failed the grasp-orientation reach check. At 95 cm, x=31.5/y=10
and x=35/y=10 cm failed because the red-region detector reported ambiguity.
All six boundary cases stopped before completing pickup. Every one of these
39 short trials had zero measured robot/furniture contact force.

Full delivery is a separate criterion. The original 85 cm counter / x=31.5,
y=19 cm run passed. A new **75 cm counter / x=28, y=10 cm** run completed the
entire route and placement in 481.4 simulated seconds, with zero furniture
contact force. A **95 cm counter / x=35, y=28 cm** run passed pickup but stopped
58.6 seconds into the episode when the thumb contacted the counter at 46.7 N.
This same pose passed the short test. The contact guard detects a collision
after it begins; it does not guarantee collision avoidance.

The 39-case sweep does **not** run the full route in every case. These finite
samples do not certify every position between them. Can mass (370 g), friction
(1.0), camera calibration, lighting, flat floor, destination table height
(75 cm), and initial robot pose stay fixed. No camera noise is added to this
free-standing sweep. Robustness to those conditions and physical hardware
still needs testing.

`results/generalization-physics/` retains the case manifest, controller hashes,
raw results, logs, camera images, and recorded physics states, including failed
cases. The evidence video replays those states with cuts at the original
movement speed. The detailed report is in `report.json`; `results.csv` lists
all 39 short-trial outcomes.

## Physical controller and shared simulation

The physical pickup work is in [physical/](physical/README.md), on this checkout's current branch. Its MuJoCo replay uses the same `PhysicalPlant`, pickup sequence, depth geometry, and staging planner as the robot. Its synthetic color mask does not exercise the physical Coke-trained YOLO model. The original demo commands above continue to use the original package.

Use [physical/DEVELOPMENT_STATUS.md](physical/DEVELOPMENT_STATUS.md) for the deployed version, current blocker, test commands, and where work continues. Passive DDS recording is in `recorder/`; captured validation results are in `evidence/`.
