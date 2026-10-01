# Unitree G1 minigolf demo

The G1 balances on a putting green, takes a backswing, and hits a free golf ball with a physical putter in Wendy's MuJoCo Sim view. After the putt, the robot keeps balancing. Use Reset to start another round. Choose a track from the course picker at the bottom of Wendy's Sim view. Selecting a course initializes a new round.

The entire robot has a free floating base, the Menagerie link masses and inertias, joint limits, and collision geometry. Ground contact supports the robot. A closed-loop controller commands its joint position servos with gravity and velocity feedforward. Measured torso tilt and angular velocity feed ankle balance corrections. The arm controller computes joint targets in a separate planning state and waits for measured balance and backswing position before striking.

The putter is a rigid attachment to the right hand. Its 0.20 kg head and 0.09 kg shaft contribute mass, inertia, and collision geometry to the robot. They move only through the robot's joints. The ball has its own free joint. Contact, rolling friction, gravity, and the cup walls determine its trajectory.

During a round, the controller writes actuator inputs only. It never assigns the robot or ball's positions or velocities, applies a supporting force to the base, or moves mocap bodies. Initialization and the explicit Reset command set the initial state. There are no automatic state resets during playback.

This is a feedback-control demo in simulation. It is not a learned policy or a controller verified on physical hardware. The robot starts in a prepared putting stance with a rigid grasp, rather than walking to the ball or picking up the club.

## Tracks

- Garden green has an open fairway and a 2.20 m putt.
- Bend turns toward a cup offset 70 cm to the side. Curved physical rails redirect the ball around the turn.
- S curve crosses a narrow green with two opposing turns. The rails guide the ball through contact. The open gaps beside the green have no hidden floor; a missed ball falls into the sunken garden.

The curved tracks use continuous visible surfaces over overlapping collision segments to keep the joins clean. Each track has a real cup opening, its own course geometry, turf color, and servo stroke speed. Ball motion always comes from club contact and MuJoCo dynamics. Track selection and explicit Reset are the only ways to restart a round.

## Run

Run `./run.sh` from this folder. Start a particular track with `./run.sh --course gates` or `./run.sh --course bridge`. The launcher installs the locked Python dependencies when needed and fetches the pinned Unitree G1 model from MuJoCo Menagerie. Python 3.12 and Git are required. Set `PYTHON` to another Python 3.12 executable if needed.

Interactive playback requires licensed access to the Wendy Sandbox simulator runtime. Set its location before starting:

```sh
export WENDY_SIM_RUNTIME=/path/to/wendy-sandbox/image/sim
./run.sh
```

That directory must contain `wendymujoco.py`. The runtime retains its separate proprietary license and is not bundled here. Existing development workspaces can also use a sibling `../../wendy-sandbox/image/sim` checkout. Headless verification does not require this runtime or the Wendy desktop app.

The simulator adapter listens on `127.0.0.1:18992`. Launch a compatible Wendy desktop build with `WENDY_LOCAL_SIM_PORT=18992`. Pause, single-step, resume, and reset work through Wendy's Sim controls. Stop with Ctrl+C or the close button beside the running simulation. Stop playback before running verification. This demo and `warehouse-demo` share the same local port; run one at a time.

## Verification

Run `./run.sh --verify-all` to check all tracks, or `./run.sh --verify --course bridge` to check one track. These commands advance a round without opening a viewer. It checks finite simulation state, the robot remaining upright, physical club contact, the ball inside the cup, and a final ball speed below 0.05 m/s. Results are written to `runtime/verification.json` and `runtime/verification-<course>.json`.

Run `./.venv/bin/python verify_physics.py` for the physical-model audit. It checks every track. Disabling `mj_step` leaves live positions and velocities unchanged, that the model has no mocap bodies or external supporting forces, and that disabling the joint actuators makes the robot collapse under gravity. It also checks ball contact with the curved rails and a ball falling through an S-curve gap. Round verification checks that the ball turns sideways on Bend and travels in both lateral directions on S curve.

Model source: Google DeepMind MuJoCo Menagerie, revision `4d038b3feae26ec82b46a4d586379114012a8ac7`. Its Unitree G1 license is included under `menagerie/unitree_g1/LICENSE`.
