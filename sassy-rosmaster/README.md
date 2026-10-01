# Sassy ROSMASTER for WendyOS

A ROSMASTER R2 rover that explores changing rooms, avoids obstacles and maps
what its lidar observes. A real visual detector identifies people from RGB
camera pixels and measures their position with aligned depth. When a person
blocks its route, it honks and asks a local language model for a new sassy remark
grounded in that encounter. The sass dial has three levels.

The app includes a MuJoCo simulator, a live camera view with detection boxes,
a 3D environment view, an occupancy map, downloadable maps, sound and stop
controls. Qwen also chooses exploration goals from sensor-visible candidates.
The motion controller rechecks each choice against current lidar and handles
braking independently of model inference. Camera inference and language-model
requests run outside the motor loop.

Software-rendered camera observations expire after 1.2 seconds; their actual age
is shown beside the image. A brief missed detection can hold an unspoken reply,
but speech requires a fresh observation of the same person before it is emitted.
Lidar braking has its own tighter freshness checks. The physical RGB-D bridge
requires capture-to-publication age below 0.3 seconds. Invalid or stale frames
produce no fresh heartbeat; the motion adapter stops when that heartbeat expires.

Rooms are sampled at startup and reset. Physical pedestrian actors move using
their own range-sensing controllers. Their positions and object labels never
enter the rover's person detector. The 3D environment view shows the simulated
world; the occupancy map and person markers use measured observations.

## Run locally

Install Python 3.12 or newer, uv and [Ollama](https://ollama.com). Start Ollama,
then download the local model and run:

```sh
cd sassy-rosmaster
ollama pull qwen3:1.7b
uv sync --python 3.12
uv run python -m sassy_rosmaster
```

Open http://localhost:8096. Click **Let it roam**, enable sound and click
**Get in its way**. The person button adds an autonomous pedestrian in a safe
corridor ahead of the rover. The room can contain up to five people. Three
pedestrians are present by default. **Clear people** removes them. The rover starts parked. Reset parks it,
samples a new physical layout and pedestrian population, and clears the map.
Camera controls switch between the room overview, overhead and rover views.

Browser sound needs a click to unlock audio playback. Speech uses an available
English browser voice, with British voices preferred. The speech bubble and
log still work if speech synthesis is unavailable.

`ROVER_LLM_URL` defaults to `http://127.0.0.1:11434` and `ROVER_LLM_MODEL` to
`qwen3:1.7b`. The UI reports the actual model response and current failures.
There is no canned dialogue fallback. Lidar navigation continues safely when
the language model is unavailable. `ROVER_SEED` is optional for reproducible
tests; normal runs use system entropy.

## Run in WendyOS

The regular Dockerfile packages the simulation, camera model and human assets.
It does not need a GPU or a local display. OSMesa renders the onboard RGB/depth
camera on the CPU, ONNX Runtime runs visual inference, and the browser renders
the 3D overview. This VM's config reaches the host's local Ollama server at
`http://10.0.2.2:11434`, verified from inside the WendyOS app container. For
another target, set `ROVER_LLM_URL` to an Ollama server that target can reach.

```sh
wendy json validate .
wendy vm create sassy-rosmaster --profile generic --yes
wendy vm start sassy-rosmaster --detach --port 50067 --yes
wendy --json run --device vm:sassy-rosmaster --build-type docker --detach --yes
```

Use the HTTP URL reported by the detached deployment. The app declares port
8096 and persists `map.json` in `/data`. Stop another copy of this app on local
port 8096 before using a user-networked VM, so Wendy can forward the app port.
The checked-in default config selects simulation mode. The VM is generic;
the ROSMASTER model and sensors run inside this app's container.

To use another WendyOS target:

```sh
wendy run --device YOUR_DEVICE --build-type docker --detach --yes
```

Inspect app state, logs and `/health` after deployment. Detached start does not
wait for readiness. `/api/state` reports pose, map, events and fault state.
`/api/map` downloads the latest occupancy grid JSON. Each cell is unknown
`-1`, free `0` or occupied `100`, in row-major order from the map's origin.

## Real ROSMASTER R2

See [the hardware guide](docs/hardware.md) and `hardware.example.json`. The real
adapter consumes ROS 2 lidar and odometry and can drive the existing Wendy
RC-car motion service through its actual `/drive` and `/stop` interfaces. A
second output option supports an Ackermann ROS driver that consumes the
configured command topic. Geometry and steering need calibration.

The hardware image is `Dockerfile.ROS2`. Start from `wendy.hardware.example.json`,
replace the ROS domain and middleware with the robot's measured settings, and
include a calibrated `hardware.json` in the image. The hardware example binds
the operator UI to localhost and starts with motion disabled. Person-specific
remarks use the included calibrated RGB/depth perception bridge. Lidar supplies
obstacle avoidance and mapping. Configure actual camera topics, aligned depth,
calibration and camera-to-base transform before running the bridge.

Copy `perception.example.json` to `perception.json` and follow the hardware
guide's calibration and topic checks. In the ROS hardware image, the bridge
command is:

```sh
/opt/rover/bin/python -m sassy_rosmaster.perception_ros --config /config/perception.json
```

The bridge runs the same bundled visual detector and measured-depth localization
as the simulated RGB-D camera. It preserves the source capture timestamp and
publishes measured person positions to the configured ROS `people_topic`.
Missing calibration, unsynchronized images or expired inference produce no
fresh perception heartbeat. The example leaves calibration confirmation flags
false and the optical-to-base transform unset. Keep them that way until the
physical camera and mounting measurements are available. The hardware guide
includes the separate ROS container launch and read-only configuration mount.

The sample hardware config deliberately refuses motion until chassis geometry,
steering calibration and the driver's independent watchdog are confirmed. It
requires fresh sensor data and an e-stop heartbeat. The adapter stops on sensor
loss, missing controller commands, transport faults, 120 seconds or 10 meters.
The hardware STOP button latches the stop; Reset and Start are separate actions.
The physical ROSMASTER is not available yet. The ROS bridge, sensor discovery,
camera performance and physical motion have not been tested on that hardware.
The bridge can be checked independently while the rover app remains parked.

Onboard speech and horn use `ROBOT_AUDIO=1`, `espeak-ng`, ALSA and the Wendy audio
entitlement. Browser sound works in the simulator without those dependencies.

## What the simulation models

The R2 model is a compact Ackermann rover in a 14 by 12 meter arena with walls,
sampled furniture and autonomous pedestrians. MuJoCo integrates planar chassis dynamics and real
bumper contacts. Drive actuators follow bicycle kinematics with acceleration
and steering limits. Tire slip, suspension and uneven terrain are not modeled.
The dimensions are example values; measure the physical chassis before use.

The controller samples collision-free forward and reverse arcs, remembers
visited positions and prefers visible places it has visited less often. It
uses lidar ray hits and odometry, not the obstacle list used by the renderer.
Mapping integrates observed rays with free/occupied evidence. It does not
perform loop closure, so real odometry drift will distort the map.

## Verify

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover fresh full lidar, physical wall containment, two minutes of roaming
with moving pedestrians and no bumper collisions, observed encounters, RGB/depth
decoding, calibrated projection, capture synchronization, stale perception rejection, hardware sensor faults,
e-stop latching, command watchdogs, travel bounds and R2 command scaling. The
hardware tests use synthetic ROS messages and a fake motion API.

Three.js is vendored under `static/vendor`; its license is included there.
YOLOX-Tiny is bundled in `models` with its Apache 2.0 license. The textured
Cesium Man asset is included under `assets/person`, credited to Cesium and
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The converted OBJ and texture adapt the original glTF asset for MuJoCo.

For a calibrated hardware setup, place the hardware example in a separate
project copy as `wendy.json`, select the measured ROS settings and deploy with
`wendy run --device YOUR_R2 --build-type docker --dockerfile Dockerfile.ROS2 --detach --yes`.
The ROSMASTER motion service and sensor/perception bridges run separately.
