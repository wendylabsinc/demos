# Wendy Go2 room assistant

A voice/text assistant with sensor-built mapping, learned camera recognition,
persistent room names and a phone interface. It runs either with live MuJoCo Go2
torque locomotion or with a Go2 Air through Wendy's ROS 2 bridge. The simulator
uses measured range and RGB-D exposures inside an authored household scene. Air
mode uses the robot's accumulated occupied-voxel map, odometry and RGB camera;
it never imports scene coordinates or pretends that the RGB camera measures
depth. Neither backend has demonstrated reliable operation across arbitrary
household layouts.

## Run

```sh
uv sync --extra dev
uv run python scripts/prepare_assets.py
uv run go2_assistant serve --port 8787 --memory results/sensor-home-v3.sqlite3
```

Open http://127.0.0.1:8787/. Camera shows the current RGB view; Sensor map shows
only observed occupancy and estimated pose. Simulator is an external observer
view, including geometry the robot has not seen. The camera display runs up to
20 FPS separately from perception. The display uses 320x180 pixels; learned
recognition receives calibrated 640x360 RGB-D exposures. Projection and person
tracking run in a bounded background worker.

In simulation, try `explore the house`, `find a cup`, `follow me`, `come here`,
`spin`, `lie down`, `stand up`, or `stop`. Air mode supports exploration, visible
object search, room memory, room navigation, spin and stop; RGB-only operation
does not expose following, approach or posture controls. Room names start empty:
after reaching a location, say `remember this as kitchen`, then `go to the
kitchen` on later trips.
With the default detector, an unknown room name cannot be recognized from images;
exploration builds geometry, and you teach its names.
The simulator person has manual operator controls. Arrows request 35 cm moves;
the actor walks toward the requested point at 0.5 m/s and stops there. No scheduled
actor route runs in the phone session. The human mesh, geometry derivative and attribution
are under `go2_assistant/scene_assets/person/`.

## Where decisions come from

The map begins unknown. In simulation, a bounded occupancy mapper integrates
measured lidar rays, relative velocity/gyro odometry and local scan matching. In
Air mode it consumes `/utlidar/cloud_base`, keeps body-height occupied voxels,
and derives free evidence only along sectors containing a nearest occupied
voxel. Empty sectors stay unknown; they are never synthetic maximum-range rays.
The paired `/odom` pose places the accumulated voxel map in the planning frame.
Routes use only derived observed-free cells, inflated for the robot footprint.
A* computes routes and frontier exploration chooses newly reachable viewpoints;
it does not read a supplied floor plan or follow a stored demo route. New voxels
invalidate routes. The planar obstacle slice can miss elevated, overhanging and
drop-off hazards and does not certify clearance for the complete 3D body.

The default local provider uses pinned YOLOS tiny INT8 for passive person
awareness and DETR ResNet-50 INT8 for active object queries. It runs one learned
inference at a time and records the selected model on each camera exposure.
Actual RGB pixels are the neural input; its trained class metadata supplies the
fixed vocabulary, including `cup`, `bottle` and `person`. Simulation uses metric
depth and a calibrated camera transform to project accepted boxes into the map.
The Air bridge provides neither depth nor camera calibration, so Air detections
remain image boxes: they can confirm that an object is visible but never gain an
invented distance or map position. No marker IDs, color lookup tables or fixture
coordinates enter recognition.

`Keys`, color-qualified requests such as `red cup`, and ownership requests such
as `my cup` are unsupported by the default detector and are rejected explicitly.
Detection scores are model outputs, not calibrated probabilities. Recognition
can miss visible objects or produce an incorrect label or box. The current
trained detectors recognized the mug and the person from actual camera
images. Following still requires a fresh tracked person detection.
A failed or absent observation never substitutes a programmed successful find.

Object search stops at viewpoints to await inference, looks around and selects
new frontiers from the map. Slow observations are accepted only while the robot
remains near their original capture viewpoint. On Air, a successful result says
only that the object is visible in the camera. Following is simulator-only: it
requires fresh spatial RGB-D tracking, retains the selected track identity and
stops on loss. It does not identify an owner from appearance.

Rooms start unnamed. Users can teach a name at the measured robot position.
The local object detectors do not classify rooms. Optional vision-language
room hypotheses with concrete evidence and sufficient confidence can add names,
but the local Ollama room-label path has not been verified: its probe timed out.
User-taught names are the supported default workflow.
SQLite stores names, last-seen objects and conversation under the sensor map ID.
Last-seen objects are search hints, never proof of a current find.

The occupancy file defaults beside the memory database with `.map.npz` suffix.
Air mode requires an explicit map/session ID. Use a new ID and map file after a
robot odometry or voxel-map reset; the bridge does not supply a durable map
revision or global relocalization guarantee. The live simulator sensor-only run
uses `results/sensor-home-v3.sqlite3`. The earlier
`learned-home` and `sensor-home-v2` maps remain for diagnosis. After relocation,
the restarted simulator could not uniquely match the accumulated v2 map, so the
current run starts a fresh map from lidar. No old coordinates are imported.
Use `--mapping PATH` to choose another map. Reload requires three consecutive,
consistent lidar scans that uniquely match a bounded saved sensor keyframe before
navigation or stored-map updates. Ambiguous or unrelated scans leave navigation
disabled. This is local SLAM without global loop closure or
arbitrary global relocalization; choose a new map file for a different environment.

Spin, resting and standing remain explicit bounded skills. The walking policy
runs live on current MuJoCo observations and applies joint torques; navigation
never teleports the robot or replays a recorded trajectory. `Sit` uses the
supported lying/rest pose, not a separate hindquarters sitting posture.

## Optional vision-language provider

The default local models need no service. Set `GO2_VISION_PROVIDER=detr` or
`yolos` to use one detector throughout. An optional Ollama provider can
attempt open vocabulary and room hypotheses:

```sh
ollama pull qwen3-vl:4b
GO2_VISION_PROVIDER=ollama GO2_VISION_URL=http://127.0.0.1:11434 GO2_VISION_MODEL=qwen3-vl:4b uv run go2_assistant serve
```

This path remains unverified in the current scene; a local probe timed out.
It provides no guarantee of identifying keys, colors, ownership or room types.
Slow results retain their original exposure timestamps, and tasks stop or wait
when observations cannot meet their freshness requirements.

## Language and voice

Language is separate from perception. Without GO2_LLM_MODEL, typed/dictated
commands use the labeled offline parser. Learned object recognition and lidar
mapping still run. To enable open-ended tool-calling conversation:

```sh
export GO2_LLM_BASE_URL=https://api.openai.com/v1
export GO2_LLM_MODEL=<tool-capable-model>
export GO2_LLM_API_KEY=<your-key>
uv run go2_assistant serve
```

Chat Completions and Responses providers are supported. The agent validates
skills and arguments; the model cannot supply actuator commands or arbitrary
map coordinates. Stop bypasses model inference and invalidates pending commands.
Browser dictation and speech replies depend on browser support and microphone
permission. Recorded transcription additionally needs GO2_TRANSCRIBE_MODEL.
The phone interface is a web page, not SMS or WhatsApp. For access beyond
localhost, set GO2_HTTP_TOKEN and use HTTPS or a trusted tunnel.

## Tests

```sh
uv run pytest -q
uv run go2_assistant demo --record --output results/learned-demo
uv run go2_assistant evaluate --output results/sensor-evaluation
```

Unit tests generate sensor rays from varying test environments and check unknown
space, biased odometry correction, route changes, map reload, model validation,
depth projection, tracking loss and stale observations. Native simulator tests
exercise the actual joint policy and measured sensors. Demo/evaluate use the real
configured vision model at real-time pacing and report its actual result; model
failure is reported as failure. They provide no room or object coordinates.
Old recordings and evaluation reports predate this sensor-based implementation.
They are historical artifacts and do not validate the current perception/mapping.

The current Linux Docker model-only check recognized `cup` with score 0.9422 and
`person` with score 0.9979 using the pinned detector models. The Linux OSMesa
end-to-end evaluation under the current host load failed the cup search after
1.4 simulated seconds with `Robot observations are stale.` Map initialization
and the injected sensor-fault stop passed. The report is
`results/linux-current/evaluation.json`; Linux object search and room roaming
still need a passing end-to-end run.

Performance depends on available CPU and renderer capacity. Inference can take
seconds, and software rendering competes with physics, lidar and tracking. The
20 FPS display rate is a configured ceiling. A recent loaded-host sample delivered
about 9.8 distinct camera frames per second, with mean frame-request latency of
69 ms and 95th percentile of 173 ms. A sensor freshness failure stops the task;
model-only recognition scores do not establish navigation or following reliability.

## Wendy deployment

`wendy.json` declares the phone port, host networking, ROS 2 peripheral scope
and a persistent memory volume. Its ROS Humble image supports simulator, Go2 Air
and the separate commissioned-service adapter. Prepare the pinned detector
weights—and the simulator's policy and robot meshes—with
`scripts/prepare_assets.py` before an image build that needs learned vision.
`--check` validates
their sizes and SHA-256 hashes offline. The small scene assets ship in the Python
package; the larger downloaded `assets/` directory is separate and is copied
into the container. Validate and stage on an explicit target:

```sh
wendy json validate
wendy run --build-type docker --device <agent-address> --deploy --yes \
  --env GO2_HOST=0.0.0.0 --env GO2_HTTP_TOKEN=<access-token>
```

That default deployment is a simulation. The manifest also declares ROS 2
Humble, CycloneDDS, domain 0, peripheral scope and host networking for the Go2
bridge. Stage Air mode without arming to inspect sensor readiness:

```sh
wendy run --build-type docker --device <agent-address> --deploy --yes \
  --env GO2_BACKEND=air \
  --env GO2_MAP_ID=<new-odometry-session-id> \
  --env GO2_HOST=0.0.0.0 --env GO2_HTTP_TOKEN=<access-token>
```

Air mode subscribes to `/odom`, `/utlidar/cloud_base` and
`/camera/color/image_raw`, and publishes `/cmd_vel`. It starts disarmed and
publishes only zero velocity. For a supervised motion session, redeploy or
launch with `GO2_ARM=1`; arming waits for fresh paired odometry and voxels,
verifies that this app is the only `/cmd_vel` publisher, and still waits for a
user task. The bridge's command expiry provides an additional stop after 500 ms,
while this app refreshes an active command at 20 Hz.

## Go2 Air integration status

The ROS adapter and controller are tested with synthetic odometry, accumulated
voxel maps, camera inputs and recorded command sinks. The Go2 bridge's fake robot
and the managed simulator are the integration targets before physical use. No
physical Go2 was accessed or moved while adding this backend.

Air motion uses a fixed 0.55 m/s forward request because the physical robot
ignores lower forward speeds. It turns before walking, requires a current route
through voxel-derived free cells, requires at least 60% current front-sector
coverage, and stops below 0.85 m clearance. Stale/reordered sensors, odometry
jumps, changed routes, incomplete coverage, a second velocity publisher or a
controller timeout produce zero velocity; sensor and ownership faults disarm.
These software checks do not establish physical stopping distance.

Before relying on the physical path, verify under operator supervision in this
order: sensor-only startup, explicit stop, stale-stream stopping, a bounded turn,
a short unobstructed forward goal, obstacle stopping, then room routing. Use a
new `GO2_MAP_ID` after every odometry/map reset. Person following, `come here`,
posture changes, metric object positions and camera/voxel depth fusion are not
part of this safe first pass.

The older commissioned-service adapter remains available as
`GO2_BACKEND=hardware` for installations that provide its documented HTTP
observation, navigator and motion contracts. It is separate from Go2 Air mode.

Vendored Wendy runtime provenance and licensing are under
`go2_assistant/_physics/`. Upstream robot/policy assets retain their pinned
manifest and licenses. `scripts/prepare_assets.py` verifies byte size and SHA-256
and supports offline validation. Downloaded runtime assets and results are excluded from Git. A wheel includes
`go2_assistant/scene_assets/`, but not the downloaded policy/robot/vision bundle.
When using a non-editable wheel installation, select the downloaded robot/policy
root with `--assets PATH` and the detector directory with
`GO2_VISION_ASSETS=PATH/vision/detr-resnet-50`.
