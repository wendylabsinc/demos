# Wendy robot AI

The default app is an image-grounded vision-language agent. It sends the current
robot camera, measured pose and lidar clearance, the user's instruction, and
recent execution feedback to a real model. The model chooses one short action.
The robot executes that bounded action, stops, captures a new image, and asks
again. There are no predefined search paths or scheduled actor events in the
active app. Missing or invalid model output stops the robot.

The simulator retains genuine ONNX-policy locomotion through MuJoCo joint
torques. Its localization is exact simulated pose, not SLAM. Camera perception
comes from the model's pixels. No prop colors, ArUco IDs, target coordinates,
object labels, or injected gesture events reach the model.

## Run locally

Install Ollama from its official distribution and download a vision model:

```sh
ollama pull qwen3-vl:4b
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python sim/scripts/prepare_assets.py
GO2_AI_URL=http://127.0.0.1:11434 GO2_AI_PROVIDER=ollama GO2_AI_MODEL=qwen3-vl:4b \
  .venv/bin/python -m showcase.server --profile sim
```

Open `http://127.0.0.1:8788`, write an instruction, and press Start AI. The five
suggested instructions are prompts, not programs. You can replace them with any
bounded room task. Stop invalidates any model request in flight. A late response
cannot restart motion. The UI displays the actual model, its latest decision,
visual observations, and recent decisions.

The simulator's person, key geometry and barrier have manual controls. Moving
an actor or changing its visible arm pose changes only the rendered world; the
model must interpret it. No automated scenario driver runs during a session.

## WendyOS deployment

```sh
python3 scripts/deploy.py --profile sim --device vm:chatgpt-go2 \
  --env GO2_AI_URL=http://10.0.2.2:11434 \
  --env GO2_AI_PROVIDER=ollama --env GO2_AI_MODEL=qwen3-vl:4b
```

The local QEMU user-network gateway exposes the host Ollama service at
`10.0.2.2:11434`. A different VM or physical device needs a reachable inference
service URL. The model runs as a separate service; neither physical profile
bundles model weights or Ollama in its container.

`deploy.py` builds with Docker by default. On a Mac without Docker, add
`--builder apple-container` (Apple's `container` tool); `--builder buildkit` is
also accepted. The option is passed straight to `wendy run --builder`.

The page loads the model before it accepts a run. While Ollama is loading it,
the page shows "Model loading" and Start AI stays disabled; if the service is
unreachable it shows that and keeps retrying. A slow model therefore never
starts a run that then fails.

### Protecting the page

The operator page and its motion endpoints are open by default, which is fine on
`127.0.0.1` and unsafe on a shared network. Set `GO2_HTTP_TOKEN` to require
`Authorization: Bearer <token>` on `/status`, `/health`, `/camera.jpg` and every
POST, including Stop. The page itself and `/config` stay open so the browser can
ask for the token, then keeps it for the browser session.

```sh
python3 scripts/deploy.py --profile air --device YOUR_WENDYOS_DEVICE \
  --env GO2_HTTP_TOKEN=$(openssl rand -hex 16) ...
curl -H "Authorization: Bearer <token>" http://DEVICE:8788/status
```

Pass the token with `--env` rather than committing it. It is sent over plain
HTTP, so on anything but a trusted LAN or tunnel it only deters casual access.
Without a token, a server bound beyond localhost prints a warning at start.

OpenAI-compatible multimodal endpoints are also supported. Set `GO2_AI_URL`
to the service's `/v1` base or full chat completions endpoint,
`GO2_AI_PROVIDER=openai-compatible`, `GO2_AI_MODEL`, and optionally `GO2_AI_KEY`.
The model must accept image input and the structured action schema. Frames are
sent to the configured service, so use a destination appropriate for your data.

## Go2 Air through the Wendy bridge

The Air profile uses the Go2 bridge's ROS 2 Humble/CycloneDDS contract. It reads
`/utlidar/robot_odom`, `/utlidar/cloud_base` and
`/camera/color/image_raw`, and publishes bounded commands to `/cmd_vel`.
It needs no robot key, admin entitlement, Unitree SDK or robot network interface.

```sh
python3 scripts/deploy.py --profile air --device YOUR_WENDYOS_DEVICE --domain 0 \
  --env GO2_AI_URL=http://127.0.0.1:11434 \
  --env GO2_AI_PROVIDER=ollama --env GO2_AI_MODEL=qwen3-vl:4b
```

`--profile ros-sim` uses the same ROS contract against a managed Go2 simulator.

### Ollama on the Jetson

`GO2_AI_URL=http://127.0.0.1:11434` works when Ollama runs as its own Wendy app
on the same Jetson, because both apps use host networking. `ollama-app/` is a
small reference app for that: at each start it downloads Ollama 0.40.4 for arm64
from its GitHub release (needing internet access, and into the container because
the persist volume is mounted noexec), serves it on loopback, keeps the model files
in a persist volume and pulls `qwen3-vl:4b` in the background. 0.40.4 is the release
verified on a Jetson Orin; its `cuda_v13` libraries match that device. Another
release can be tried with `OLLAMA_VERSION`, after checking that the GPU is used.

```sh
cd ollama-app
wendy run --device YOUR_WENDYOS_DEVICE --detach --yes   # add --builder apple-container without Docker
```

The first start downloads about 1.5 GB of Ollama and 3.3 GB of model, and the page says "Model loading" until
the model is resident. Two settings in its Dockerfile matter on a Jetson:

- `OLLAMA_CONTEXT_LENGTH=8192`. Left alone, Ollama picked a 262k-token context
  for this model, a roughly 43 GB allocation and a 102 s first load on the test
  Orin. At 8192 (the size the showcase asks for) it is about 4.4 GB.
- `OLLAMA_KEEP_ALIVE=-1` keeps the model loaded between runs so Start does not
  wait for a reload. The page's warm-up also pins it.

Put the Jetson in its maximum power mode before judging speed. On the test Orin
the default 30 W mode gave about 9.7 tokens/s; MAXN (`sudo nvpmodel -m 0`, then
reboot) gave about 31.8 tokens/s warm and a 16 s cold load with the settings
above. Check the active mode with `sudo nvpmodel -q`. A different Jetson will
differ. Another reachable Ollama or OpenAI-compatible endpoint also works; set
`GO2_AI_URL` to it.

The Air sends an accumulated occupied-voxel map rather than raw lidar scans.
The app projects body-height points into five-degree obstacle sectors. Missing
sectors remain unknown, not fabricated clear space. Because the short-range map
leaves gaps in wide rooms, the Air profile defaults to
`GO2_ALLOW_SCAN_GAPS=true`: motion still needs measured front clearance and
measured turn sectors, but it cannot protect against obstacles in unobserved
directions. Set `--env GO2_ALLOW_SCAN_GAPS=false` to require a complete measured
front corridor for forward motion and complete coverage for rotation.

## Go2 EDU legacy profile

The existing EDU-only SDK path remains separate:

```sh
python3 scripts/deploy.py --profile hardware --device YOUR_GO2_COMPANION --domain 0 \
  --env GO2_AI_URL=YOUR_VISION_MODEL_ENDPOINT \
  --env GO2_AI_PROVIDER=ollama --env GO2_AI_MODEL=qwen3-vl:4b \
  --env UNITREE_INTERFACE=YOUR_ROBOT_NETWORK_INTERFACE \
  --env ODOM_TOPIC=YOUR_ODOMETRY_TOPIC --env SCAN_TOPIC=YOUR_SCAN_TOPIC \
  --env RGB_TOPIC=YOUR_RGB_CAMERA_TOPIC
```

The EDU profile remains uncommissioned. It needs developer access, a WendyOS
companion and a verified ROS sensor bridge. Inspect its actual DDS domain,
frames, QoS and timestamps. Odometry, scan and RGB topics are required; depth
is optional. Its Unitree SDK motion worker remains separate from ROS DDS.

Both physical profiles permit only forward steps and in-place turns lasting at
most one second. They enforce measured clearance, freshness, geofence and
session budgets and have independent command watchdogs. They do not guard
drop-offs. The operator must put the already-standing robot in a bounded flat
area and keep its stop control ready; the app sends no posture commands.

A model's completion report is explicitly unverified. Visual-language models
can misidentify objects or fail a task. The app shows those outcomes instead of
substituting a programmed successful ending.

## Verification and recording

```sh
python3 -m pytest tests -q
python3 sim/scripts/prepare_assets.py --check
python3 marketing/ai-record.py --url http://127.0.0.1:8788 --seconds 60
```

The AI recorder observes the running app and sends no robot actions or actor
movements. It saves video and actual model/robot state. FFmpeg is required.
The previous scripted baseline and its footage are preserved under
`marketing/archive-controller-baseline`; they are not proof of this AI app.

The pinned Go2 policy/model sources and hashes are in `sim/assets.lock.json`.
The preparation script downloads and verifies the bundle and its licenses into
the ignored `sim/assets` directory. MuJoCo provides robot physics; the
locomotion policy controls joints; the VLM chooses task actions. The deterministic
safety controller constrains motion without choosing task goals.
