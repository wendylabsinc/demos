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
service URL. The model runs on the host Mac; the app and physics run on WendyOS.
The app does not bundle the model weights in its container.

OpenAI-compatible multimodal endpoints are also supported. Set `GO2_AI_URL`
to the service's `/v1` base or full chat completions endpoint,
`GO2_AI_PROVIDER=openai-compatible`, `GO2_AI_MODEL`, and optionally `GO2_AI_KEY`.
The model must accept image input and the structured action schema. Frames are
sent to the configured service, so use a destination appropriate for your data.

## Physical Go2

```sh
python3 scripts/deploy.py --profile hardware --device YOUR_GO2_COMPANION --domain 0 \
  --env GO2_AI_URL=YOUR_VISION_MODEL_ENDPOINT \
  --env GO2_AI_PROVIDER=ollama --env GO2_AI_MODEL=qwen3-vl:4b \
  --env UNITREE_INTERFACE=YOUR_ROBOT_NETWORK_INTERFACE \
  --env ODOM_TOPIC=YOUR_ODOMETRY_TOPIC --env SCAN_TOPIC=YOUR_SCAN_TOPIC \
  --env RGB_TOPIC=YOUR_RGB_CAMERA_TOPIC
```

Hardware remains uncommissioned. A Go2 EDU needs developer access, a WendyOS
companion and a verified ROS sensor bridge. Inspect its actual DDS domain,
frames, QoS and timestamps. Odometry, scan and RGB topics are required; depth
is optional. The raw RGB image goes to the same agent used in simulation.
The Unitree SDK motion worker runs separately from the ROS DDS library.

The safety executor permits only forward steps and in-place turns lasting at
most one second. It enforces scan clearance, freshness, geofence and session
budgets; the hardware worker also has an independent real-time watchdog.
It does not guard drop-offs. Physical operation requires a bounded flat area
and commissioning with the robot's stop control.

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
