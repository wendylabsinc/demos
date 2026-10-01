# Construction yard drone in a WendyOS VM

MuJoCo integrates a 1.2 kg quadrotor, four rotor forces/reaction torques, 25 ms motor dynamics, gravity and collision contact at 500 Hz. A 250 Hz geometric flight controller reads simulated GPS, velocity, attitude and gyro measurements. Commands change bounded flight references. They never assign drone or worker positions.

The onboard cameras produce actual perspective RGB and metric depth from the same MuJoCo scene: 320×240 for survey evidence and 640×480 for live worker tracking. Navigation maps unlabeled ray ranges and depth observations, plans routes with A*, and chooses survey positions by unobserved camera coverage. There is no fixed A-B-C-D patrol. Current missions revisit the recorded baseline views in an order chosen by distance and the measured map.

Before/after photographs persist in the Wendy volume. ORB features and RANSAC align views; calibrated ground-plane reprojection is the fallback. Pixel differences identify candidate changes. A compact RGB CNN, trained from scratch on actual MuJoCo camera crops, classifies fire, oil and other changes. It has three convolution layers with 16/32/64 channels, 2×2 pooled features and a three-class output. ONNX Runtime runs the frozen model locally on CPU. Its input is image pixels only: no hazard flags, zone labels, simulator poses or world-object list. Confidence, margin and a crop-aspect limit derived from training images govern abstention. Uncertain findings retain their scores as visual changes. Scores are not calibrated real-world incident probabilities.

Microphone recordings are decoded with PyAV and transcribed locally by Whisper tiny.en with Silero voice activity detection. A pretrained Qwen3-0.6B language model running in local llama.cpp interprets free-form text and speech into bounded flight actions. Button actions explicitly request their named operation. No API key or cloud inference is required.

## Run with Wendy

```sh
wendy vm create construction-drone --profile generic --disk 16 --yes
wendy vm start construction-drone --detach --cpus 4 --memory 4096 --port 50701
wendy run --device vm:construction-drone --builder apple-container --build-type docker --detach --yes
```

Skip creation when this dedicated VM already exists. Install the [Wendy CLI](https://docs.wendy.dev/latest/) and make sure `wendy` is on your PATH. Select the intended VM explicitly. On a host with Docker running use `--builder docker`. The first build downloads checksum-pinned model weights and the ARM64 llama.cpp runtime. Open <http://127.0.0.1:8895> and wait for `/api/health` readiness. The application starts on the ground.

Click **Run baseline flight** and wait for the camera coverage mission to finish. Click **Place fire** or **Place oil leak**, then choose arbitrary open ground in the yard. Click **Scan current mission** to compare a second flight with the saved imagery. Evidence and reports persist in `construction-drone-missions` at `/data`. Older synthetic-camera baselines are ignored; record a new baseline after upgrading.

Press the microphone button and allow microphone access in a browser that supports MediaRecorder. Audio uploads to the VM. Type requests when a microphone is unavailable. Examples include "follow me", "please check the fuel depot", "go to zone A and then land", "hold position", and "record the first mission". A is materials, B is fuel, C is equipment, D is excavation.

The arrow controls apply force-controlled movement goals to the simulated worker. Release stops its goal at its current measured position. There is no automatic walking loop. Follow mode searches camera coverage and decodes an ArUco badge on the worker from RGB pixels. Metric depth and camera calibration estimate its position. It follows this tagged simulated worker, not arbitrary untagged people, and does not read the worker's world position. Optional COCO detection is available with `YARD_DRONE_COCO_PERSON_DETECTION=1`; it is disabled for flight control by default because it is slow on this CPU. Missing or stale observations stop tracking motion.

## Simulation limits

The worker is a 75 kg upright planar actor with inertia and contacts, not a biomechanical gait model. Oil follows a conservative coarse thin-film flow. Fire uses fuel and energy balances, diffusion, heat loss and transported smoke. These reduced numerical models evolve with simulation time and render into the actual sensor images. They are not validated fluid or combustion CFD.

The yard geometry is simplified, simulated GPS/IMU measurements are ideal, and the incident classifier was trained and tested on rendered simulator scenes. Its held-out metrics do not measure real-world performance. Unknown changes must remain review findings. This demo verifies simulator behavior only and does not establish outdoor flight or safety performance.

## Local development

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt pytest
.venv/bin/python -m yard_drone.assets
```

Provision Whisper and Qwen from `models.lock.json` and a llama.cpp binary for the local architecture. `yard_drone.models` provisions the Linux ARM64 container runtime. Set `YARD_DRONE_WHISPER_MODEL`, `YARD_DRONE_LLM_MODEL` and `YARD_DRONE_LLAMA_BIN` when developing locally. The incident CNN and its report are bundled in `yard_drone/incident_cnn.onnx` and `yard_drone/incident_cnn.json`. Set `MUJOCO_GL=osmesa` before importing MuJoCo on headless Linux. macOS offscreen rendering needs graphics access.

```sh
YARD_DRONE_DATA=./data .venv/bin/python -m yard_drone.server
.venv/bin/python -m pytest tests -q
```

`GET /api/state` returns physics, observations and mission results. `/api/stream` streams state, `/api/camera.jpg` returns the actual latest sensor frame, `/api/report` returns comparison findings and `/api/evidence/` serves saved photographs. `POST /api/command` accepts `{"text":"please inspect the fuel depot"}`. `/api/voice` accepts bounded binary audio. `/api/action` accepts explicit button actions, `/api/incident` accepts `{"type":"oil","position":[4,3]}`, and `/api/worker` accepts metre goal offsets or `{"stop":true}`.

## Incident model training and validation

The training workflow generates 450 training crops (120 fire, 120 oil, 210 other) and 225 held-out validation crops (60 fire, 60 oil, 105 other). Labels come from scene generation only. Images come from the actual survey RGB camera using Linux OSMesa, the same rendering backend as the deployed application. Incident fields evolve for random ages up to 90 simulated seconds before capture. Training and validation use separate random seeds and disjoint checkerboard release cells. They share the same yard geometry and renderer; camera footprints can overlap. Live VM test frames are never used as training data.

`generate_rgb_incidents.py` creates the first 360 training and 180 validation crops with no embedding or pretrained vision-model dependency. Camera poses include offsets within ±0.9 m and ±2.6 m, views near image boundaries, and fixed or random yaw. Clean/current pairs can differ by independent uniform XYZ translation up to 0.025/0.025/0.01 m and roll/pitch/yaw rotation up to 0.002 radians. Crop context varies between 8, 18 and 36 pixels; the runtime uses 18. A positive label requires the projected offline release to match an actual connected change component and overlap genuine changed pixels in an additional clean render at the same current pose. This avoids assigning hazard labels to unrelated static edges. Worker changes, static parallax components and unchanged scenes provide negatives.

`generate_cnn_negatives.py` then appends a preselected 90 training and 45 validation no-hazard crops of random image regions, visible geometry edges, and the worker and its shadows. It preserves the original images and refuses duplicate augmentation. `train_cnn_incident.py` divides the resulting 450 training images into 360 fit images and 90 internal selection images, stratified by class. Fit images receive reflection, rotation and brightness augmentation. Inverse-frequency loss weights balance the classes. Checkpoint selection, temperature scaling and confidence/margin thresholds use only the 90 selection images. The crop-aspect limit is derived from training images before external validation: crops with a longer-to-shorter-side ratio above 3.9 cause abstention. The 225 validation images are evaluated after these choices.

The shipped checkpoint, selected at epoch 107 of 120, classified 199/225 held-out crops correctly (88.4%). It accepted 202/225 crops (89.8% coverage), with 186/202 correct (92.1% accepted accuracy); 23 caused abstention. Across all raw predictions, including abstained crops, three fires were classified as other, one oil crop as fire and eight as other. Fourteen other crops were classified as hazards: one fire and thirteen oil. This is a small renderer-domain check, not evidence of real-world incident recognition. The model report in `yard_drone/incident_cnn.json` records the full confusion matrix, per-class recall, calibration settings, generation conditions and artifact hashes.

Generate a fresh dataset with Linux OSMesa. On macOS, run both generators in the application image created by the Wendy build:

```sh
INCIDENT_DATASET="$PWD/training/incident-rgb"
mkdir -p "$INCIDENT_DATASET"
container run --rm --cpus 2 --memory 2G \
  --volume "$PWD:/app" --volume "$INCIDENT_DATASET:/training" \
  construction-drone-mujoco:latest \
  python generate_rgb_incidents.py --dataset /training \
  --train 360 --validation 180 --max-age 90 \
  --train-seed 601117 --validation-seed 703219
container run --rm --cpus 2 --memory 2G \
  --volume "$PWD:/app" --volume "$INCIDENT_DATASET:/training" \
  construction-drone-mujoco:latest \
  python generate_cnn_negatives.py --dataset /training --train 90 --validation 45 \
  --train-seed 941273 --validation-seed 1051387
```

The image sets `MUJOCO_GL=osmesa`. On headless Linux, use that backend with the same dependency versions. Both generators use actual camera rendering without model inference. Do not merge images from earlier macOS-rendered datasets into this dataset.

PyTorch and the ONNX exporter are development dependencies; the application runtime does not need them. Once the RGB dataset and manifests exist, train for 120 epochs locally:

```sh
uv pip install --python .venv/bin/python torch==2.8.0 onnx==1.18.0
.venv/bin/python train_cnn_incident.py \
  --dataset "$INCIDENT_DATASET" \
  --output yard_drone/incident_cnn.onnx \
  --epochs 120 --size 96 --device mps --threads 2 --seed 314159
```

The shipped model was trained with macOS MPS. Use `--device cpu` when MPS is unavailable. This choice changes training execution only; the exported application model runs on CPU.

Set `YARD_DRONE_VISION_MODELS=./models/vision` if enabling the optional COCO person detector. It is separate from incident classification and tagged-worker following.
