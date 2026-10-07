# Warehouse camera search

A simulated Unitree Go2 searches a warehouse using four rendered camera feeds,
a YOLO11n detector trained on ten simulated objects, calibrated camera rays and a route around the
shelves. A pretrained ONNX walking policy applies joint torques in MuJoCo.
The dashboard plays a recorded bark when the robot reaches the detected target.

Navigation requires a validated pose from the companion
[ROS localization app](../warehouse-localization/README.md). The virtual sensor
adapter sends LiDAR and drifting odometry to AMCL. The search controller waits
when localization is unavailable. This is a simulation demo; the cameras and
robot motion have not been deployed as a physical warehouse system.

## Requirements

- Python 3.12 and Git. The launcher creates an isolated environment and fetches
  the pinned Go2 model. Set `PYTHON` to another Python 3.12 executable if needed.
- Licensed access to the Wendy Sandbox simulator runtime, including
  `go2rl.py`, `wendymujoco.py`, `assets/go2_policy.onnx` and its matching
  `assets/go2_policy.yaml`. These proprietary files are not bundled here.
- A running localization container with simulated input enabled. See the
  [simulation setup](../warehouse-localization/README.md#simulation-validation).
- Graphics access for MuJoCo camera rendering. Headless Linux needs a supported
  rendering backend such as OSMesa and `MUJOCO_GL=osmesa`.

## Run

From this directory, set the runtime location and start the demo:

```sh
export WENDY_SIM_RUNTIME=/path/to/wendy-sandbox/image/sim
export LOCALIZATION_URL=http://127.0.0.1:18790
./run.sh
```

The launcher installs `requirements.lock` on its first run and fetches MuJoCo
Menagerie revision `4d038b3feae26ec82b46a4d586379114012a8ac7`.
YOLO needs the `yolo11n.pt` weights in this directory. Ultralytics can fetch
them on first use; keep network access available or supply those weights before
starting. Weights, downloaded robot models and generated runtime files stay
outside Git.

On the first run, the launcher renders shuffled object layouts and fine-tunes
YOLO11n. Training uses simulator segmentation labels; live searches use only
camera images and the requested class. Weights are saved to `runtime/items-yolo.pt`.
This detector is trained for the simulated props and has not been validated on
real warehouse objects. To retrain, run `.venv/bin/python train_items.py`.

Open <http://127.0.0.1:18992/>. Choose **ball, bottle, cup, bowl, banana, apple,
orange, chair, suitcase, or backpack**, then click **Find it**. You can also type
“find the backpack” in the request box. The robot waits for a request, searches
the four camera feeds, and barks on arrival. Asking for another item cancels the
previous route without teleporting the robot. **Reset robot** resets the robot
and returns it to waiting. Enable bark audio once to hear arrivals.
Stop the process with Ctrl+C. Set `PORT` to use a different local HTTP port.

Requests can also be sent to `POST /api/search` with JSON
`{"query":"find the bottle"}`. `GET /api/items` returns the available names.
Invalid or ambiguous requests return HTTP 422 without changing the active search.

For Wendy's native Sim view, launch a compatible desktop build with
`WENDY_LOCAL_SIM_PORT=18992`. Its Sim controls support pause, single-step and
reset. The camera dashboard runs independently of that desktop viewer.
This demo and `g1-minigolf` use the same local port; run one at a time.

The launcher also recognizes a sibling `../../wendy-sandbox/image/sim`
checkout for existing development workspaces. `WENDY_SIM_RUNTIME` supports
other checkout locations.

## Verify

Validate all ten labels and camera-to-floor estimates in the default layout
and a held-out shuffled layout:

```sh
./.venv/bin/python validate_items.py
```

Results and annotated camera snapshots go to `runtime/items-validation*`.
This check uses known positions exclusively to score the detector and ray
estimates. The live controller does not read those positions.


After the robot arrives, keep the demo running and use another terminal:

```sh
./.venv/bin/python verify.py
```

The check reads the local HTTP status and recorded events. It checks detection,
arrival, upright motion, camera snapshots and the bark event. Results go to
`runtime/verification.json`; events go to `runtime/events.jsonl`.
These checks validate the simulated run, not a hardware controller.

The route planner uses the digital twin's collision volumes for shelves and
props, without their semantic labels. The requested destination comes from
YOLO's pixel box and the calibrated camera ray. The virtual LiDAR localizes
against the surveyed static warehouse geometry; movable props are excluded
from its AMCL scan in this simulation.

The live detector reads camera JPEGs through the local HTTP camera endpoint.
The live camera endpoints are `/camera/cam1/snapshot.jpg` through
`/camera/cam4/snapshot.jpg`. Each camera also has a `/stream.mjpg` endpoint.

## Assets

The Go2 model retains its BSD-3-Clause license in
`menagerie/unitree_go2/LICENSE` after download. The arrival recording and its
CC0 source attribution are under [audio/](audio/SOURCE.txt). The Wendy Sandbox
runtime retains its separate proprietary license. Use the YOLO weights under
their upstream terms.

[HOWTO.txt](HOWTO.txt) preserves the original architecture and validation notes.
