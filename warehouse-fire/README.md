# Warehouse thermal fire watch

A working 40 × 40 m MuJoCo warehouse demo with 88 radiometric thermal cameras,
64 independently controlled sprinkler zones, a digital twin map, and Web Push.
The native Wendy app displays the live MuJoCo scene through its local simulation
bridge. The dashboard displays the thermal frames and detected locations.

## Run

Python 3.12 is required. From this directory:

```sh
./run.sh --auto
```

Open http://127.0.0.1:8788/. The automatic scenario starts fires in an aisle,
on a storage rack, and in the east corner. The dashboard's scenario button
starts all three at once. Click the map to ignite another surface. Reset clears
all simulated fires. Pause, single step, and reset also work in Wendy's Sim tab.

The Wendy native app uses port 8788 by default. For another port, set `FIRE_PORT`
on the demo and `WENDY_LOCAL_SIM_PORT` when starting Wendy. Select the running
warehouse simulation in the Sim tab. No modification of the Wendy app is needed.

Click **Enable** under Fire notifications in a supported browser to authorize
and register Web Push. A test notification follows registration. Push messages
work through the browser's push service even when the dashboard tab is closed.
The browser needs network access; remote dashboard deployments need HTTPS.
The local dashboard uses a secure localhost context. Notifications explicitly
say DEMO. Alert banners and map markers work without browser permission.
`push_sent`, `push_failed`, and `push_unsubscribed` events distinguish delivery
from an alert merely being queued. macOS Focus and browser settings can suppress
the visible notification even after the push service accepts it.

## What is simulated

- MuJoCo models the warehouse's occluding geometry, camera poses, and gravity,
  free-joint movement and contacts of sprinkler water particles.
- A radiometric adapter raycasts each 32 × 24 camera pixel onto visible surfaces.
  The static warehouse permits caching rays. Racks block those rays. A heat
  field generates temperature frames with sensor noise. Ceiling and rack aisle
  cameras cover accessible surfaces; opaque racks are not transparent.
- Fire growth and water cooling use an explicit simplified temperature model.
  MuJoCo does not simulate combustion, infrared radiation transport, or fluids.
- The detector receives temperature frames. It finds connected hot regions of
  at least two pixels above 80 °C, sustained across three scans from a camera.
  Only then does the calibrated twin map pixels to world surface coordinates.
  It merges overlapping camera reports, activates the estimated local 5 × 5 m
  sprinkler zone, and queues Web Push. Valves close after five seconds without
  hot observations. The rendered flames never enter the thermal detector.
- Ground truth fire positions are isolated in `runtime/truth.json` for
  verification. They do not enter localization or valve selection.

## Sim to real with Wendy

The same `FireController` accepts simulated and hardware radiometric frames.
`hardware.py` supplies an HTTP gateway contract for actual cameras and valves.
Deploy the bundled Dockerfile and `wendy.json` with an explicit Wendy target:

```sh
wendy run --device <device-selector> --detach --yes
```

The Docker image is tested headlessly on Linux. No physical device was selected
or tested for this demo. For hardware operation,
configure the gateway URLs in a private JSON file and run
`python demo.py --hardware /path/to/gateway.json`. Set that command in your
hardware deployment image. The camera installation must match the twin's
camera intrinsics, poses, camera order and warehouse geometry. Adapt `twin.py`
to the surveyed installation before deployment; `runtime/calibration.npz`
exports the simulated camera surface hits for inspection. Do not use simulated
calibration for a different real warehouse.

The thermal gateway must return:

```json
{"timestamp": 0.0, "sequence": 1, "celsius": "88 × 24 × 32 numeric array"}
```

`timestamp` is capture time in Unix seconds, synchronized with the controller;
`sequence` must increase. Replace the explanatory string with nested numeric
arrays. Nonfinite temperatures, invalid dimensions, old captures, future
captures, and repeated sequences cause a sensor fault. Thermal cameras must
supply calibrated Celsius values; colored JPEGs are insufficient.

Valve commands POST `{"zone": 0, "on": true}` and require
`{"ok": true, "zone": 0, "on": true}` acknowledgement. Hardware mode disables
simulated ignition and reset. Observation faults preserve the existing valve
state and report an alert; they never claim a fire is resolved. An actuator
fault remains visible and requires gateway/operator intervention. Actual
camera acquisition, gateway authentication, valve drivers and independent
fire protection interlocks belong to the installed hardware integration.
This demo has not been validated as a building fire protection system.

The Wendy manifest declares networking, HTTP port 8788 and persistent `/data`
for events and push credentials. The HTTP gateway avoids needing local GPIO or
camera entitlements. Local sensor drivers would need their matching entitlements.
The dashboard and controls are intended for a private demo network; add access
control and HTTPS before exposing them beyond that network.

## Verification

```sh
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python verify.py
```

The verifier builds the real MuJoCo model, runs the automatic three-fire scenario
without sleeping, checks localization against isolated ground truth, verifies
only local zones activate, and verifies cooling and resolution. It also checks two hot pixels are visible
at each of 400 sampled warehouse surface locations on a 2 m grid. It writes
`runtime/verification.json`. Tests also cover occlusion, no false alarm at
ambient temperature, independent confirmation, invalid hardware input and
valve acknowledgements. Test output establishes simulator behavior only.

All generated data, push credentials and Python environments stay outside Git.
MuJoCo and the dashboard bridge require no proprietary runtime files.
