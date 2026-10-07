# Autonomous city taxi

A local MuJoCo city simulation displayed in Wendy's native Sim view, with a
browser dashboard for the onboard cameras, LiDAR, route and passenger missions.
The taxi drives among four traffic cars, obeys stop lines and a STOP sign, waits
for passengers, and repeats trips between Central station, Riverside hotel and
City gardens.

## Run

Requires Python 3.12 and uv. The first run creates a separate environment and
installs the pinned dependencies.

```sh
./run.sh
```

Open http://127.0.0.1:8788/ for the camera dashboard. Drive releases the emergency
brake, Brake brings the taxi to a stop while traffic continues, Pause freezes the
whole simulation, and Reset restarts the city. The simulation starts driving
when launched. It connects to no physical vehicle.

A Wendy native build using `WENDY_LOCAL_SIM_PORT=8788` discovers the taxi in its
Sim tab. Native pause, single-step, reset and close controls use the same backend.
If an existing Wendy build is configured for another demo, prepare a separate
local copy instead of changing that app:

```sh
python3.12 prepare_viewer.py /path/to/Wendy.app
```

Then open `runtime/Wendy Taxi.app`. This copies your existing Wendy build,
sets its local simulation port and bundle identity, and signs the copy locally.
The generated application is ignored by Git and retains Wendy's own license.
It can run beside the original Wendy app. To use another server port, pass
`--port 8790` to both `run.sh` and `prepare_viewer.py`.

Stop the backend with Ctrl+C in its terminal. Closing a simulation in Wendy
stops physics and rendering; the HTTP server stays available until terminated.

## Simulation and autonomy

- MuJoCo 3.10 integrates five vehicles with mass, yaw inertia, actuator forces
  and collision geometry. Each vehicle has planar X, Y and yaw joints.
  A simplified dynamic bicycle model computes front and rear lateral tire
  forces, propulsion and braking. The controller writes actuator inputs only.
  It never assigns live vehicle poses, uses mocap bodies, or plays an animation.
- Planar constraints support the cars vertically. Suspension, roll, pitch,
  wheel traction contacts and wheel spin are not simulated. The visible wheels
  are rigid decoration. This is a driving-control demo, not a full vehicle model.
- The taxi uses pure pursuit on a mapped, rounded city loop. The traffic cars
  use clockwise and counterclockwise lane routes. This demo does not compute
  arbitrary street routes or overtake stopped cars.
- A 360-degree horizontal LiDAR fires 180 MuJoCo geometry rays at 10 Hz with a
  60 m range. The taxi masks its own geometry. Measured ranges in its forward
  corridor control obstacle braking. Missing or invalid scans command a stop.
- Six calibrated MuJoCo RGB cameras cover front, front-left, front-right, left,
  right and rear. Each renders at 300 × 180, up to 5 Hz at real-time
  playback. The chase camera renders at 640 × 400, up to 20 Hz. The
  cameras render occlusion and the moving scene; they are not canned images.
- Control uses known lane geometry, simulated pose and traffic-signal state.
  The STOP location comes from the map. Camera images are for inspection, not
  visual sign recognition or a learned driving policy. The displayed 30 km/h
  limit is above the controller's 19.8 km/h maximum.
- Passenger boarding and dropoff are timed mission events after measured
  speed falls below 0.18 m/s near a stop. People and buildings are simple
  procedural geometry. There is no pedestrian crossing behavior.

The native viewer receives the actual MuJoCo geometry poses over the local
HTTP/WebSocket bridge. The dashboard and viewer share one 100 Hz physics worker.
A separate process renders the cameras from bounded pose snapshots, so a slow
frame cannot stall vehicle control. Camera images stream over one WebSocket.
Fixed route geometry, stop lines and static native-viewer poses are cached.
The status API reports measured real-time factor, camera frame rates and image
ages. Rendering drops frames under load instead of building a delayed queue. The HTTP server binds only to localhost.

The minimap reads the same road, building and tree coordinates as the MuJoCo
scene. Trees have at least 4.7 m of clearance between their canopy footprint
and the road edge. Traffic signals face approaching traffic, and taxi bays
align with the road.

## Verify

```sh
.venv/bin/python verify.py
.venv/bin/python verify_layout.py
```

The verifier advances five minutes of MuJoCo physics, checks completed trips,
stationary STOP-sign and red-light waits, crossings while the signal is green,
collision count, finite state and maximum lane deviation. It also parks a traffic
car in the initial state, checks LiDAR braking and resumption, suppresses scans
to test stale-sensor stopping, and tests the emergency brake. With `mj_step`
disabled, controller calls must leave all vehicle poses unchanged.

Results are written to `runtime/verification.json`. Camera rendering and the
native bridge can be checked against the running server with
`.venv/bin/python verify_live.py`, which resets the demo after checking pause,
single-step, resume, geometry and all camera images. Results go to
`runtime/live-verification.json`. No physical car
or WendyOS device deployment is tested by this demo.

All city and vehicle geometry is generated by `world.py`; no external mesh,
trained model or proprietary simulation runtime is required for the backend.
