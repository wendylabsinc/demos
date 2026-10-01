# Warehouse localization

A ROS 2 Jazzy app that turns odometry and LiDAR into mapping or a validated
AMCL pose. It publishes no robot-motion commands. The
[warehouse camera demo](../warehouse-demo/README.md) consumes its HTTP pose
endpoint and waits when observations are missing, stale or uncertain.

`LOCALIZATION_MODE=mapping` starts SLAM Toolbox. `localization` starts Nav2
AMCL and requires a saved or surveyed occupancy map. Mapping reports
`ready=false` and does not return an AMCL map pose.

## Build and run

The Dockerfile installs ROS 2 Jazzy, Nav2, SLAM Toolbox, CycloneDDS and the
Python runtime dependencies. Build from this directory:

```sh
docker build -t wendy-warehouse-localization .
```

For a commissioned WendyOS device, `wendy.json` selects mapping mode, host DDS
domain 0 and a persistent `/data` volume. Select the intended target explicitly:

```sh
wendy run --device YOUR_DEVICE --build-type docker --detach --yes
```

The default input topics are `/utlidar/robot_odom` and `/utlidar/cloud_base`.
`ODOM_TOPIC` and `CLOUD_TOPIC` can override them. Verify the target's actual
DDS domain, middleware, sensor frames and timestamps before changing this
configuration. The defaults describe required inputs, not observed hardware.

## Simulation validation

Generate the simulation occupancy map locally. This setup also installs the
dependencies for the isolated quality checks:

```sh
python3 -m venv .venv
./.venv/bin/python -m pip install numpy Pillow PyYAML pytest
./.venv/bin/python make_sim_map.py
./.venv/bin/python -m pytest tests/test_quality.py -q
```

The generator writes `evidence/map.pgm` and `evidence/map.yaml`. These files
describe the simulated warehouse; they do not represent a real building.
Generated evidence stays outside Git and is mounted into the container at run
time. The Docker build excludes it.

Start AMCL in a separate terminal:

```sh
docker run --rm --name warehouse-amcl \
  -p 127.0.0.1:18790:8790 \
  --mount type=bind,src="$PWD/evidence",dst=/data \
  -e ALLOW_SIMULATED_INPUT=true \
  -e LOCALIZATION_MODE=localization \
  -e MAP_FILE=/data/map.yaml \
  wendy-warehouse-localization
```

Then start the [warehouse demo](../warehouse-demo/README.md#run). Its virtual
sensor adapter feeds `/measurements`, seeds an approximate initial pose and
waits for AMCL to satisfy freshness and covariance checks. The ROS container
has no MuJoCo access. `ALLOW_SIMULATED_INPUT=true` explicitly enables the test
measurement endpoint; the checked-in hardware configuration leaves it disabled.

Inspect the current status with:

```sh
curl http://127.0.0.1:18790/status
```

Stop the container with Ctrl+C in its terminal. The optional
`tests/mapping_smoke.py` requires a separate mapping-mode container with
simulated input enabled at `http://127.0.0.1:18791`, plus the warehouse demo's
Python dependencies, robot model and licensed simulator runtime. It feeds
stationary virtual measurements and checks mapping state; it does not validate
a complete room map.

## Hardware inputs and map commissioning

`INPUT_ODOM_FRAME`, `INPUT_BASE_FRAME` and `INPUT_CLOUD_FRAME` default to
`odom`, `base_link` and `base_link`. Unexpected frames produce faults in
`/status`. A cloud in another frame needs a measured transform upstream.
Output frames are `wendy_map`, `wendy_odom` and `wendy_base`.

Map under operator control after fresh input data passes the status checks.
Save the `/wendy_localization/map` topic with Nav2's `map_saver_cli`, explicitly
remapping its `map` input topic. Restart in localization mode with `MAP_FILE`
pointing to the saved YAML and its companion image. Fixed cameras must be
calibrated in `wendy_map` before their target coordinates can guide navigation.

`GET /status` returns readiness, pose, covariance and input faults. Pose remains
null until localization checks pass. `POST /initialpose` accepts an approximate
operator pose in metres and radians:

```json
{"x":0.3,"y":-0.25,"yaw":0.12,"variance":[0.25,0.25,0.09]}
```

This initializes AMCL particles. It does not supply a measured pose. Surveyed
wall alignment and position checks are still needed before enabling navigation.

[HOWTO.txt](HOWTO.txt) retains historical inspection and simulation results.
Those recorded device states do not establish current hardware readiness.
