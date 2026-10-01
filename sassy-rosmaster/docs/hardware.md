# Running Sassy ROSMASTER on a real R2

The real adapter controls the ROSMASTER R2 through the existing Wendy RC-car
motion service. It consumes real ROS 2 lidar and odometry. The included RGB-D
bridge runs the same local person detector and depth localization used by the
MuJoCo camera. Navigation, mapping and personality code are shared with the
simulator.

The adapter and failure handling have automated tests with synthetic ROS
messages and a fake motion API. This project has not moved a physical car,
verified its sensor drivers, or tested ROS discovery and camera inference on
an R2. The physical ROSMASTER is not available yet. Prepare the configuration
copies now and leave their confirmation flags false until the hardware arrives.
The hardware configuration contains placeholder measurements and refuses to
start until the operator confirms the measurements and stop path.

## Required interfaces

| Interface | Required behavior |
| --- | --- |
| R2 motion API | The genuine `templates/python/rc-car/motion` service, with its STM32 serial controller and PWM steering servo. `/health`, `/drive` and `/stop` must work. |
| Lidar | `sensor_msgs/msg/LaserScan` on the configured `scan_topic`. At least 90 valid finite rays cover 360 degrees. Scan geometry and capture timestamps must be valid. |
| Odometry | `nav_msgs/msg/Odometry` on the configured `odom_topic`. The header frame is the configured odometry frame and the child frame is the vehicle body frame. Position and quaternion must be finite and planar. |
| Physical e-stop | A power or motor-disable circuit independent of this app. A bridge publishes its state as `std_msgs/msg/Bool` on `estop_topic` at 10 Hz or faster. `true` means stop. The bridge must stop publishing if it loses contact with the physical input. |
| RGB-D camera | Rectified `sensor_msgs/msg/Image` color, depth registered to those same pixels, and calibrated `sensor_msgs/msg/CameraInfo`, all in the configured optical frame. |
| Person perception | The included `perception_ros` bridge publishes `std_msgs/msg/String` JSON on `people_topic`. The example hardware config requires a fresh heartbeat even when its list of people is empty. |

Lidar supplies obstacle clearance. Person-specific honking and speech use the
local visual detector on actual RGB pixels and measured aligned depth, both in
simulation and through the real-camera bridge. Simulator actor positions and
geometry labels never enter that detector. Set `require_people` to `false` and
`people_topic` to an empty string to operate with ordinary obstacle avoidance
and mapping only.

The scan must use the vehicle body origin with its X axis pointing forward and
its Y axis pointing left. Reproject an offset or rotated lidar upstream using
the calibrated transform and publish `/scan_base` in `base_link`. The adapter
rejects a scan in an unrelated `laser` frame. It does not invent a transform.

ROS sensor timestamps must use synchronized system clocks. Capture age must
stay below 0.3 seconds. The adapter pairs a scan with odometry within 0.1
seconds using the scan's midpoint when `time_increment` is supplied. Its
occupancy map uses odometry and has no loop closure. Odometry drift will move
the map. It is suitable for a small room experiment, not production SLAM.

The strict scan policy rejects NaN, infinity, missing rays and readings outside
the sensor's declared range. A preprocessing bridge may publish a finite
`range_max` for a documented valid no-return measurement. It must preserve
unknown or invalid measurements as invalid. Do not turn missing coverage into
free space.

## Configure the chassis and driver

Copy `hardware.example.json` to `hardware.json`. Measure the wheelbase, overall
length and width in meters. The example dimensions are placeholders rather
than an R2 specification. Measure the steering angle at the wheels for the
motion service's full normalized steering input and set
`steering_full_scale_rad`. Verify its direction with the wheels lifted. The
local R2 service documents negative normalized steering as left; this app's
positive steering angle means left, so its default `rosmaster_steer_sign` is
`-1`. Change the sign only after checking the actual installation.

Use these settings in the existing RC-car motion service:

```text
MAX_SPEED=0.3
WATCHDOG_S=0.15
```

The stock motion service defaults to a 0.6-second watchdog. That is too long
for this adapter's configured stop envelope. Its watchdog checks every half
watchdog interval, so `WATCHDOG_S=0.15` gives a nominal upper stop delay of
0.225 seconds, plus serial and scheduling delay. Test that stop path on the
actual driver and controller. Set `driver_command_timeout` to the verified
bound, which must not exceed 0.3 seconds. A process crash, broken radio or DDS
failure can prevent this app from publishing zero; the driver's watchdog and
the physical e-stop remain necessary.

Match `motion_api_max_speed` to the driver's reported `/health` value. The
adapter checks its connected state and speed scale before sending motion.
It converts speed in meters per second to `/drive` throttle and wheel angle
in radians to the calibrated normalized steering input. It calls `/stop` for
a zero command. The R2's real steering path is the PWM servo. This adapter does
not assume that `set_car_motion` yaw or an unconsumed ROS topic can steer it.

Set `geometry_confirmed`, `steering_calibration_confirmed` and
`driver_watchdog_confirmed` to `true` only after those checks. The configured
speed cannot exceed 0.3 m/s and steering cannot exceed 0.55 radians. The planner
must use the same wheelbase, footprint and steering limit as the driver.

Only one app may command the car. Stop the RC browser, gamepad and any other
motion producer before using autonomous roaming. Keep the driver serial node
assigned only to the motion service.

## Configure the RGB-D bridge

Copy `perception.example.json` to `perception.json`. Its topic names are
placeholders, its measured transform is absent, and all calibration confirmation
flags are false. The bridge rejects that file until it has a verified camera
configuration. It does not guess calibration or register raw depth itself.

Set `rgb_topic`, `depth_topic` and `camera_info_topic` to the actual camera
driver outputs. Color must be `rgb8` or `bgr8`. Depth must be aligned axial
optical Z in `32FC1` metres or `16UC1` millimetres. RGB, depth and CameraInfo
must have identical dimensions and the same `optical_frame` in their headers.
An aligned-depth topic name alone does not establish that its pixels match the
RGB image. Verify registration against real objects at different distances.

CameraInfo must describe the delivered rectified pixels through its `P`
matrix. The bridge accepts identity `R`, zero stereo projection offset, and
calibration expressed for the actual image crop and binning. Recalibrate or
re-express unsupported crops upstream. It rejects unknown focal lengths,
incorrect image dimensions and unsupported rectification.

Set `camera_to_base` to the measured, row-major 4 by 4 rigid transform in
metres. It maps `[optical_right, optical_down, optical_forward, 1]` to
`[base_forward, base_left, base_up, 1]`. Include the camera's measured position
and orientation relative to `base_frame`; an approximate mounting diagram is
not a calibration. This bridge supports a fixed camera mount. A moving camera
needs a capture-time transform implementation before it can use this path.

After those checks, set `calibration_verified`, `rgb_rectified`,
`depth_registered_to_rgb` and `fixed_camera_mount` to `true`. Match `base_frame`
and `people_topic` to `hardware.json`. Set `use_sim_time` to `false` for the
physical camera and synchronized system clocks. Use `true` only when every
source and consumer uses the same ROS simulated clock.

The example permits 25 milliseconds between RGB, depth and CameraInfo captures.
`camera_info_static` accepts fixed calibration whose timestamp is zero or does
not follow each image, only after the operator explicitly selects that contract.
The current subscriptions use best-effort, volatile QoS. CameraInfo must arrive
after the bridge starts; a historical latched sample alone will not arrive.
Configure the driver to publish calibration with its images, or start the bridge
before that calibration is published.

Rendering is specific to simulation. On hardware, ROS callbacks decode the real
camera images and a separate worker runs the bundled local detector, samples
valid torso depth and applies the measured optical-to-base transform. There is
no cloud image upload. Color detections without usable depth do not receive
invented positions. Lidar braking remains independent of visual inference.

## Run the bridge while the rover is parked

The bridge is a separate process and publishes no drive commands. In the ROS
hardware image, run it with the ROS environment sourced:

```sh
. /opt/ros/jazzy/setup.sh
/opt/rover/bin/python -m sassy_rosmaster.perception_ros --config /config/perception.json
```

For a separate container on the Linux ROS host, build the same image and mount
the measured configuration read-only. Set `ROS_DOMAIN_ID` and
`RMW_IMPLEMENTATION` in the shell to the robot's verified values before running
these commands:

```sh
docker build -f Dockerfile.ROS2 -t sassy-rosmaster-ros2 .
docker run --rm --network host \
  --env ROS_DOMAIN_ID --env RMW_IMPLEMENTATION \
  --volume "$PWD/perception.json:/config/perception.json:ro" \
  sassy-rosmaster-ros2 \
  /opt/rover/bin/python -m sassy_rosmaster.perception_ros \
  --config /config/perception.json
```

The ROS image entrypoint sources ROS before Python starts. Both the bridge
and rover app must share the measured ROS domain and compatible discovery
settings. The detector and its model are already in this image. Keep the rover
app parked, and do not change the chassis, steering or watchdog confirmation
flags merely to test perception.

Inspect `/sassy_rosmaster/people`, or the configured replacement topic, with
`wendy device ros2 echo /sassy_rosmaster/people --count 3`. Check that timestamps
advance, a visible person's base coordinates follow actual measured range,
and fresh frames without people contain an empty array. Interrupt the aligned
depth feed and stop the camera driver to check the reported errors and lost
heartbeat while motion remains disabled.

The capture-to-publication age must remain within `max_age`, at most 0.3 seconds,
including synchronization, CPU inference and publishing. The bridge preserves
the original RGB capture time. It does not replace stale results with a new
timestamp. Missing images, calibration errors or expired inference produce no
fresh heartbeat and therefore stop a rover configured with `require_people`.
Measure this budget on the actual R2 compute device before any motion test.
Simulation CPU timings do not establish hardware camera performance.

## Person bridge format

The perception bridge supplies each detected person's measured ground-plane
position in meters relative to the vehicle body. Use a depth camera or another
range source. A bounding box from a color image does not supply a distance.

```json
{
  "stamp": 1790812800.125,
  "frame_id": "base_link",
  "people": [{"x": 1.2, "y": 0.1}]
}
```

`stamp` is the original ROS RGB capture time in seconds. The included bridge
publishes each completed fresh capture once, targeting a 0.1-second inference
interval, and uses `"people": []` when it has no grounded person detections.
It includes detector confidence and a local tracking ID as extra fields. Do not
repeat an old capture timestamp as a new detection. The adapter derives
distance, bearing and world coordinates from these measurements. A missing,
stale or malformed required perception heartbeat stops the car.

## Build and deploy through WendyOS

Use `Dockerfile.ROS2` for the hardware app. It contains ROS Jazzy, its generated
message packages, CycloneDDS, `espeak-ng` and ALSA tools. The regular Dockerfile
is the simulator image and does not install ROS. Verify the target's existing
ROS distribution and middleware before building; change the ROS image and
package names together if it requires a different supported distribution.

The app must join the measured host ROS domain. Retain the ROS framework block
and host-network entitlement in its hardware `wendy.json`:

```json
{
  "frameworks": {
    "ros2": {
      "distro": "jazzy",
      "rmw": "rmw_cyclonedds_cpp",
      "domainId": 0,
      "discoveryScope": "host"
    }
  },
  "entitlements": [{"type": "network", "mode": "host"}]
}
```

Domain 0 and CycloneDDS are examples. Replace them with the robot's verified
settings. The RC motion service keeps its separate serial entitlements. Add
the audio entitlement to this app when using a robot speaker.

Pass these environment variables to the hardware app:

```text
ROVER_MODE=hardware
ROVER_CONFIG=/app/hardware.json
HOST=127.0.0.1
PORT=8096
ROBOT_AUDIO=1
```

Include the calibrated `hardware.json` in the build context. Run the container
with the ROS environment sourced; the ROS base image's entrypoint does this.
Its Python command is `/opt/rover/bin/python -m sassy_rosmaster`. The venv can
import apt-installed ROS packages and contains the pinned web dependencies.

The hardware UI binds to localhost by default. Forward port 8096 through SSH
when opening it from another computer. Restrict access to the operator. The
HTTP Start control is not authenticated. The simulator's public bind address
is a different setting from the hardware default.

Inspect the actual topics before starting:

```sh
wendy device ros2 topics --all
wendy device ros2 topic info /scan_base
wendy device ros2 echo /scan_base --count 1
wendy device ros2 echo /odom --count 1
wendy device ros2 echo /sassy_rosmaster/estop --count 2
```

Confirm the types, frames, capture age, clock alignment and complete coverage
from messages. Topic names alone do not prove that sensors work.

## Start and stop behavior

The app starts parked. After startup and every fault, valid readings do not
restart motion. Release the physical e-stop, restore fresh sensor inputs,
press Reset and then explicitly press Start. Reset checks the sensors and
sends zero; it keeps motion disabled. A physical or UI e-stop latches until
that reset. A missing e-stop heartbeat stops motion.

The adapter's separate watchdog checks every 20 milliseconds. It stops when
controller commands are absent for 0.2 seconds, sensor captures are stale,
odometry cannot pair with lidar, perception fails, or ROS/HTTP transport fails.
It disarms when a hardware run reaches 120 seconds or 10 meters of accumulated
odometry travel. The operator can choose lower bounds in the configuration.
Pause, faults and shutdown send a zero command. A fault requires Reset and a
new explicit Start after the cause is fixed.

For a first physical test, lift the driven wheels and verify steering, Stop,
e-stop and loss of controller heartbeat. Then use a supervised, clear indoor
area with a short duration and low speed. Keep access to the physical power
cutoff. The car brakes for people; its honk and speech are personality effects
and do not establish that a person has moved away.

The small-car adapter is for supervised indoor experiments. It does not support
public-road driving or unattended operation around people.

## Message definitions

The transport follows the upstream [ROS LaserScan definition](https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs/msg/LaserScan.msg)
and [AckermannDriveStamped definition](https://github.com/ros-drivers/ackermann_msgs/blob/ros2/msg/AckermannDriveStamped.msg).
For a ROS-controlled Ackermann base with a verified command consumer, set
`motion_backend` to `ros2_ackermann` and point `command_topic` at that driver's
actual `ackermann_msgs/msg/AckermannDriveStamped` subscription. It still needs
an independent command timeout and the same calibrated sensors and stop path.
