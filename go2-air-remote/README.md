# Go2 web remote

A phone-friendly page with hold-to-move buttons and the robot's camera. It is a
plain ROS 2 app: it publishes `/cmd_vel` and `/api/sport/request` and shows
`/camera/color/image_raw`. This is the same contract the WendyOS Go2 simulator
and the WendyOS Go2 ROS 2 bridge (`go/ros2/go2-bridge` in WendyOS) provide, so
it drives either one. It needs no robot key, serial or agent access. Like the
simulator examples it declares ROS 2 scope `peripherals` on domain 0, so it
reaches the bridge on the device and nothing on the LAN reaches the robot.

```
Browser ──WebSocket / MJPEG──▶ this app ──ROS 2──▶ Go2 bridge ──▶ Wendy agent ──▶ Go2
```

- Hold ▲ ▼ ⟲ ⟳ to move; letting go stops the robot. Forward speed is
  `GO2_MAX_SPEED` (default 0.55 m/s; the physical Go2 ignores forward requests
  below about 0.5 m/s) and turning `GO2_MAX_YAW` (default 0.6 rad/s).
- The app publishes zero velocity when no command arrives for 0.4 s, when the page
  disconnects or is backgrounded, and when STOP is pressed. On a real Go2 the
  agent also stops any move after 500 ms unless it is refreshed.
- The Stand up, Lie down and Hello buttons send sport requests 1006, 1005 and 1016.
- The camera is shown as MJPEG at up to 10 frames per second and 640 px wide. The
  app subscribes to the camera only while the page is open.

This is a demo, not a safety system. Keep the vendor remote within reach.

## Run

On a device running the Go2 bridge, or on a Go2 simulator VM:

```sh
export GO2_REMOTE_TOKEN="$(openssl rand -hex 6)"; echo "page key: $GO2_REMOTE_TOKEN"
wendy run --detach --env GO2_REMOTE_TOKEN="$GO2_REMOTE_TOKEN"
```

Open `http://<device-ip>:8883` and enter the key. The status turns green once
`/lowstate` messages arrive.

The image builds the `unitree_api`/`unitree_go` message types with
`tools/prepare_unitree_msgs.py`, the same pinned and patched sources the
simulator and bridge use.
