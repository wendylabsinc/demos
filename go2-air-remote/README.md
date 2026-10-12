# Go2 Air web remote (minimal)

A phone-friendly page served from the Jetson. The Jetson talks to the Go2 directly over
Unitree's WebRTC protocol, using
[unitree_webrtc_connect](https://github.com/legion1581/unitree_webrtc_connect)
(pinned commit). No Wendy agent changes are needed.

- Hold ▲ ▼ ⟲ ⟳ to move; letting go stops the robot. Speed is capped at
  `GO2_MAX_SPEED` (default 0.3 m/s) and `GO2_MAX_YAW` (default 0.6 rad/s).
- The server sends StopMove when no command arrives for 0.4 s, when the page
  disconnects or is backgrounded, and when STOP is pressed. Only the most recent
  page can control the robot.
- The Stand up, Lie down and Hello buttons send single sport commands.

This is a demo, not a safety system. Keep the vendor remote within reach.

## Run

The Go2 accepts **one** WebRTC client at a time, so close the Unitree app and any other
client (including a Wendy agent pairing) before starting this.

```sh
wendy run --detach \
  --env GO2_IP=192.168.100.100 \
  --env GO2_AES_KEY="$GO2_AES_KEY" \
  --env GO2_REMOTE_TOKEN="$GO2_REMOTE_TOKEN"
```

Then open `http://<jetson-ip>:8883` and enter the access key.

- `GO2_AES_KEY` is the robot's per-device key (32 hex characters). Firmware 1.1.15 and
  later require it; older firmware ignores it.
- `GO2_REMOTE_TOKEN` is any random string; the page asks for it.
