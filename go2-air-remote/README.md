# Go2 Air web remote (minimal)

A phone-friendly page served from the Jetson. Robot commands go through the Wendy
agent's existing Go2 pairing, so the agent keeps the robot connection and its key.
The app only needs the robot's serial.

```
Browser ──WebSocket──▶ this app ──local agent socket──▶ Wendy agent ──WebRTC──▶ Go2
```

- Hold ▲ ▼ ⟲ ⟳ to move; letting go stops the robot. Speed is capped at
  `GO2_MAX_SPEED` (default and agent maximum 0.3 m/s) and `GO2_MAX_YAW` (default and
  agent maximum 0.6 rad/s).
- The app sends Stop when no command arrives for 0.4 s, when the page disconnects or
  is backgrounded, and when STOP is pressed. The agent also stops any move after
  500 ms unless it is refreshed, so the robot stops if this app dies.
- The Stand up, Lie down and Hello buttons send single commands.

This is a demo, not a safety system. Keep the vendor remote within reach.

## Requirements

- An agent with `WendyUnitreeTeleopService.Command` (WendyOS #2242), with the Go2 paired
  through `wendy device pair --unitree`.
- The `admin` entitlement in `wendy.json`. It mounts the agent socket
  (`WENDY_AGENT_SOCKET`), which gives this app full control of the agent, so treat it as
  trusted code.

## Run

```sh
wendy run --detach --env GO2_SERIAL=<robot serial> --env GO2_REMOTE_TOKEN="$GO2_REMOTE_TOKEN"
```

Then open `http://<jetson-ip>:8883` and enter the `GO2_REMOTE_TOKEN` key.

`*_pb2*.py` are generated with grpcio-tools 1.84.0 from the copies of the WendyOS
protos in `proto/`:

```sh
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/*.proto
```
