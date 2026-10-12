# Go2 Air web remote control (experimental)

Phone → existing Wendy/private tunnel → Jetson web bridge → **already-paired
agent WebRTC connection** → Go2. No ROS/DDS, AI, autonomy, camera, vendor login,
robot keys, or second robot connection. The browser never connects to Unitree.

The default backend is **fake**. Page loads, login and health checks never
open robot control or send a robot command. Hardware control needs both an
operator-enabled agent and this server's explicit `--allow-motion` flag;
clicking **Enable supervised control** then attempts an initial StopMove.
Even Enable is therefore an active hardware action, not a passive health check.
This demo and Air firmware support are **not physically commissioned**.

## Local fake preview

Needs Go 1.27.2 (the Go launcher can fetch the pinned toolchain), and Node 24+
only for browser-logic tests. No Python dependencies or external UI assets.

```sh
cd go2-teleop
export GO2_TELEOP_TOKEN="$(openssl rand -hex 32)"
# Display this temporary key privately; do not paste it into logs or tickets.
printf '%s\n' "$GO2_TELEOP_TOKEN"
go run . --secure-cookie=false
```

Open <http://127.0.0.1:8882>, enter the key, connect, then enable the fake
controls. Hold Forward/Back/Turn; release sends zero velocity. **Stop / Disable**
ends the control connection. Reload/reconnect/backgrounding never auto-enables.
The access key is not retained in browser storage or URL parameters. Login uses
an HttpOnly, SameSite=Strict, process-local session cookie and same-origin checks.
Restart the server to invalidate sessions; change the key to revoke access.
Use HTTPS by default. `--secure-cookie=false` is only for localhost or an
explicitly approved encrypted private mesh—not an unprotected LAN/public HTTP
service. Do not expose this server publicly or log login bodies/cookies.

Build without launching a server or touching hardware:

```sh
CGO_ENABLED=0 GOOS=linux GOARCH=arm64 go build -mod=readonly -o /absolute/shared/path/go2-teleop-linux-arm64 .
```

A Wendy container build is provided. Default `wendy.json` grants only host
networking and still uses the fake backend. Provision a random `GO2_TELEOP_TOKEN`
privately in the runtime environment; no key is checked in. A host executable can
instead use `--token-file /private/path/key` (regular file, mode 0600, 32+ random
characters). Do not pass secret values in publicly visible command examples.

## Hardware deployment — separate approval required

Before any deployment or hardware Enable, agree on the **exact** Jetson/device
selector and physical robot with the owner, verify successful agent pairing,
exclusive robot control (including vendor apps), direct sight, a supervised
physical stop and a bounded flat clear area. A reported serial/hostname alone
is not physical identity. Do not replace somebody else's active session.
First validate the actual iPhone/tunnel path with the **fake backend**.

The required WendyOS agent adds `WendyUnitreeTeleopService.Control`, and must be
started with `WENDY_UNITREE_TELEOP=1` after separate operator approval. Existing
pairing RPCs/configuration/keys are unchanged. Older agents cannot run this
hardware backend. A direct binary update uses `device push-agent`, not an OS OTA
or SSH; installing/restarting the agent and changing its configuration are
separately approved operations, not part of a build.

After approval, `wendy.hardware.example.json` illustrates the **admin** entitlement
needed for `WENDY_AGENT_SOCKET`. This grants **the full protected agent admin
socket**, not only robot motion: treat the bridge/container as trusted admin
code. Do not casually grant it to other apps. The hardware bridge uses only this
private Unix socket (no unauthenticated TCP fallback) and requires an exact
`--serial`. The host socket is normally
`/var/lib/wendy/agent-control/agent.sock`; inside an approved Wendy admin container
`WENDY_AGENT_SOCKET` is `/run/wendy/agent/agent.sock`.

Hardware runtime arguments, supplied only after commissioning approval:

```text
--listen <approved-private-interface>:8882 --allow-motion --serial <owner-confirmed-serial>
```

Keep HTTPS cookie defaults unless the explicitly approved private tunnel needs
HTTP inside its encrypted transport. Wendy supports `--user-args` for container
arguments. The Docker default binds `0.0.0.0:8882`; restrict the interface/firewall
and verify access is private before deployment. No deployment command is run by
the tests. No automatic mode change or stand-up is performed; if normal vendor
manual controls/Stop are not already known to work in the selected mode, do not
try to make them work by changing modes blindly.

## Limits and failure behavior

- Exactly one browser controller, and one agent control stream per paired serial.
- Fixed requested bounds: forward ±0.15 m/s, yaw ±0.3 rad/s; no strafe, pose, gait
  or arbitrary robot API. These are requested velocities, not measured motion.
- Each agent response supplies a single-use challenge. Reply delivery and the
  next input share 350 ms; robot acknowledgement has a 200 ms budget. Browser
  input refreshes about every 100 ms with one request in flight, not a move queue.
- Release sends zero/StopMove. Stop, stale/invalid input, timeout, page blur,
  backgrounding, network loss, robot errors and pairing replacement terminate
  control and attempt StopMove. A new explicit Enable is required.
- Acknowledgements are protocol results, **not proof of physical stopping or
  collision clearance**. The page shows disabled/unverified, never “robot safe”.
- This is a host-side watchdog, **not a safety-rated emergency stop**. Host crash,
  radio/link loss, scheduling stalls or robot firmware faults can prevent Stop
  reaching the robot. Robot-side watchdog behavior is unverified; movement can
  outlive the host-side timeout. The physical stop and direct supervision remain
  mandatory. There is no obstacle, fall, stairs, drop-off or people avoidance.

The first real trial, after explicit live-motion approval, should be one short
supervised input/release/Stop, not an unattended play session. Air command/Stop
acceptance, physical stopping latency, robot-side timeout and the real iOS tunnel
remain unverified until that trial. Camera/standing/modes/skills are out of scope.

## Checks

```sh
go test -mod=readonly -race ./...
go vet -mod=readonly ./...
node --test tests/browser.test.cjs
```

Go tests exercise the HTTP/WebSocket bridge with fake controllers: auth/origin,
no startup/page-load motion, hold/release/Stop, exclusive control, stale/invalid
inputs, timeout/disconnect and manual re-enable. Node uses deterministic
DOM/socket/timer fakes for release-in-flight, blur/background, cancellation,
Stop, watchdog and ignored late responses; it is **not** an actual Safari/iPhone
test. Agent tests use local Pion and gRPC fixture peers, never vendor or robot
connections. Docker and real hardware/tunnel integration require separate checks.

## Protocol source and licensing

`proto/wendy/agent/services/v2/unitree_teleop_service.proto` is an exact snapshot
of the WendyOS contract; generated Go bindings override its Go import path so
this small demo does not import the entire agent module. The snapshot/bindings
retain WendyOS Apache-2.0 terms and NOTICE under `internal/agentpb/`; original
demo code is MIT. See [protocol provenance](internal/agentpb/SOURCE.md).
