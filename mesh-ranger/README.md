# Mesh Ranger

An offline field-station demo for two or more Orin Nano devices. Capture a real
webcam frame in the browser and send it to a discovered worker running Moondream
1.8B. Sensor scenarios also use the bundled Qwen 2.5 0.5B model. Nearby workers
share independent inference jobs. Each result names the worker and reports inference time and
GPU residency. This distributes requests; each worker holds the whole model.

The second part of the demo makes Wendy Net's mDNS service projection visible.
Every container announces `_ranger._tcp.local.` on its private interface. Other
nodes discover it without a configured peer list. The dashboard compares the
publisher's declared container address with the resolved endpoint used locally.
For example, `172.18.0.2:8090` may be projected as `10.99.x.x:18090`. Multiple
publishers can have the same private IP without clients confusing them.

The application uses resolved A/SRV records to connect. The `origin` TXT field
is only a display aid. It never overrides a resolved address. Wendy projects
admitted service records; it does not flood raw multicast between devices.
Addresses are observer-local. Do not copy one node's mesh address to another.

## Rehearse on a laptop

Use Python 3.11 or later. The three-process rehearsal needs no GPU or model:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python rehearse.py
```

Open <http://127.0.0.1:8090> and click **Send 6 events**. Responses are explicitly
labeled simulation. Peers are wired over loopback; this mode does not verify
Wendy Net, mDNS reflection, CUDA, or model quality. Ctrl-C stops the rehearsal.

Run the behavior tests with:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Deploy to Orin Nano

Use current WendyOS agents with app mDNS projection, a compatible NVIDIA driver,
and an already configured Wendy Net. The engine/model packaging comes from
[the tested NVIDIA PAIR example](https://github.com/wendylabsinc/wendyos/blob/main/Examples/NvidiaPAIR/README.md), including its pinned
Ollama image and model manifest. That example targets Orin Nano with JetPack 7.2.
An older JetPack installation needs its own compatible engine image.

Provision organization credentials and build/deploy while online. The Dockerfile
fetches and verifies the model during the build. Runtime inference and discovery
need no model downloads or external APIs. Mesh mode restricts app networking;
o Internet connection indicator is inferred from that configuration.

Find the intended devices and inspect their mesh state before deployment:

```sh
wendy discover --type lan --timeout 5s
wendy net status --device DEVICE_A
wendy net topology --device DEVICE_A
wendy net status --device DEVICE_B
wendy project validate .
NODE_NAME=gate-orin wendy run --device DEVICE_A --detach --yes
NODE_NAME=shed-orin wendy run --device DEVICE_B --detach --yes
```

Replace DEVICE_A and DEVICE_B with discovery selectors. If Wendy Net is not
configured, follow the [Wendy Net guide](https://github.com/wendylabsinc/wendyos/blob/main/go/internal/cli/assets/docs/guides/wendy-net.mdx).
Changing transports can interrupt other apps, so it is a separate setup step.
This example does not alter device routing or Internet sharing.

The app publishes TCP 18090 mapped to container port 8090. Its Ollama backend
binds only to container loopback. The vision model manifest and every downloaded
blob are SHA-256 verified by `fetch_vision_model.py` during the build. GPU warmup precedes service advertising and
can take several minutes. Startup fails if the model is missing or Ollama does
not report GPU residency. `/health` means the web app has finished startup;
a completed job is the inference check. The manifest deliberately omits a
host TCP readiness probe because mesh publication does not make the service
available at the device LAN address. Check `/health` through the preview proxy
or inside the container.

For a laptop dashboard over authenticated LAN management, run:

```sh
python3 preview.py --device DEVICE_A --port 19090
# In another terminal:
python3 preview.py --device DEVICE_B --port 19091
```

Open <http://127.0.0.1:19090> and <http://127.0.0.1:19091>. The proxy maintains one authenticated `wendy net forward` process to app port
18090, instead of starting a remote Python process for every request. Only browser
requests use this management proxy; inference between workers still uses the
discovered Wendy Net endpoints. The proxy binds to loopback and needs the Wendy
CLI on PATH. Alternatively, use a browser in a connected Wendy Net client. A private container port is not automatically a laptop
LAN endpoint. You can always inspect the app from its own container:

```sh
wendy device attach --device DEVICE_A sh.wendy.examples.mesh-ranger -- \
  python -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8090/api/state").read().decode())'
```

## Real webcam inference

Open a dashboard through `preview.py` on localhost. Click **Enable camera** and
allow camera access for the session. Select a physical camera, then click
**Capture & analyze**. The page sends the captured JPEG bytes to the station and keeps the camera
preview live. Use **View frame** on a result to inspect its exact submitted image. Local preview video is mirrored; the submitted frame
keeps the camera's original orientation.

The browser scales and compresses a frame to fit 32,000 base64 characters. The
server validates the JPEG, routes it through the same discovered worker pool,
and passes it in Ollama's `images` input to `moondream:1.8b`. The result includes
the serving node, model, GPU residency, processing time, and SHA-256 of the exact
JPEG. A response with a missing or different frame hash is rejected.

Capture twice to exercise both workers when the fleet is idle. A single request
can stay local. The result must name a different worker and a resolved mesh
endpoint to establish remote image inference.

Camera permission starts a local preview only. Frames are sent when you click
**Capture & analyze**, or continuously after **Start live analysis**. **Off** stops the camera; closing the page stops its tracks.
Frames remain in bounded memory history on the participating station and worker.
They are not written to the persistent volume, and neither pixels nor vision
descriptions are printed to telemetry logs. Model initialization uses a blank
frame that is never shown as a captured image or counted as a user job.

The two Orins do not need camera entitlements for this browser capture path.
The webcam connects to the browser's computer. Device-attached V4L2 capture is
not implemented. Camera access requires localhost or HTTPS.

## ESP32 SensorLink camera

The dashboard also accepts an actual MJPEG camera channel from Wendy Lite.
Discover the board and inspect its advertised video channels first:

```sh
wendy discover --json --type all --timeout 5s
wendy --json device camera list --device wendy-lite:YOUR_CAMERA.local
python3 preview.py --device DEVICE_A --port 19090 \
  --sensorlink-device wendy-lite:YOUR_CAMERA.local
```

Open `http://127.0.0.1:19090`, click **Find ESP32 camera**, then choose its
advertised channel under **Camera source**. Selection starts its live preview.
Click **Capture & analyze** or **Start live analysis** to send real frames to
the fleet. Select **Browser webcam** to switch back. **Off** stops the active
camera subscription. Discovery does not capture frames or change pairing.

`preview.py` runs one `wendy device camera view --stdout --non-interactive`
subscription for the selected channel. The CLI handles the Wendy Lite connection
and SensorLink protocol. JPEG bytes reach the local browser through the loopback
preview server, then sampled frames enter the station's existing `/api/events`
endpoint. The mesh diagram labels ESP32 ingress and shows the actual worker's
request and return route. This is an application data-flow diagram, not a map
of physical radio hops. The ESP32 supplies images; inference runs on the GPU
workers.

The bridge retains only the newest frame, accepts at most 2 MiB per JPEG, and
refuses frames older than two seconds. It ends the subscription after 15 seconds
without viewer polling. The browser uses decoded image dimensions, even if the
manifest advertises another size, and applies the same 512-pixel starting width
and 32,000-character base64 budget as webcam capture. A source frame is submitted
at most once during continuous sampling. Worker capacity still bounds outstanding
jobs. The displayed result hash proves the resized JPEG submitted to inference,
not the larger original camera JPEG. No camera images are saved to disk.

The camera preview stays live when inference pauses or the station disconnects.
A missing camera frame disables new captures without marking the GPU station as
disconnected. If the board reports **camera is busy**, stop its other viewer or
release its existing gateway subscription before selecting it again. This demo
does not alter pairing or take over another camera consumer. H.264 SensorLink
channels and paired gateway capture are not supported by this MJPEG bridge.

Bridge tests cover fragmented JPEG input, memory bounds, actual manifest
selection, stale-frame rejection, process cleanup, and the busy-camera error.
UI checks cover frame provenance, duplicate sampling, stale-frame rejection, and
preview continuity during a station outage. These tests use controlled fixtures;
live ESP32-to-GPU inference needs a connected camera and running fleet.

Startup warms both models sequentially before advertising the worker, releasing
the text model before loading the vision model. The first vision warmup
may take several minutes; subsequent starts can reuse the CUDA cache. The
`mesh-ranger-v2` TXT marker keeps old text-only workers out of the vision fleet.

## Live analysis and visible data flow

Enable the camera, then click **Start live analysis**. The preview stays live.
The browser samples up to eight frames per second when capacity allows and caps
outstanding jobs at the discovered worker count, up to eight. It captures a fresh
frame when a slot becomes available instead of building an old-frame backlog.
The displayed results-per-second value counts actual completions over the last
ten seconds. Camera FPS and inference throughput are different; this captioning
model does not analyze every video frame.

The coordinator reserves a worker while its request is in flight. New jobs skip
those reservations and try another discovered endpoint. Reservations are local
to each coordinator, so requests from another station can still cause a busy
response and retry.

The mesh diagram draws cyan requests toward their selected worker and green
results back toward this station. Returned paths animate for seven seconds; the
last completed route remains highlighted afterward. A local inference pulses
at this station. These are application request routes, not physical radio hops
or packet-level telemetry. Remote "in flight" means an inference HTTP request is
outstanding, not a separately measured remote GPU state.

`/api/state` includes each job's current `flow` and bounded `flow_history`, with
phase, endpoint, node, and timestamp. Retries stay visible and do not count as
successful results. **Pause analysis** stops sampling while the live preview and
submitted jobs continue. **View frame** pauses sampling to inspect a past image;
**Back to live** resumes preview. **Off** stops both sampling and the camera.

## Five-minute demonstration

1. Open both dashboards. Show each private announcement and the other node's
   resolved `10.99.*:18090` service. The activity panel shows added/updated events.
2. Capture and analyze two webcam frames on A, or send six sensor events. Show completed results attributed to both
   workers, with GPU residency. Discovery alone is not evidence of inference.
3. On B, click **Withdraw this announcement**. Watch A's discovery log report
   removal and its worker count drop. DNS and mesh leases can delay removal.
   Submit another batch on A after removal; B is no longer a candidate there.
4. Reannounce B. Watch it return through DNS-SD without editing a peer list.
   Submit another batch and check that both workers serve jobs again.
5. With an independently accessible local network, remove only the Internet
   uplink and repeat. Keep the LAN/Wendy Net links intact. This is the physical
   offline proof; a UI label or successful online request cannot establish it.

For a separate abrupt-failure demonstration, stop only this demo's worker on B
or disconnect its mesh carrier. A may retain its service briefly; connection
failure causes a retry on another worker. Reconnect and allow discovery to
converge. Withdrawal itself is a discovery demonstration, not a simulated crash.

## Verify a larger fleet

The same image can run on additional compatible NVIDIA Jetsons. Every worker
keeps a complete copy of each model; the fleet distributes whole requests.
It does not combine device memory or split one inference across GPUs.

Start another `preview.py` process for each device, using separate loopback
ports such as 19092 and 19093. Wait for every station to discover the other
workers. Then check the expected fleet size from station A:

```sh
python3 verify_fleet.py --workers 4
```

To verify image inference, capture a webcam frame on A, take its `event_id` from
`/api/state`, and reuse those exact bytes across the fleet:

```sh
python3 verify_fleet.py --workers 4 --frame-event EVENT_ID
```

The verifier waits for each job before submitting the next, requires a result
from every discovered worker, and checks model, GPU residency, and frame hash.
It prints metadata only. Increase `--workers` when adding more devices.

## Wendy Lite sensor input

Wendy Lite is optional. The complete demo can run from dashboard-generated
measurements. `sensor_bridge.py` accepts newline-delimited JSON from a sensor
adapter and submits it to a Ranger station. It assigns an event ID once and
reuses it across retries:

```sh
printf '%s\n' '{"source":"lite-gate","zone":"north gate","motion":true}' | \
  python3 sensor_bridge.py --url http://127.0.0.1:8090
```

Run the bridge next to the station or through a verified port forward. A Lite
PIR or reed-switch app can provide these records through a gateway adapter;
board firmware and the transport adapter are not included. Do not assume a
Wendy Lite board's management USB stream is a serial JSON feed. The integration
contract is `POST /api/events`:

```json
{
  "event_id": "gate-boot7-42",
  "source": "lite-gate",
  "zone": "north gate",
  "motion": true,
  "door_open": false,
  "temperature_c": 22.0
}
```

`temperature_c` is optional. Send actual calibrated measurements. The model's
explanation is a suggestion and stays separate from those measurements.

## Behavior and limits

- The same image runs on every worker. Each station has four dispatch consumers,
  a 32-event queue, one inference slot, and bounded in-memory event/result history.
- Round-robin dispatch tries local and discovered workers. Busy or unreachable
  workers enter a five-second cooldown; jobs retry the fleet up to three times.
  `/api/infer` never forwards, preventing forwarding loops.
- Worker timeout is 180 seconds per request. During a partition a worker may
  finish after the caller times out. Retries can duplicate computation across
  workers; there is no exactly-once execution guarantee.
- Event IDs deduplicate while retained in a station's memory. Reusing an ID with
  different measurements is rejected. Queues and results do not survive restarts.
  Only node identity and CUDA caches use the persistent volume.
- Discovery names/TXT values are not application authentication. This demo trusts
  its admitted mesh peers and uses HTTP within Wendy's transport. It has no
  public-facing authentication and should remain within the demo network.
- Simulation peers are filtered out of real inference discovery. A CPU-only
  engine test can explicitly set `REQUIRE_GPU=0`; deployed defaults require GPU.
- The small model is for demonstrating distributed execution, not dependable
  incident classification. Moondream descriptions can also be wrong; inspect the submitted frame alongside the response.

The DNS-SD code follows [Wendy's service guide](../../docs/guides/wendy-net-services.mdx)
and the [python-zeroconf API](https://python-zeroconf.readthedocs.io/en/latest/api.html).

## Verified on hardware

On 2026-11-10, this example ran on Ci Mesh 1 and Ci Mesh 2, both Orin Nano
workers. Six real inference requests submitted to A completed with three served
by each node. Worker inference times were 573–847 ms; both reported 456,382,217
bytes of GPU residency. These are sample timings, not a controlled benchmark.

Both containers announced `10.81.219.18:8090`. A discovered B at
`10.99.0.2:18090`; B discovered A at `10.99.0.1:18090`. The test withdrew B's
announcement, observed its removal on A, reannounced B, and observed its return.
It restored B's announcement before exiting. No manual peer URLs were configured.

To repeat the check with both preview proxies running:

```sh
python3 verify_fleet.py --exercise-discovery
```

Eight initial behavior tests and a local real DNS-SD add/remove/reannounce check also
passed. The local three-worker simulation completed six jobs, two per worker.
That initial text-only run did not verify physical WAN disconnection, multi-hop
radio routing, physical sensor input, or browser layout. The small model did invent an absent temperature in one
initial test, so treat its prose as a suggestion, not a measurement.

### Webcam vision verification

The vision revision was also tested on 2026-11-10 using the physical MacBook Pro
camera through Chrome. A browser-captured JPEG completed on A in 8,749 ms. The
same frame was resubmitted to A and dispatched to B through its discovered
`http://10.99.0.2:18090` endpoint, completing in 4,309 ms. Both results used
`moondream:1.8b` and returned SHA-256
`4b14079b386124c3ddde7bccc9471c9be8358deaf70a7d6c3e70f855b48cb911`,
independently checked against the captured JPEG bytes. A reported 890,975,027
GPU-resident bytes; B reported 1,162,042,408. These are two sample runs, not a
benchmark or a claim that every image-processing stage executes on the GPU.

Both vision containers published private address `10.81.219.19:8090`, while the
resolved endpoints remained distinct. Eleven behavior tests pass, including
image validation, forwarding image bytes to the engine, and rejecting a response
without matching frame proof. The camera was attached to the browser's Mac,
not directly to an Orin. A physical Internet-uplink disconnection remains untested.

### Four-worker fleet verification

Later on 2026-11-10, the fleet expanded to Ci Mesh 1, Ci Mesh 2, Ci Orin Nano,
and Joannis Thor General. Each station discovered the other three through
reflected DNS-SD. Eight sequential vision requests reused a retained webcam
frame, with exactly two completed by each worker. All results reported GPU
residency and the expected model and frame hash. Processing times ranged from
2,340 to 8,003 ms. The new Orin and Thor received requests through A's discovered
`10.99.0.5:18090` and `10.99.0.9:18090` endpoints. B assigned the new Orin
`10.99.0.4:18090`, demonstrating that projected addresses are observer-local.

The local preview ports for that run were 19090 and 19091 for the original pair,
19092 for Ci Orin Nano, and 19093 for Joannis Thor General. A fifth device was
skipped at the operator's request. No existing apps or device identities were
changed, and no storage cleanup was performed.

### Continuous webcam verification

Version 0.3.0 was exercised in Safari with the physical MacBook Pro camera.
The camera stayed live while newly sampled frames were sent to all four workers.
A bounded observation finished with 27 completed frames: 2 on Ci Mesh 1, 8 on
Ci Mesh 2, 8 on Ci Orin Nano, and 9 on Joannis Thor General. Every sampled state
had at most four outstanding frames and no duplicate active worker endpoint.
Completed jobs carried real model results, GPU residency, and frame hashes.
There were no failed jobs in that observation. The measured completion rate
was about 0.63 frames per second over that observation window; this is not an
8-FPS inference claim. Thirteen behavior tests pass, including route changes
on failure and exclusion of workers already reserved by the coordinator.

If the station connection drops, analysis pauses while the browser camera stays live.
The dashboard keeps its last observed state, retries with a capped backoff, and
puts diagnostics under **Connection details**. Active analysis resumes after reconnection. **Cancel auto-resume** cancels that
intent, and **Pause analysis** or **Off** also prevents automatic resumption.
Device identity failures remain blocked. Interrupted submissions are never
replayed automatically. Changing coordinators clears pending frame bookkeeping
from the previous station.

The active dashboard at port 19090 now uses Thor as coordinator after Ci Mesh 1
and Ci Orin Nano stopped accepting agent connections. Ci Mesh 2 remains a peer.
The unreachable agents cannot be repaired through the app forward. Their
readiness must be checked separately before counting them as available workers.

The persistent-forward verification completed 90 successful state polls over
94 seconds with 66 real webcam results, 36 on Thor and 30 on Ci Mesh 2. All
results included an image hash and nonzero GPU residency; no failed jobs or
HTTP requests were observed in that window. Median state latency was 43.5 ms.
A subsequent intentional preview restart left Safari's camera live, displayed
**Cancel auto-resume**, then resumed sampling and returned new GPU results
without a reload or another button click. This verifies recovery from a local
proxy interruption, not repair of the two unavailable device agents.

### Memory containment in 0.4.1

The manifest sets a 4 GiB container memory ceiling and a 512-process/thread
ceiling. The agent must enforce these cgroup limits on deployment; the manifest
alone is not proof that a running device has them. A memory overrun must stop
the app rather than consume the host's remaining memory. This budget still
needs hardware validation alongside other applications running on the device.

The bundled engine allows one loaded model and one inference at a time. Text
and vision generation use the same 2048-token context during warmup and normal
requests. Switching models explicitly unloads the previous runner before loading
the next; a failed unload fails the request. Models expire after 30 idle seconds
instead of remaining resident indefinitely. During continuous camera analysis,
the vision runner is also unloaded before its fifth generation attempt to release
allocations retained across images. Startup warmup and failed generation attempts
count toward this limit; cached results do not. Model switching and runner
recycling can therefore add model-loading latency.

The memory changes have mocked-engine tests for startup, text/vision switching,
finite keepalive, unload failure, and recycling during successful and failed
continuous inference. They do not establish sustained on-device
memory use or verify the host agent's OOM priority; those require device checks.
