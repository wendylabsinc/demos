# Passive perception performance checks

## Exposure budget profile, 8 October 2026

Sixteen observations through the deployed spawned observation worker were
accepted in a separate passive process. The controller stayed FROZEN with
ownership retained; its existing SDK hold publications continued. This probe
created no actuator, acquired no camera lease and started no trial. It added
another perception workload, so these timings are not motion qualification.

The 400 ms gate measures **age since depth exposure**, including capture lag and
waiting between observations. It does not measure only inference or HTTP latency.
Warm observations (15 samples, including one periodic YOLO recheck) arrived at
221.1–273.8 ms exposure age, median 236.6 ms. The initial identity observation
arrived at 366.6 ms. Acquisition currently requires an exposure within 300 ms
(75% of the final 400 ms limit), then processing and final receipt must still
satisfy the final freshness gate.

An additive partition of the median-age warm observation, immutable frame 31095:

| Interval | ms |
| --- | ---: |
| Depth exposure to SDK receive | 66.5 |
| SDK receive to aligned frame timestamp | 9.3 |
| Aligned frame timestamp to fetch request | 35.1 |
| Fetch, metadata validation and concurrent RGB/depth decode | 48.7 |
| Completed prefetch waiting to be consumed | 23.9 |
| Exposure-pose lookup and geometry dispatch | 1.7 |
| Geometry, including visual/depth tracker and table validation | 49.3 |
| Remaining archive wait | 0.0 |
| Completion and parent receipt | 2.0 |
| **Total exposure age** | **236.6** |
| **Headroom before 400 ms** | **163.4** |

Inclusive component durations across the warm observations:

| Component | Median ms | Maximum ms | Samples |
| --- | ---: | ---: | ---: |
| Policy timing/depth HTTP bundle | 15.3 | 40.7 | 15 |
| Exact JPEG/depth recorder HTTP bundle | 36.8 | 53.5 | 15 |
| Depth PNG decode | 8.5 | 11.7 | 15 |
| Color JPEG decode | 6.8 | 13.6 | 15 |
| Mask / visual-depth tracker | 15.3 | 19.6 | 15 |
| Cylinder fit | 9.1 | 14.1 | 15 |
| Geometry including detection and table checks | 55.9 | 73.5 | 15 |
| Archive encoding and file write | 25.3 | 37.9 | 15 |
| Full worker request/response | 61.4 | 111.2 | 15 |

The HTTP requests run concurrently, as do RGB/depth decoding and geometry/archive
work. These rows must not be added. Archive work finished before geometry in all
16 samples, adding no remaining critical-path wait in this run. World transforms,
depth selection and table validation are measured together within geometry;
they are not separately instrumented.

YOLO service prediction took 24.486 ms initially and 24.416 ms at the periodic
recheck. Complete client round trips took 33.8 and 37.6 ms respectively. Service
prediction includes preprocessing, prediction and output processing; it is not
an isolated CUDA kernel measurement. Two YOLO calls do not establish its p95.

Cold worker startup took 1,917.7 ms. The first observation request took 744.2 ms,
including a 381.6 ms calibration-ray bundle fetch and a 140.7 ms completion-to-
receipt tail. A newer exposure was selected during that request, so request
latency and exposure age differ. Cold spans overlap the selected exposure;
the derived cold partition can contain a negative pre-fetch interval.

The preserved real trial explains why good individual results can still freeze:
frame 18361 was reported at 311.7 ms exposure age, leaving 88.3 ms headroom.
The next round trip took 86.9 ms and the tracking loop normally pauses 10 ms
between requests. The watchdog stopped on the old observation before replacement
frame 18363 arrived. That replacement was individually fresh at 270.7 ms, but
its parent worker-result event was logged 10.7 ms after the failure event.
Reported age and event timestamps are sampled at different points; this is a
headroom diagnosis, not an exact reconstruction of the watchdog's clock read.

Raw measurements, derived frame partitions, deployed file hashes, and the
SHA-256-verified original trial JSONL are preserved under
`offline/validation/perception-timing-20261008/`. Browser camera health age uses
a later frame timestamp and can understate true depth-exposure age: a separate
read showed 49 ms health age versus 147.7 ms exposure age after a 20.9 ms fetch.

For repeatable profiling, copy `profile_perception.py` into the existing physical
image and run with `PYTHONPATH=/app`, while no trial is running:

```sh
PYTHONPATH=/app OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /tmp/profile_perception.py --samples 16 > /tmp/profile.json
python3 analyze_perception_profile.py profile.json > derived-frames.json
```

The profiler retains all retry events and joins spans across requests by frame
and acquisition group. It writes exact-frame evidence into a separate
`/data/perception-profiles/<UTC timestamp>/` directory. The report includes
controller state before/after; a FROZEN controller may continue hold commands.

Run inside the physical container on the NX2, with the camera and inference
services already running. This probe uses public immutable frame bundles; it
never acquires a lease, opens USB/DDS, or issues motion.

```sh
python3 /app/benchmark_perception.py --samples 8 --max-age-ms 400 --max-p95-ms 180
python3 /app/benchmark_perception.py --samples 8 --config /data/calibration.json --max-age-ms 400 --max-p95-ms 180
```

The command prints per-frame acquisition/decode/total/exposure-age measurements,
and returns nonzero for a rejected observation or p95 latency above the supplied
budget. The second invocation also exercises measured target and table geometry.
The 180 ms default is a performance target, not a replacement for the physical
runtime's acquisition or final exposure-age checks. Neither invocation includes
the controller's archive write or hardware policy. Probe sample counts are
bounded to 1–32; warm-up counts to 0–3.

## Measured changes, 7 October 2026

On immutable NX2 payloads, concurrent Pillow PNG/JPEG decoding reduced median
decode time from 15.30 to 10.40 ms over eight paired measurements, with identical
depth and BGR arrays. Loading decoder plugins in the Camera constructor moves a
measured 58.8 ms first PNG initialization out of bounded observation acquisition.
Camera workers are reused across frames and shut down with their camera lease.

Semantic depth masking now operates on the exact existing YOLO inset before
expanding its result to the full image. Six same-frame comparisons matched every
mask pixel and reduced median mask time from 16.98 to 8.49 ms. False crop borders
preserve the previous binary erosion, including tracking ROIs. Thresholds,
ambiguity, clipping, lid exclusion and the 6 mm cylinder fit are unchanged.

The geometry inference worker is reused rather than created/joined per frame.
One profile observed a 36 ms shutdown/join, but the subsequent full-path samples
did not establish an aggregate p95 improvement from that change alone. The client
also shares the same BGR byte string between the outgoing checksum and request;
it still hashes the current image again when validating the response.

Six candidate camera-to-detection samples passed the 180 ms p95 target at 131.9 ms
and had exposure ages 149–279 ms. Full geometry remained above that target:
the final six samples had p95 181.9 ms and ages 260–303 ms, and independently
rejected the table plane by the existing 5 mm gate after the scene changed.
GPU inference itself remained approximately 22–25 ms. These passive observations
do not qualify a pickup or establish that every camera-age freeze is resolved.

Exact reports, including failed qualification attempts, are preserved in
`yolo/validation-pipeline-20261007.json`. Focused verification:

```sh
PYTHONPATH=.:yolo python3 -m unittest test_camera_pipeline test_camera_input test_camera_lease test_yolo_coke
```

All 35 focused tests passed. Camera serialization and HTTP scheduling still
dominate observed jitter; a metadata-only endpoint in the existing camera owner
could avoid downloading a 542 KB timing NPZ, but requires a separately verified
owner contract and was not changed by this work.
