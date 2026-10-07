# Passive perception performance checks

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
