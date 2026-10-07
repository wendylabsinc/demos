# NX2 YOLO validation — 2026-10-07

The old Coke-trained checkpoint is wired into the physical handoff on
`unitree-g1-nx-2` (`10.10.20.238:50052`, camera `338622073335`). This snapshot
verifies deployment and inference; it does not establish a physical pickup.

## Source and threshold

- Demo base: `7600ecf57554bd4b2165fbc740cc27a8cdded609`.
- Model source: `wendylabsinc/g1-bartending-demo` at
  `2dcc7f82ad67388927a9b843a7412b6b9ee65d26`.
- Model SHA-256: `87fe92dc461f56077e007974045d67acd7a0acdaab00530b9e2fe0c5ec35bfc7`.
- Confidence must be **strictly greater than 0.30**. Configuration cannot lower
  the threshold below 0.30. No physical color or CPU fallback exists.

## Software checks

- All 20 new semantic/depth/HTTP/model-contract tests pass.
- All 90 tests outside the rendered simulation module pass.
- The full pinned Linux EGL suite ran 103 tests: 98 passed and 5 failed in the
  existing rendered policy scenarios. The same five scenario failures were
  reproduced on the unchanged base commit on Linux and macOS, with identical
  Linux failure names and errors; no policy/control code
  was changed to bypass them.
- The archived real-camera fixture is recognized at confidence 0.658983,
  annotation IoU 0.751345. The optimized decoder agrees with Ultralytics within
  1 pixel and 0.01 confidence; this replay uses CPU exclusively for offline
  testing.
- Both Wendy manifests validate, and the deployment-wrapper audit passes.
- The Linux test image includes OpenCV for the raw-image HTTP contract test.

Run from `g1-can-handoff`:

```sh
docker build -f physical/Dockerfile.sim -t g1-handoff-yolo-sim:test .
docker run --rm g1-handoff-yolo-sim:test
```

## Device evidence

The detector (`0.1.2`) and passive handoff (`0.2.13-coke-yolo`) were deployed
using native Wendy through the timed deployment wrapper. The camera and
independent recorder were retained. Device-side source hashes match the
checkout; hashes and compact measurements are preserved in
[validation-nx2-20261007.json](validation-nx2-20261007.json).

The detector health reports `pytorch_cuda`, GPU `Orin`, the pinned checkpoint,
and class `Coca-Cola`. Model recognition on the saved fixture must pass before
the readiness port binds. Live requests measured:

| Frame | Inference | Exposure age after inference | Detections above 30% |
| --- | --- | --- | --- |
| 23980 | 28.074 ms | 275.340 ms | 0 |
| 23983 | 39.122 ms | 207.650 ms | 0 |
| 23985 | 39.477 ms | 231.701 ms | 0 |

Keeping requests on the warmed CUDA thread resolved earlier stale-exposure
rejections without relaxing the 300 ms limit.

## Physical policy trial remains blocked

The operator authorized one pickup/lift trial. The read-only `POST /plan`
returned HTTP 409: `Coke YOLO target absent/ambiguous`. A can is visible, but
there is no qualifying semantic detection. An offline diagnostic on a saved
JPEG of the live view scored the can at approximately 0.0584, below 0.30.

No `POST /trial` was issued, no motor adapter was initialized, and the motion
command count remained zero. After the final source deployment, the runtime
was verified in `PASSIVE` with healthy telemetry and zero commands. The full
physical pickup/lift test must wait for a qualifying current Coke observation
and the existing calibrated plan/recorder/ownership gates. A saved fixture
pass, deployment success, or healthy CUDA endpoint is not physical success.
