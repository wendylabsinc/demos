# Existing NX2 camera owner: compact tracking update

This is the exact four-file source package read from the deployed
`g1-can-camera` owner, with a narrow additive tracking contract. The original
camera service SHA-256 was
`ddb88b790755e62160075e8529509989ce28ee2b7b73dc5c6ec8dff8a2d833e6`.
The immutable original remains in `../offline/reference/camera/`.

The app ID, camera serial 338622073335, 640x480 at 15 FPS, JPEG quality,
lease behavior, other HTTP endpoints and hardware reset settings are retained.
No second USB camera owner is created. `source_metadata` adds original depth and
color exposures and calibration to the existing exact-frame recorder bundle.
`/frame.tracking-rgbd` atomically selects the latest available frame; optional
`after_frame_id` and `stream_id` demand progress in the same stream. A condition
notification wakes the reader on capture, with a bounded 200 ms wait. Missing
progress or a stream restart is an error, never a repeated or re-timestamped frame.

Native deployment replaces this camera app first, then the physical controller.
The old controller still understands all its original endpoints. The new
controller uses the advertised compact contract and preserves old recorded
fixtures through its legacy decoder. The physical worker's continuous buffer
requires advancing source frames; readiness must prove the new camera contract.
`verify.py` checks exact deployed hashes and two fresh advancing bundles without
acquiring a lease or starting motion.

Use the workspace `wendy-run-timed` wrapper for deployment; never restart an
owned physical controller before its supported-release confirmation. SDK
capture settings were not changed by this experiment: the measured ~76 ms
capture/alignment lag remains a separate follow-up.
