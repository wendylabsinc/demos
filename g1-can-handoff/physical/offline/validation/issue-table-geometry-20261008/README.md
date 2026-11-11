# Native table-height fault acceptance, 2026-10-08

Both original table-height faults are already repaired by the existing exposure-time pose correction. No production geometry, calibration, or acceptance limit was changed in this task. Added exact native fixtures and a regression that requires actual accepted observations on the failing inputs.

| Original attempt / fault frame | Frozen transform table error | Exposure-corrected table error | Cylinder RMSE | Target depth points | Corrected repeats |
|---|---:|---:|---:|---:|---:|
| 20261008T002131Z / 27901 | -5.022 mm, rejected | -1.970 mm, accepted | 2.854 mm | 2959 | 5/5 |
| 20261008T002809Z / 34006 | +5.160 mm, rejected | +0.392 mm, accepted | 3.257 mm | 2891 | 5/5 |

The frozen-transform replay repeats the exact native `Table plane differs by over 5 mm` fault on each original fault frame (3/3 each). Every corrected replay accepts all eight captured observations from semantic frame 27892 through 27901, or all three from 34004 through 34006. Values are deterministic across repeats.

Inputs were copied through native Wendy `container_exec` read-only from the original Nx2 controller DATA volume at `10.10.20.238:50052`, freshly rediscovered as Unitree G1 Nx 2. They include original JPEG/depth PNG ZIPs, original timing NPZs, and the original full-image calibrated ray map (the existing tracked shared ray map is byte-identical). Native NPZ color/depth/ray byte hashes are recorded. Capture timestamps, exposure timestamps, stream ID, camera serial, and frame IDs match the original events, establishing these are the old inputs despite later robot/app restarts. Recorded CUDA semantic source image SHA and exact frame identity are verified before geometry. The camera client decodes the original bytes with its ordinary identity, finite exposure, calibration, and 12 ms RGB/depth skew gates.

Nearest exposure poses were decoded from original CDR in sealed recorder session `20261007T235101Z-76f038c2`, segments 30 and 37. Segment SHA values were checked against the original manifest. Raw nearest CDR rows and decoded rows are retained in `native-poses.json`; all replay pose skews are <=1.063 ms, including 0.405 ms and 0.958 ms at the faults. Existing accepted historical calibrations and stationary pelvis pivot are retained. The original table-height gate stays 5 mm, target depth support >=40, cylinder fit <=6 mm, pose alignment <=20 ms, attitude <=0.04 rad, and unmodeled waist motion <=0.002 rad.

Independent disjoint tabletop fits on the two original source calibration images pass current broad-plane validation: held-out RMS 2.076 / 2.115 mm; training/held-out normal disagreement 0.00386 / 0.00647 degrees, with 16775 / 16702 held-out points. This validates the current broad-table fitter on those pixels. It does not turn the historical model-based accepted calibration into independently measured camera-to-robot metrology, and it does not substitute a new unaccepted calibration for the original replay.

This is full original RGB-D at the affected geometry seam, with every captured visual update after the last recorded semantic reference. Each replay initializes tracker state on that reference using its exact recorded CUDA response. Earlier tracker history before the semantic reference, the complete physical trial, live capture headroom, physical motion, and physical pickup are not replayed or verified. No native camera stream, service, deployment, DDS ownership, or motion state was changed. The prior 153 derived-table-point replay remains useful but is no longer the only evidence for these two faults.

Commands (from `physical`, using the existing Python venv):

```sh
PYTHON=/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 "$PYTHON" offline/table_geometry_replay.py --uncorrected --repeat 3 --report offline/validation/issue-table-geometry-20261008/red-uncorrected.json
# Expected exit 1: both original native faults reproduce.
HANDOFF_OFFLINE=1 PYTHONPATH=offline:. OPENBLAS_NUM_THREADS=1 "$PYTHON" offline/table_geometry_replay.py --repeat 5 --report offline/validation/issue-table-geometry-20261008/corrected-repeated.json
# Exit 0: both original native fault inputs accepted 5/5.
"$PYTHON" offline/run.py --test test_table_geometry_acceptance --report offline/validation/issue-table-geometry-20261008/tests.json
# 3 tests pass; runner blocks external network and real Unitree SDK imports.
```

`capture-manifest.json` protects the newly copied evidence without modifying the original offline manifest. Acquisition scripts are saved as provenance and are not invoked by offline tests. NumPy/Accelerate emitted matmul floating-point warnings during broad-plane fitting on this host; every asserted fit/quality metric was finite and passed its numerical bound. Raw stderr is retained rather than hiding those warnings.
