# Tracking-position failure: offline diagnosis and acceptance

Native attempt `20261008T004012Z` failed with `Tracked can moved outside the checked pickup path`. Captured target XY drift is **10.100853 mm**, crossing the unchanged 10 mm pregrasp guard during cold control ownership. Tracking began against the preownership plan and continued through the handover.

Ranked hypotheses were ownership changing the camera/base reference, missing exposure-pose alignment, and real can displacement. The current runtime already has the causal ownership repair: stop old tracking, settle control, refresh scene, rebuild the plant, prepare/check a complete owned path, then restart tracking. This change predates this investigation; it is not newly implemented here. No displacement tolerance was increased. Current exact-frame exposure correction reduces the diagnostic drift to **9.721255 mm**, showing attitude/projection contributes to the borderline failure, without proving the can stationary.

## Same native observations through actual runtime

`offline/tracking_position_replay.py` invokes actual historical/current `Runtime.run(True)`, real `pickup.prepare`, IK, continuous collision preflight, publication guards and the beginning of real `pickup.execute`. Camera/perception and controller constructors are replaced with recorded inputs and inert sinks. The first pickup hand-command boundary checks real `PhysicalPlant.validate_start` and `check`, then stops with a dedicated exception. It never emits or claims a physical pickup.

```sh
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m offline.tracking_position_replay --repeat 3
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m unittest test_tracking_position_replay -v
```

All three repeats in `repeated-red-green.json` gave the same verdict:

| Sequence | Result | Pickup command sink |
| --- | --- | --- |
| Captured old runtime source | Exact native tracking error | Blocked |
| Current owned refresh/replan | Reaches checked first pickup sink | Accepted offline |
| Current, an additional 25 mm after owned preflight | Exact native tracking error | Blocked |

The real checked plans independently audit 4,342 preownership and 5,953 owned arm samples. Full-path preflight also checks the finger trajectory. The saved October 8 01:03 runtime source retains the original handover sequence; its SHA256 and immutable observation/feedback capture hashes are in each result.

## Evidence limits

The native recording ends during handover. It has **no stable post-settle feedback, exact feedback at new planning time, or post-settle RGB-D exposure**. Consequently the current branch uses a **counterfactual settle boundary and holds the terminal native scene constant afterward**. Recorded before/terminal snapshots supply measured state, recorded staging start supplies the left arm, and ownership event supplies the held right arm. Receipt times are rebased offline. Saved calibration is reanchored to terminal IMU/base values; no independently measured new camera/table calibration is claimed. This is actual runtime/planning acceptance of recorded geometry under explicit missing-input substitutions, **not an exact native settled-tracking replay**.

`exact-frame-geometry-diagnostic.json` separately fits the original decoded JPEG/depth/rays for frames 44504 and 44725 with their native exposure poses. Saved YOLO/visual-tracker boxes drive the fit. Frame 44723's semantic recheck RGB-D and the complete tracker frame sequence were not preserved here, so this diagnostic cannot count as fresh semantic inference replay or prove target identity after handover.

No device commands, DDS ownership, services, demos, deployments, network I/O, substantial motion or physical verification were performed.

## Added guard hardening

`runtime.py` now copies/freezes each checked tracking reference and rejects malformed/nonfinite measured centers before publishing. New samples still compare with one immutable path origin; 4/8/12 mm cumulative drift rejects 12 mm rather than silently rebasing on each sample. Timestamp advancement, exposure age, latched uncertainty, semantic rechecks and the existing 10/40 mm limits remain enforced.

`reference-tests-old-source-red.txt` shows both newly added reference/invalid-position regressions fail against HEAD's prehardening methods. `reference-tests.txt` shows both pass with current methods. The dedicated module also covers historical/current Runtime.run and the genuine displacement block. No broad regression rerun was performed by this agent.
