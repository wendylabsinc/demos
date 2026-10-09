# Physical pickup development

## Evidence update, 2026-10-09

The recorded reach exposes sustained left shoulder-roll load: approximately
17 Nm reported torque with almost no motion, Kp 150 and approximately 0.12 rad
target error. Winding temperature reaches 85 C while casing stays 53 C. This is
the main demonstrated controller blocker; high stiffness and an obstructed
target are the leading explanation. The later saved retry stops on table-depth
perception, so overheating is not established as the sole remaining issue.
Full physical pickup/lift and repeatability remain to be verified.

See [THERMAL_FINDINGS.md](THERMAL_FINDINGS.md) for current recorded evidence,
the distinction between gain tuning and contact planning, and completion
criteria. This documentation update makes no controller or deployment change.

## Historical development notes

Work now lives in `g1-can-handoff/physical/` on the local `demos` repository's `main` branch. The imported controller files are tracked in local commit `8531d34`; these documentation updates are uncommitted. The temporary review checkout is no longer the working location. `MIGRATION.json` records hashes of the 251 imported source/evidence files.

The robot runs version 0.2.9-retry-from-hold. It automatically refreshes the approved model-based scene estimate per requested pickup, checks complete paths, holds on failure, and permits an explicit retry from a pregrasp hold without releasing control. A possible held can prevents a fresh pickup. Startup sends no commands.

Local changes beyond that deployed version include detailed collision diagnostics, continuous upward lift planning with 10 mm steps and locally bounded IK, full-path validation across the existing grasp orientation candidates, and the documented Dex3 timeout-bit stop command.

These latest path changes are NOT deployed or physically validated. The latest replay rejects every candidate without sending commands. The original orientation produces middle-finger/table interference or fails continuous IK; an alternative produces elbow/torso interference. No collision, tracking, temperature, or perception threshold was relaxed. Carried-can attachment collision geometry remains unfinished.

The requested left-hand resting curl was performed with low gains and then holding torque was removed. No arm commands were sent for that operation.

## Verification in this checkout

The existing local Docker image pins the MuJoCo/scientific dependencies. From the `g1-can-handoff` directory:

```sh
docker run --rm --network none -v "$PWD:/work:ro" -w /work/physical   --entrypoint python3 g1-handoff-shared-sim:test   -m unittest test_runtime_lifecycle test_pickup test_staging test_cartesian_lift test_pregrasp_collision test_scene_refresh test_dex3_packaging -v
```

Simulation source is `physical/sim_runtime.py`. The failure setup and successive replay results are `evidence/later-path-setup.json` and `evidence/later-path-*-simulation.json`. Evidence distinguishes simulations from physical trials. Older successful simulations do not validate the currently blocked setup.

Continue diagnosis and fixes here. Do not deploy over an active hold: restarting the old runtime ends its hold. Do not run physical pickup until the new candidate path passes the required checks.

## Continuous lift candidate, 0.2.10

The current setup replay passed with a grasp point 10 mm higher on the can, a 127 mm visual lift, a one-second verified hold, and zero pregrasp contacts. `evidence/higher-grasp-current-simulation.json` records matching source hashes. Physical pickup now ignores the unused offer endpoint, tests grasp heights of 0/10/20 mm above can center, and validates full paths. Continuous lift IK uses bounded adaptive subdivisions. Earlier blocked replay results above describe the preceding code/setup tests; physical verification of this candidate is pending.
