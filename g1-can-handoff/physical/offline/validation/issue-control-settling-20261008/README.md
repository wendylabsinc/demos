# Native control-settling replay

The actual `control_settle.wait_for_stable_control` rejects the two original
quiet-body traces with the previous 0.03 rad/s limit and accepts both with the
current strict `< 1.0 rad/s` limit. Three deterministic repetitions produce six
red-to-green results. All 678 negative-control runs reject. See `acceptance.json`.

This changes the allowable measured body-joint speed only. The replay verifies
that exactly 1 rad/s and speeds beyond either sign of the limit still reject at
every one of the 29 body joints. Every base joint and IMU axis is independently
tested with an injected continuous drift. Stale body/left/right feedback,
body/left/right motor faults and both hands' device-error fields still reject.
Mutations are explicitly negative controls, applied to copies of native samples.

`native-feedback.json.gz` SHA-256:
`45b018d0d16590dc011412aa2d77f4108bbbfc01a06178c75a5db82e36314093`.
It contains 741 rows for original trial 2 and 752 for trial 3, including original
raw CDR and decoded measured body and both hand feedback. No hand feedback is
synthesized. The export retained at most one row per topic per 20ms and the
200ms prelude. Source segments and verified SHA-256 values are recorded in the
artifact and report. `export_native_feedback.py` and `export_cdr.py` preserve
the export procedure used by the prior capture task.

The original exporter included per-motor `motorstate` but omitted the two
additional hand `error` words checked by `Runtime.sample`. The replay reads
these directly from the preserved little-endian final HandState CDR fields;
all 994 native hand rows contain zero in both error words. The exact generated
schema used to verify their placement is copied to `HandState_schema.py`.
Unexpected CDR header/length fails closed. This avoids replacing missing fault
data with assumed zeroes.

The replay maps recorder monotonic receipt timestamps into a virtual clock,
retains per-topic sample ages, and invokes the production loop with 20ms reads.
It does not refresh old samples to the current time. The native controller's
callback timestamps and exact loop sample selection were not recorded, and its
hand readers used a different polling cadence. Therefore this is a deterministic
replay of original recorder-selected feedback, not an exact reconstruction of
the original controller schedule. Original physical settling, a new ownership
transition and motion execution remain unverified. No robot was accessed or
commanded for this replay.

Run from the physical directory:

```sh
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python -m unittest test_control_settling_acceptance -v
PYTHONPATH=. /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python offline/control_settling_replay.py --repeats 3
```

Dedicated unittest result: 3 tests passed in 6.255s. The unittest suite covers
the native red/green result three times and each negative case on both traces;
the acceptance report additionally repeats every negative case three times.
Existing production code was unchanged by this replay work.
