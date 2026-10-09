# Gentle grasp during close

The physical pickup enters `close` after `reach`. Each closing finger that has
moved at least 0.02 rad, remains at least 0.12 rad short of its commanded target,
and spans no more than 0.005 rad over 0.2 seconds of advancing measured feedback
switches to a gentle holding target. Repeated readings do not advance this timer.
Two blocked finger groups and all closing fingers stopped establish `grasped`;
the closure ramp then ends instead
of continuing toward the fixed CLOSED pose. An empty endpoint is not a grasp.

`HANDOFF_GRIP_STOP_S` is seconds, default 0.2, valid 0.05..1.
`HANDOFF_GRIP_STOP_SPAN_RAD` is stopped encoder span in radians, default 0.005,
valid 0.001..0.02. This prevents a slow, still-moving finger counting as stopped.
`HANDOFF_GRIP_HOLD_EFFORT_NM` is nominal proportional holding effort per joint,
default 0.5, valid 0.05..1. With physical finger P=2, the hold target is at most
0.25 rad beyond the measured contact position. It is anchored at contact, rather
than repeatedly advanced to squeeze harder. Its output is clamped against fresh
position feedback to bound the nominal proportional term; damping, thermal
scaling, hardware saturation and uncalibrated motor telemetry still affect the
actual torque. These values do not bypass feedback, temperature or fault checks.

Finger target offsets remain logged as `finger_position_error_rad`. They do not
invalidate an established grasp. Arm tracking remains separate. Once `grasped`
is admitted, production runs finish the planned lift and hold on robot feedback
alone. No further camera/depth, can-position, identity or lift-validation query
is required. `grasped` denotes encoder contact evidence, not independently
verified Coke identity or lift. Completion reports `lift_verified=false`, with
visual rise and visual hold evidence null, rather than inventing visual success.

`HANDOFF_POST_GRASP_CAMERA_CHECKS` accepts only `0` or `1`, default `0`.
The deployed descriptor sets `0`; `1` restores the diagnostic camera lift checks.
The shared sequence and simulator retain an explicit
`post_grasp_camera_checks=True` diagnostic option for the saved regression cases;
set it to `False` to exercise the deployed feedback-only completion path.
All pre-grasp perception and robot feedback, cancellation, fault, collision,
thermal and arm-motion checks remain unchanged. An in-flight continuous camera
failure after grasp cannot reinstate a visual stop. Startup still needs a camera
for planning and pre-grasp checks; this does not make a missing camera ready.

The runtime watchdog refreshes the gentle hand command and its thermal scaling
while active arm planning is busy. The normal failure/cancellation and supported
release path remains in place; no new commands are sent after cancellation.
This change does not add a powered hand reset or a frozen-state recovery policy.

Simulation hand P/D now match the physical sender (2/0.2); the generic scene's
P=12 was unsuitable for evaluating a 0.5 Nm nominal holding target.

Recorded-failure regression uses sealed, hash-verified CDR from the 21:14 and
22:45 failures in `offline/validation/grip-transition-signals-20261009/trace.json.gz`.
Both begin reducing thumb effort before 50 C winding and establish the completed
grasp more than 2 seconds before their real stops.
That is trigger/capped-command evidence, not a prediction of physical cooling or
proof that the lower effort retains a can. No real motion was run by these tests.

Run the focused tests with the project venv Python:

```sh
python -m unittest test_gentle_grasp test_pickup test_arm_gravity test_runtime_packaging test_dex3_packaging test_sim_runtime
```
