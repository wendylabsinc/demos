# Visual/depth-led Coke demo

The deployed default is `HANDOFF_TARGET_IDENTITY=color_depth` and
`HANDOFF_PERCEPTION_POLICY=phase_boundary`. Initial acquisition uses a single
red component with measured depth. The existing cylinder, table, workspace,
freshness and planning checks must accept it before movement. This identifies
a red can-shaped target; it does not assert Coca-Cola branding.

Accepted targets use the existing appearance/depth tracker. Uncertainty can
reacquire red/depth pixels only within the existing target neighborhood, then
must pass the same geometry and movement bounds. There is no YOLO request on
initial acquisition, phase checks, tracking, or reacquisition. The separate
YOLO service and unchanged checkpoint remain available for diagnostics.

Other explicit modes are `yolo_once` (initial semantic detection per fresh
track, including post-handover replanning, then stop on tracking uncertainty
without another classification) and `yolo_reacquire`
(previous behavior). `HANDOFF_PERCEPTION_POLICY=continuous` is the independent
rollback option for continuous observation. It does not introduce periodic
YOLO in `color_depth` or `yolo_once` modes.

Run `python offline/run.py --test test_color_depth_acquisition` to replay the two
initial recognition failures. Source ZIPs and calibration remain under
`offline/evidence/low-yolo-20261008`; the test clock is pinned to recorded time
without rewriting exposures. Run the complete regression and simulation suite
with `python offline/run.py --profile all`.

The checkpoint before these changes is retained by Git at `41dcc41`. No trained
candidate was deployed. New-view branding accuracy and a physical pickup are
not established by the offline tests.
