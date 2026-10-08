# Regular Start button and deployed compact tracking trial

Target: Unitree G1 Nx 2, `10.10.20.238:50052`, camera serial
`338622073335`. Controller `0.3.8-latest-frame`, camera
`0.1.1-compact-tracking`, application source commit `24e1249`.

The camera and controller deployed through native timed Wendy runs using the
existing Thor Gamma ARM64 builder. Read-only readiness verified deployed source
hashes, advancing original camera exposures, healthy recorder and YOLO, and a
PASSIVE controller with no actuator and zero commands. Deployment preserved the
accepted calibration and perception configuration. The demo later refreshed its
scene normally; `calibration-after-trial.json` is that post-trial configuration.

One click on the regular Start demo button was accepted and started one trial.
The page keeps its acknowledgement separate from subsequent controller errors.
Start is disabled until initial status is available and during maintenance;
the server also rejects starts during maintenance. Stop and Release stay usable.

The trial stopped before grasping on a predicted staging collision between the
table and left index finger: penetration 1.6223 mm at path fraction 0.304.
It did not complete a pickup or verify a lift. After the operator supported the
arms and used Release, two status observations verified PASSIVE, no actuator,
no running demo, and unchanged cumulative command count 6628.

All 93 accepted perception observations were within 400 ms exposure age:
minimum 165.9 ms, median 222.4 ms, maximum 272.3 ms. This is one staging trial,
not full pickup qualification or a matched comparison to the older passive
profile. The collision remains unresolved.

`trial.tar.gz` contains 651 native events, five exact RGB-D evidence frames,
source provenance and a verified manifest. `trial.json` contains controller and
recorder status, timestamps and artifact checksums. `operator-panel.jpg` shows
the persistent Start acknowledgement and final PASSIVE state.

The DDS exporter initially waited for sealed segments, then exposed a bug in
periodic recorder-status rows without `topic`. The exporter now preserves those
rows separately from DDS counts, while rejecting unknown row types. Regression
coverage includes mixed status/DDS rows, checksum failures and unsealed tails.
The successful sealed export contains 245,009 DDS samples and 90 recorder-status
rows, with no recorder-reported drops. Its 53.7 MB raw archive stays local as
`trial.dds.tar.gz`; its checksum and manifest are committed here. Upstream DDS
loss remains unknown. The smaller exact-frame/event fixture is committed under
`offline/evidence/20261008T173157Z/` for work without the robot.

Validation: 19 operator/transport/artifact tests passed. Earlier compact camera,
worker and motion tracking verification passed 73 focused tests. No collision,
freshness or motion interlock was disabled by this change.

```sh
python -m unittest test_operator_panel test_operator_transport test_recorder_export test_trial_evidence
```
