# First operator-confirmed pickup success

GitHub checkpoint: `g1-first-working-recorded-v0.3.29`.
The full replay archive is attached to the [GitHub release](https://github.com/wendylabsinc/demos/releases/tag/g1-first-working-recorded-v0.3.29).
Download `trial.tar.gz`, `source-sha256.json` and `SHA256SUMS`; run
`shasum -a 256 -c SHA256SUMS` in their download directory to verify them.
The Git commit's copied controller-source digest matches the deployed successful
version; the historical build stamp retains its original base commit and dirty flag.

Oliver confirmed this latest completed attempt as success #1. Completion: 2026-10-09 23:32:29 UTC (16:32:29 PDT), controller 0.3.29-post-grasp-feedback, source digest69e62188a2f4684701d2c38076f66fb13f7721496d3117f6c0b2b373de7988a3.

Measured finger evidence established thumb, middle and index contact; gentle holding effort was0.5 Nm after0.2-second stop detection. The commanded lift completed129.94seconds after arm ownership acquisition. Physical success is operator-confirmed; automatic visual lift verification was intentionally disabled and remains false in the native result.

The later holding watchdog stopped on waist alignment17.23seconds after completion; supported release followed. Both events remain in the original evidence and trial.json, separately from the confirmed completed pickup. Earlier pre-close perception failure is not relabeled as success.

trial.json is the success annotation; offline/attempts.jsonl includes its outcome. trial.tar.gz contains3523 chronological events, planning snapshots and models,5 selected exact RGB-D frames and controller source/build stamp. The63,192,020-byte archive and every manifest member were SHA256-verified on transfer. Archive hash:87ef2066836d402e658af7af55aebe1354d44e720ac2479ad23bb47906432fe1. Original events are also retained as events.jsonl.gz. Raw high-rate DDS recording remains in recorder session20261009T231424Z-43ce47f5 on NX2; it is not included in this archive. No new demo, deployment or motion was issued while annotating this result.
