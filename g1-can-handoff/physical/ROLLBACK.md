# Restoring the version before visual/depth tracking

Both rollback references are saved locally and on `origin`:

- Last deployed controller/model: tag
  `g1-nx2-yolo-deployed-pre-tracker-20261007`, commit
  `63d955391aab911d51c392ab125b22ddfab66e55`.
- Complete source work in progress immediately before the tracker change:
  branch `codex/g1-nx2-yolo-before-tracker`, commit
  `6e7472a93e30fd21e81bdbf71fcf2ba4cf64ebc4`. This includes the new 10% setting
  and attempt diagnostics but was not deployed; its empty-crop qualification
  still failed with the old model.

Use the deployed tag to restore the previous running system. First verify the
controller is idle and has released its arm/finger adapters; a frozen hold is
not permission to restart. Use the operator's supported Release action when
necessary. Create a separate checkout so the tracker work remains available:

```sh
git worktree add --detach ../demos-g1-nx2-yolo-rollback g1-nx2-yolo-deployed-pre-tracker-20261007
python3 /Users/olivertaylor/Documents/Wendy/scripts/check-wendy-deployment-wrappers --root /Users/olivertaylor/Documents/Wendy
```

Deploy the saved inference and controller directories, in that order, using
`/Users/olivertaylor/Documents/Wendy/scripts/wendy-run-timed` with each directory
as `--prefix`, device `10.10.20.238:50052`, and
`--builder docker --dockerfile build.stagefile.yaml --detach --yes --chunking off`.
The directories are `g1-can-handoff/physical/yolo` and
`g1-can-handoff/physical` within the rollback checkout.

Verify both app versions, model checksum, readiness, camera freshness and
zero initialized motion adapters afterward. Deployment never starts a demo;
an explicit Start remains necessary.
