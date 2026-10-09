# Saved versions and rollback

## Before the readability cleanup — 2026-10-09

The complete local source checkpoint is commit
`f9464d3a54de0bebee8009a48d1257c8606dc364`, tagged
`codex/g1-handoff-before-readability-20261009T181731Z`.

The verified source/model/assets archive and per-file checksum manifest are at:

`/Users/olivertaylor/Documents/Wendy/backups/g1-can-handoff-before-readability-20261009T181731Z/`

Inspect the original in a separate checkout without replacing your edits:

```sh
git worktree add /tmp/g1-handoff-before-readability codex/g1-handoff-before-readability-20261009T181731Z
```

The archive's `RESTORE.md` describes its contents. Existing recordings and
validation evidence stay in `physical/offline/`; they were not reformatted or
removed. A source checkout is separate from a robot deployment. The normal
supported-release preflight and Wendy deployment wrapper still apply.

## Before visual/depth tracking

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
