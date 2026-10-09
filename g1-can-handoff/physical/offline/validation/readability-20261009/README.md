# Readability cleanup validation — 2026-10-09

Saved baseline: `f9464d3a54de0bebee8009a48d1257c8606dc364`.
Rollback instructions are in `../../../ROLLBACK.md`; the code tour is in
`../../../CODE_GUIDE.md`.

- Formatted 223 active Python files and documented the main control,
  perception, planning and deployment flows.
- Verified executable AST parity for 221 Python files, excluding docstrings.
  The controls server separately matches its original AST except loading the
  extracted HTML template. The browser JavaScript matches its original AST.
- Controls tests now extract JavaScript functions using JavaScript itself,
  rather than assuming each function occupies one line.
- Parsed Stagefile configuration is unchanged.
- 1194 saved models, assets and archived evidence files are byte-for-byte unchanged.
- 100 focused offline tests passed. Their mocked redeploy success messages are
  test output, not actual device deployments.
- Full baseline: 452 tests, 14 failures, 3 errors.
- Full after cleanup: 455 tests, the same 14 failures and 3 errors.
  The failure identities match exactly; the additional state tests pass.
- Black formatting check and the workspace deployment-wrapper audit passed.
- No robot deployment or hardware motion was issued.

The preexisting failures include disabled local settling-timeout raises,
updated table/temperature/planning assumptions, source-fingerprint replay
checks, an incomplete replay test double, the Mac `/data` default, and existing
simulation pickup/perception failures. They were not repaired or weakened as
part of this readability change. Full tracebacks and test identities are in
`baseline-tests.log`, `after-tests.log`, and `report.json`.
