# Pi OAuth provider for the Tables 5--8 campaign

Date: 2026-08-30

## Purpose

Run the GPT-5.6 Sol synthesis-author arm through the existing ChatGPT/Codex
OAuth subscription without using `OPENAI_API_KEY`, `codex exec`, or a coding
agent's built-in system instructions.

## Result

- Backend name: `codex` (kept for the existing campaign schema).
- Actual provider route: Pi's `openai-codex` provider with model
  `gpt-5.6-sol`.
- Request shape: the campaign system instructions plus one user message,
  reasoning `high`, `tools: []`, `tool_choice: "none"`, no prior response, and
  no conversation state.
- Pi package: `@earendil-works/pi-coding-agent@0.84.4`, pinned by the package
  lock, package URL, integrity value, source commit, and a hash of every file in
  the installed dependency tree.
- Node runtime used for the focal verification:
  `/home/aadivyar/.local/share/cursor-agent/versions/2026.07.23-e383d2b/node`
  (`v24.5.0`). Production requires this path through
  `CSD_PI_NODE_EXECUTABLE`; there is no fallback to an arbitrary `node` on
  `PATH`.
- Private OAuth file:
  `/home/aadivyar/.pi/csd-table5-8/auth.json`. This file is outside the
  repository and is never copied into results or logs.
- Verified OAuth account fingerprint:
  `ce215c6030a9cdd9c3509c7e87a711fe18014b8604bf0960ed9b53f42f125f6f`.
- Installed dependency tree: 12,516 files, SHA-256
  `90223f63fdbab516024cb558a78ebf1e9ab9096ff199916b87f6d72d3f8604b1`.

## Verification

The campaign/runtime/provider suite passed with `433 passed, 2 warnings`.
The two real bridge tests used the exact Node 24 executable and verified both
local OAuth loading and secret-free authentication failure output. A separate
no-prompt live OAuth check returned provider `openai-codex`, model
`gpt-5.6-sol`, and the account fingerprint above.

The first real pilot exposed that the production success writer stores its
attempt evidence in top-level `strategy_code`, `compiled_dir`, and
`evaluation_result` fields. The pilot parser now accepts that exact shape while
requiring the real `success_report.json`, both Dafny files, the compiled Python
artifact, and successful one-example evaluation evidence inside the same run.
The production-shaped regression test failed before this repair and passed
after it.

The first manifest build then exposed a tracked CRANE symlink whose target is a
directory. The source binder now hashes the symlink marker and exact link target
instead of following it and trying to open the target as a file. A focused
tracked-symlink regression test failed before the repair and passed after it.

A broader `pytest tests` collection attempt stopped on four unchanged legacy
test/import mismatches outside this change. The affected source and test files
had no diff in this worktree. They are not part of the campaign runtime gate.

## Reproduce

From the repository root on focal:

```bash
export CSD_PI_NODE_EXECUTABLE=/home/aadivyar/.local/share/cursor-agent/versions/2026.07.23-e383d2b/node
export CSD_PI_BRIDGE_PATH="$PWD/synthesis/generate/pi_oauth/provider/bridge.mjs"
export CSD_PI_AUTH_PATH=/home/aadivyar/.pi/csd-table5-8/auth.json

cd synthesis/generate/pi_oauth/provider
npm ci --ignore-scripts --no-audit --no-fund
cd ../../../..

/apps/conda/aadivyar/envs/csd/bin/python -m pytest -q \
  tests/runtime tests/providers/test_codex_provider.py
```

Do not launch a paper campaign from an uncommitted tree. Build fresh provider
pilots and a new sealed manifest after the final commit, then let the existing
GPU admission checks decide when rows may start.
