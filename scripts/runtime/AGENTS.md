# AGENTS.md — `scripts/runtime/`

## Scope

Cold-queue and babysitter runtime scripts. Repo-wide rules live in `../../AGENTS.md`.

## Cold synthesis queue (`run_cold_synthesis_queue.py`)

- SMILES cells must export `CSD_CONSTRAINED_TEMPERATURE=0.7` in synthesis and
  held-out envs. Default argmax (`0.0`) collapses every example to the same
  tiny SMILES and drives unique-valid accuracy to zero.
- Always forward the job's `gpu_mem_util` via
  `CSD_VLLM_GPU_MEMORY_UTILIZATION` (consumed by `synthesis.run_synthesis`).
  Do not hard-code the global `VLLM_GPU_MEMORY_UTILIZATION` for cold jobs.
  Also set `CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX` to the same admitted value;
  a pooled worker must not retry above the memory budget the queue reserved
  while sharing focal with other users.
- Do not put strategy coaching into `--task` strings (Critical Prompting Rule).
- Claude Code synthesis queue commands use the fixed `claude-opus-5` model ID;
  direct Anthropic and Bedrock model IDs are separate routes and should not be
  changed by this contract.
- The `full-baseline-20260803` profile must contain exactly 20 cold cells and
  800 author attempts. Its manifest must bind the `aadivya@fermi.ai` Max
  profile and all five raw baseline hashes before dispatch.
- For that profile, use one exact example above the maximum baseline accuracy,
  cap the maximum syntax rate at 90%, and use the explicitly labeled 95%
  exception only when the maximum baseline accuracy is 100%.
- Before accepting a baseline into campaign evidence, record its nonblank and
  distinct-output counts. Reject exact 0/0 batches that are entirely blank or
  contain only one repeated malformed output.
- Repair reruns must use a new `--campaign-output-name` and exact repeated
  `--include-label` values. Never overwrite the original 100 baseline files.
- Recompute SMILES baseline accuracy from distinct RDKit-valid, in-class,
  non-exemplar molecules across the full trial; do not trust per-row averages.

## Corrected exact-zero campaign

- The `full-baseline-corrected-20260805` profile must contain the exact approved
  20 cells and exactly 675 remaining author calls. Validate that full manifest
  and its independent approval before applying exclusions. This profile is now
  bound to the approved Spider-only resume: require both `gsm-` and `smiles-`
  exclusions after validation, require exactly `--gpus 0,2,3`, and require
  exactly the four Spider cells to remain. GPU `1` must stay free.
- Dispatch phases are strict: ten fresh changed-target cells, two exclusive-GPU
  memory retries, two remaining-call recoveries, three unchanged cells that
  never started, then three held-out-only jobs.
- Recovery reports count restored completed evaluations plus remaining calls;
  the two jobs must never exceed their original 40-call budgets.
- Held-out-only jobs must pin the exact compiled CSD path and SHA-256. Validate
  the pin again before dispatch.
- Startup and every dispatch must revalidate an independent `gpt-5.6-sol`
  approval bound to the corrected evidence SHA-256 and queue-manifest SHA-256.
  The exact-zero synthesis block remains a hard stop until that approval exists.

## Exact-zero baseline repair monitor

- Run `scripts.runtime.incident_repair.exact_zero_baseline_monitor` every 300
  seconds while a selective exact-zero repair pool is active.
- Keep `.context/exact-zero-repair-synthesis.blocked` in place until every
  manifest row passes review and corrected evidence plus the later queue inputs
  pass independent validation. Queue launchers must treat it as a hard stop.
- Enforce that stop in the campaign builder immediately before launch, in the
  queue at startup, before each GPU reservation, and again in each worker before
  it runs a synthesis job.
- Require the exact expected number of unique manifest labels, the matching
  `source_exact_zero_count`, every frozen source SHA-256 and literal 0/0 score,
  and the configured repair root
  `outputs/baselines/exact-zero-repair-20260804`. Source and replacement paths
  must be unique, must not overlap, and replacement paths must stay under that
  root. The frozen source may itself contain blank or repeated malformed output
  because that is the preserved failure being repaired.
- Bind every report and block marker to the repair-manifest SHA-256. Malformed
  manifest or acceptance input must record `monitor_error`, keep synthesis
  blocked, and allow the next scheduled poll to retry.
- Read each manifest, frozen source, and replacement artifact once per poll;
  derive its hash, diagnostics, and scores from those exact bytes.
- Preserve quarantined artifacts. Blank exact 0/0 output and one repeated
  malformed answer are system failures, not baseline scores.
- Diverse, nonblank exact 0/0 output remains blocked until skeptical review is
  recorded against the exact artifact SHA-256 in the acceptance file. Never
  accept a label-only approval after its artifact changes.

## Zero-acc babysitter

Repair agents run in the sibling worktree
(`~/csd-generation-babysitter-repair`), never by checking out branches on the
live cold-queue tree.
