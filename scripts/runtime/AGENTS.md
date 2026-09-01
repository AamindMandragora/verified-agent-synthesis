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
  exclusions after validation, require exactly `--gpus 0,1,2,3`, and require
  exactly the four Spider cells to remain. GPU `1` may be selected only when
  the live memory gate can safely fit every planned worker.
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

## Paper baseline held-out queue

`run_paper_baseline_queue.py` is the bounded 2026-08-28 fixed-baseline queue.
Keep its 38-row scope exact: six GSM Qwen3.5-2B/4B rows for unconstrained,
GCD, and IterGen; two Spider Qwen3.5-2B/4B CARS rows; and the 30 requested
SMILES rows. Every row must retain its canonical split hash, exact model and
strategy, sample count, and isolated output path. Do not broaden the manifest
or silently accept an existing JSON without complete provenance validation.

The allocator must use live `nvidia-smi` memory, all worker reservations, and
the 2,000 MiB margin, and must filter to the manifest GPU scope. It must never
evict or stop another user's process. Dry-run mode prints both fixed-evaluator
and possible cold-rerun commands without claiming or calling anything.

After a valid baseline result, compare the frozen same-row metaDecode artifact:
GSM/Spider trigger on strict accuracy or syntax improvement; SMILES triggers on
strict unique-valid improvement. Ties do not trigger. Use the existing
post-14B atomic claim helper for at most one cold 40-iteration rerun per row,
never add warm-start inputs, and never delete an interrupted or failed claim.

- Manifest creation requires an external exact binding map for all 38 frozen
  metaDecode rows. Check each bound file's SHA-256 and same-row provenance
  before writing the manifest; do not auto-discover or leave blank bindings.
- Bind the manifest to the full current commit and hashes of the queue,
  evaluator, and benchmark scoring sources. Refuse startup when those hashes
  change or the fixed source files are dirty.
- A baseline win must schedule and run the claimed cold 40-attempt rerun
  through this same allocator, with the stored strict thresholds and a final
  validated held-out artifact. A surviving child is waited for and reattached
  by its PID start identity; it is never duplicated.
- Rerun state must record the active `synthesis` or `heldout` phase. A
  controller restart after synthesis must recover the success or exhausted
  best attempt through the cold queue before held-out evaluation; a restart
  during held-out must wait for or re-run only the hash-pinned held-out step.
- Bind every direct runtime dependency, including `crane_repo_runner.py`, GSM
  and SMILES dataset loaders, and `.context/run_post14b_rebar_queue.py`.
  Startup rejects any staged or unstaged tracked worktree change.

## Paper Tables 5--8 campaign queue

`run_table5_8_queue.py` is the separate 8-run GSM-Symbolic campaign for the
Table 5 backend cells and Tables 6--8 ablations. It uses exact profiles
`gpt5.6-sol`/Pi provider-only ChatGPT OAuth, `gemini3.7-flash`/direct Gemini API, and
`opus5`/Claude Code, and always
evaluates GSM-Symbolic with `Qwen/Qwen3.5-2B`. Table 5 has 3 author-model runs;
Tables 6--8 add 2 token-budget runs, 2 beam-size runs, and 1 helper-mask run.
The default Opus run supplies the shared budget-1, beam-2, and mask-on controls,
so the 8 physical runs populate 11 paper cells. Every row uses exactly one GPU
and records accuracy, syntax rate, synthesis attempts used, accepted/exhausted
status, constrained work, phase and total wall times, phase timestamps, and
available per-attempt evaluation times. Runtime evidence must set
`phase_timing_coverage` to `all_phases` or disclose a pre-timing restart with
`recovery_anchor`. The synthesis command uses the
canonical train split selected inside `run_synthesis`; held-out commands use the
matching canonical test split and an isolated output file.

The Gemini campaign route loads only `GEMINI_API_KEY` from the canonical
private `synthesis/.env`, passes no Vertex or backup credential, and binds only
the successful key's SHA-256 fingerprint in reports, pilots, and manifests.
`campaign_environment()` must also install the exact canonical non-secret Pi
Node/bridge/auth-file paths and Claude config/account settings. Do not require a
normal focal login shell to export those settings separately, and do not place
credential contents in the repository.

The manifest records the full commit, CRANE commit, and SHA-256 for every
direct runtime dependency. A dirty dependency is a launch error. State is
written by replacement under a file lock and records `phase`, PID, and process
start identity. A restart waits for a surviving child and cannot treat an
unchanged pre-existing held-out file as this run's result.

GPU admission intersects the command-line GPU list with each row's scope,
assigns one row to one GPU, and requires the selected GPU to fit
`max(memory_reservation_mib,
ceil(gpu_mem_util * total_memory)) + 2,000 MiB`, including earlier worker
reservations. When no row fits, the controller polls rather than exiting.
Provider preflight verifies the Pi OAuth route without sending a model prompt,
while Gemini and Opus startup checks use their exact approved routes. Use
`--dry-run` to print exact commands; it does not write claims or start a
provider/GPU job.
