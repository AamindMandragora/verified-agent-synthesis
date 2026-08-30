# Runtime queues

`run_paper_baseline_queue.py` runs the held-out fixed-strategy cells for the
2026-08-28 paper update. It is deliberately separate from the synthesis
provider launchers.

The queue builds or validates one manifest containing exactly 6 GSM cells, 2
Spider CARS cells, and 30 SMILES cells. Each row binds its model, strategy,
canonical split and SHA-256, sample count, and output path. A result is skipped
only when its JSON, answer count, metrics, and provenance all validate.

Example manifest construction (no evaluation is run):

```bash
python scripts/runtime/run_paper_baseline_queue.py \
  --repo /home/aadivyar/csd-generation-worktrees/paper-missing-results-20260828 \
  --manifest .context/paper_baseline_manifest.json \
  --python /apps/conda/aadivyar/envs/csd/bin/python \
  --state-dir .context/paper_baseline_state \
  --claims-dir .context/paper_baseline_claims \
  --metadecode-bindings .context/paper_baseline_metadecode_bindings.json \
  --write-manifest
```

The binding file is required and must map each of the 38 cell IDs to the
exact frozen metaDecode JSON and its SHA-256. Manifest creation checks the
artifact's same-row model, dataset, class, split, and sample provenance;
startup checks the commit and fixed-source hashes again. The rerun environment
also selects `/home/aadivyar/.claude-csd-synthesis` and expects the configured
`ssdear@gmail.com` Claude account. Startup also requires a clean
`legacy/CRANE` checkout at commit
`616379ce33ac6245933c16e6264b41f7d5800183`.

Cold reruns use the canonical train split selected by `synthesis.run_synthesis`
and set `CSD_OUTPUT_NAME`; they do not pass unsupported output-name or split
flags.

Use `--dry-run` to print the exact fixed-baseline and possible cold rerun
commands. It never creates a claim or calls an evaluator. A real queue uses
only the GPUs named by `--gpus` (default `0,1,2,3`) and admits a cell only when
the live free memory, existing reservations, every worker's demand, and the
2 GiB margin fit. It does not displace another user's process.

After a valid result, the queue compares the frozen same-row metaDecode
artifact. GSM and Spider trigger a single atomic cold 40-iteration rerun when
accuracy **or** syntax is strictly higher; SMILES uses strict unique-valid
rate. Ties do not trigger. The existing post-14B claim helper is used, and a
claim is never removed after an interruption or failure.

Rerun state records `phase=synthesis` or `phase=heldout` plus child PID/start
identity. On restart, a completed synthesis report is recovered through the
cold queue's success/exhaustion selection, while a held-out child is waited
for or restarted only from its hash-pinned compiled CSD. No author attempt is
repeated merely because the controller restarted.

## Tables 5--8 synthesis queue

The missing Table 5--8 synthesis cells are described by
`run_table5_8_queue.py`. It builds exactly 11 GSM-Symbolic rows: 3 Table 5
author-model runs, 3 Table 6 token-budget runs, 3 Table 7 beam-size runs, and 2
Table 8 mask runs. All rows use the exact `Qwen/Qwen3.5-2B` evaluator and at
most 40 cold synthesis attempts. The export records held-out accuracy, syntax
rate, attempts used, accepted/exhausted status, and constrained work for every
row. Table 5's author profiles are GPT-5.6 Sol through the pinned Pi
provider-only layer with ChatGPT/Codex OAuth, Gemini
3.7 Flash through the direct Google AI Studio API, and Claude Opus 5 through
the approved first-party Max account.

```bash
python scripts/runtime/run_table5_8_queue.py --dry-run
```

The dry run prints executable commands without calling providers or claiming
work. A real manifest binds the current commit, the approved CRANE commit,
and hashes for every runtime dependency. The controller records synthesis and
held-out phases in locked, replace-written state files, preserves output
provenance, and waits for free GPUs instead of forcing a dispatch. Provider
preflight only checks local configuration; it does not spend credits. The
Gemini route reads only `GEMINI_API_KEY` from the canonical private
`synthesis/.env`, passes no Vertex or backup credential to the author child,
and stores only the successful key's SHA-256 fingerprint in sealed evidence.
