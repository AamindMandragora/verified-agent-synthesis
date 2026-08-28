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
