# Corrected full-baseline queue launch

Date: 2026-08-05
Launch time: 12:54:20 UTC

## Outcome

The independently approved corrected synthesis queue is running on focal.

- Controller PID: `3111473`
- Controller process group: `3111473`
- Queue profile: `full-baseline-corrected-20260805`
- Allowed physical GPUs: `0,2,3`
- GPU `1` compute processes at verification: `0`
- Planned new author calls: `675`
- Controller log: `logs/full-baseline-corrected-20260805-cold-controller.log`
- Combined run log: `logs/full-baseline-corrected-20260805-combined.log`
- State directory: `.context/full-baseline-corrected-20260805-cold-state`
- PID file: `.context/full-baseline-corrected-20260805-cold.pid`

## Bound inputs

- Pinned code commit: `6766ebd397f4bcdbd4cc3332e1051a9efd6258a6`
- Corrected evidence SHA-256:
  `57392e149cea23efe6f596b921c6cd3d74ae7519e7c286393adadc9bdb579ab7`
- Queue manifest SHA-256:
  `c5f5683311ecf8f03211fb4fd8572b92e472dc5475367afe72413eb31098aa18`
- Approval JSON SHA-256:
  `d43a5cef6c9484430fc87e2a7dca4e6bdf963148d0ff5a7769c6422b4a602b9a`
- Archived block SHA-256:
  `c882e6ae8178349ea6006933d17b1b8105c73a20612c8d800ec79df16057a4a1`

## Launch transition

1. The old exact-zero monitor PID `336923` was checked as its own process-group
   leader and stopped with `SIGTERM`; it was confirmed dead before launch.
2. The active block file was moved without changing its bytes to
   `saved-results/2026-08-05-exact-zero-repair-synthesis-block-resolved.json`.
3. The controller started with the exact approval JSON, manifest, code commit,
   and `--gpus 0,2,3` gate.
4. The controller and three phase-one synthesis children remained alive through
   the post-launch check. Their physical GPU environments were `3,0`, `2`, and
   `3`; none included GPU `1`.
5. Approval revalidation returned the exact `675`-call plan after launch.

## First phase observed

- `spider-qwen35-2b`: GPUs `3,0`, 40 attempts.
- `smiles-acrylates-qwen35-2b`: GPU `2`, 40 attempts.
- `smiles-chain_extenders-qwen25-1p5b`: GPU `3`, 40 attempts.

The controller may share an approved GPU when its memory reservations fit; the
queue never considers GPU `1`. Later phases remain blocked by the strict phase
barrier until every earlier-phase job has reached a terminal state.

## Spider GPU 1 replacement — 2026-08-19

### Purpose

Allow the Spider-only controller to consider GPU `1` without weakening the
shared-memory checks that protect other users' jobs.

### Result

- Replacement launch time: `08:37:03 UTC`.
- Controller PID: `1663018`.
- Pinned code commit: `189a647061e40bdaffdd312fc34e9e89999a5e29`.
- Exact GPU scope: `0,1,2,3`.
- Exact exclusions: `gsm-` and `smiles-`; four Spider cells remain.
- Controller log: `logs/spider-only-relaunch-20260819-gpu1-controller.log`.
- State directory: `.context/full-baseline-corrected-20260805-cold-state`.
- Lock and PID directory: `.context/spider-only-relaunch-20260819-gpu1/`.
- Queue manifest SHA-256:
  `06c285b2c948c16d9d09b3473ed34ed08ff12ac7efd81bbaaf767d53a0a4d05c`.
- Approval SHA-256:
  `3b3c26bb62edccf6fa056098fd17e6c9e97f74f009e54043222b179044708a2c`.
- Independent `gpt-5.6-sol` verdict: pass for the exact commit, manifest,
  approval, GPU scope, and preserved memory gate.

The old PID `2919913` was confirmed to have no children, stopped with
`SIGTERM`, and confirmed absent before the replacement launched. Its log and
the shared state directory were preserved. The replacement's first poll found
no safe two-GPU bundle and waited 30 seconds; it did not dispatch or report an
error. At that poll, used memory was GPU 0=`37274 MiB`, GPU 1=`32403 MiB`, GPU
2=`29938 MiB`, and GPU 3=`27601 MiB`.

### Verification

- Red tests: the new four-GPU scope tests first failed twice against the old
  `0,2,3` validator.
- Green tests: `68 passed` in `tests/runtime/test_cold_synthesis_queue.py`.
- Full runtime tests: `224 passed`.
- Candidate approval validation: campaign name matched, all `20` jobs were
  present, and the manifest commit matched `189a6470`.
- Live check: exactly one controller, exact `--gpus 0,1,2,3`, old PID absent,
  four Spider jobs remaining, and no immediate launch error.

### Reuse

Check the current process and log before relying on this point-in-time result:

```bash
ssh aadivyar@focal "bash -lc 'ps -fp 1663018; tail -n 30 /home/aadivyar/csd-generation-worktrees/full-baseline-campaign-20260803/logs/spider-only-relaunch-20260819-gpu1-controller.log'"
```
