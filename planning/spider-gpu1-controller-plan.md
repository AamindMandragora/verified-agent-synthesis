# Spider GPU 1 Controller Plan

Date: 2026-08-19

```text
Spider queue
    |
    v
Check GPUs 0,1,2,3 every 30 seconds
    |
    +-- fewer than two safe GPUs --> wait
    |
    `-- two safe GPUs -----------> launch one Spider cell
```

## Done when

- The approved Spider-only scope accepts exactly GPUs `0,1,2,3`.
- The old `0,2,3` scope is rejected so the controller cannot silently omit GPU 1.
- Existing memory checks still choose only GPUs with enough room for every worker.
- Targeted tests pass, the approval is re-pinned to the reviewed commit, and the live controller polls all four GPUs.

## Work

1. Add a failing scope test for `0,1,2,3` and for rejecting `0,2,3`.
2. Change the exact Spider resume scope and nearby operating instructions.
3. Run targeted and full runtime tests.
4. Have an independent judge check safety, scope, and launch evidence.
5. Re-pin the manifest and approval, restart the waiting controller, and verify its first live poll.

## Safety

- Do not stop another user's GPU process.
- Adding GPU 1 changes only the allowed set; the existing memory reservation gate remains mandatory.
- Preserve the old controller log and state directory.
