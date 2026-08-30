# Table 5–8 Gemini 3.7 Flash change plan

Date: 2026-08-29

## Target

```text
private focal .env
  GEMINI_API_KEY
        |
        v
Table 5–8 controller ----> direct Gemini API ----> gemini-3.7-flash
        |                         |
        |                         +--> one provider pilot after commit
        v
manifest/report/export
  key SHA-256 only; never the key
```

The change is done when every Table 5–8 Gemini row uses profile
`gemini3.7-flash`, backend `gemini`, and model `gemini-3.7-flash`; the direct
AI Studio key is the only Google author credential passed to synthesis; held-out
evaluation receives no author credential; and reports, pilots, manifests, and
exports bind the SHA-256 fingerprint of the key that actually succeeded.

## Steps

1. Add tests that prove the 31-row scope is unchanged and the old Gemini 3.1
   Pro/Vertex route is gone.
2. Add tests for direct-key loading, credential isolation, exact-model auth
   probing, key-fingerprint provenance, pilot validation, and held-out stripping.
3. Run those tests before implementation and save the expected failures.
4. Replace the old profile in the queue and remove its Vertex project/ADC fields.
5. Make direct Gemini author provenance record only the successful key's SHA-256.
6. Load the key from the existing private credential file without copying it to
   the worktree, logs, reports, manifest, or saved results.
7. Update the nearby runtime/generator documentation and paper-facing profile
   label; leave the active baseline controller and its artifacts unchanged.
8. Run focused tests, the full runtime suite, formatting checks, and a repository
   search for old profile/model/Vertex-only assumptions.
9. Commit one exact candidate and give it to a fresh Sol judge for independent
   review against the requested behavior and launch safety.
10. After approval, create a new sealed manifest and run a fresh one-attempt
    Gemini pilot only when a campaign GPU is safely available. Do not launch the
    31-row campaign until Codex, Opus, Gemini, disk, and GPU gates all pass.

## Evidence to retain

- failing and passing test commands with counts
- final commit, tree, and source hashes
- exact profile/backend/model and non-secret key fingerprint
- pilot report, compiled strategy, and manifest hashes after the live pilot
- independent review verdict

## Rollback

The active baseline queue is separate and is never touched. Before the new
Table 5–8 controller launches, rollback is simply keeping the current controller
stopped and discarding the new sealed package. No old Vertex route remains as a
runtime fallback in the replacement code.
