# Reference CSD strategies (Dafny)

Verified reconstructions of published strategy families (Unconstrained, GCD/SynCode,
CRANE, IterGen, CARS) built only from `CSDHelpers` in
[`../library/VerifiedAgentSynthesis.dfy`](../library/VerifiedAgentSynthesis.dfy).

**Each file here is a strategy BODY, not a whole Dafny file.** There is exactly one
copy of the decoder contract, in
[`../library/GeneratedCSD.dfy`](../library/GeneratedCSD.dfy); a body is inserted at
that template's `// QWEN_INSERT_STRATEGY_HERE` marker, the same way a synthesized
strategy is. So a reference carries no `include`, no `module`, and no
`method MyCSDStrategy` of its own, and it is held to the current contract instead of
a stale copy of an older one. `synthesis/evaluate/run_reference_strategy.py` builds
the full file with `synthesis.generate.generator.inject_strategy_into_template`.

The template pre-declares the body's variables: `helpers` (a fresh `CSDHelpers`),
`generated`, `insideConstrainedOut`, `currentConstrainedOut` (initialised to the
incoming state) and `cost`. A body must keep
`Tied(parser, generated, insideConstrainedOut, currentConstrainedOut)` as a loop
invariant and end with `cost := helpers.cost;`.

Two entry states must both work: GSM starts outside a span with an empty prefix,
while Spider and SMILES start the body already inside one
(`generatedPrefix == ["<<"]`, `insideConstrained == true`).

For a concise index of every `LM`, `Parser`, and `CSDHelpers` member, see
[`../library/README.md`](../library/README.md).

## Files

| File | Role |
|------|------|
| `unconstrained.dfy` | **Unconstrained generation.** Pure `UnconstrainedStep` loop, no grammar enforcement. The lower-bound baseline for syntax validity. Two concessions to the contract, neither of which constrains the model: entering inside a span it rolls that span back and drops the opener before free-running, and if the model never emits `<<` it spends one held-back step opening an empty span at the very end. |
| `gcd.dfy` | **Greedy Constrained Decoding (SynCode-style).** Opens a `<<` span at token 0 and hard-masks every token; no unconstrained reasoning, no group boosting, rollback, or adaptivity. Closes with `>>` when the parse is complete. |
| `crane.dfy` | **CRANE-style.** Free-text reasoning; inside `<<`…`>>`, `GroupBoostedConstrainedStep` with empty groups (hard mask only); closes as soon as the parse is complete. |
| `crane_faithful.dfy` | **CRANE, IterGen-unit variant.** Free text until `<<`, then `ForwardUntilSymbol("start", 1)` per round with the `var` grounding check and `BackwardToSymbol` backtrack, matching CRANE's `generate_gsm_symbolic_with_itergen`. Same loop `CSDHelpers.CraneGeneration` runs, lifted into the body so the span state it tracks is the contract's own — `CraneGeneration` generates from an empty prefix and exposes no `Tied` postcondition, so it cannot be used under the span contract. |
| `itergen.dfy` | **IterGen-style.** Greedy grammar-masked decoding with schema-grounded symbol-boundary backtracking, delegated to `RegenerateUnitOnGroundingFailure` (recurrence penalty 0.3, `backwards_limit` 10). |
| `cars.dfy` | **CARS-style (full adaptive rejection sampling).** `ConstrainedStep` for the first constrained token (`constrain_first`); `SoftConstrainedStep` with zero boost for exploration (unconstrained, like new trie nodes); on grammar violation the attempt is rejected, the failing token is penalised, and the span is rolled back to its entry point; retries use `SafePenalizedConstrainedStep` (hard mask + accumulated penalties, like revisited trie nodes with `log_theta`). |

## Design distinction: GCD vs CRANE

GCD forces constrained mode from the first token (`OpenConstrainedSpan` immediately,
all model-chosen tokens are grammar-masked). CRANE allows unconstrained reasoning
before `<<` and uses adaptive switching. On benchmarks like Spider/SMILES where the
output is a formal expression, CRANE can reason ("I need to join these tables...")
before emitting `<<SELECT ...>>`, while GCD constrains from token 1.

## Verify

A body cannot be verified on its own — it has to go into the template. The test that
does this for all six is:

```bash
python -m pytest tests/verify/test_span_contract_dafny.py -q
```

To check one by hand, insert it at the marker in a copy of
`synthesis/verify/library/GeneratedCSD.dfy` (kept in that directory so the relative
`include` resolves) and run `dafny verify` on the copy.
