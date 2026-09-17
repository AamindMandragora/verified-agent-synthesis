"""Disposable per-change TDD test for Change 3 (persistent tried-token penalty).

Verifies the mechanism directly on _TensorizedLMBase, no model/vLLM load:
  T1  byte-identical no-op when nothing is penalized (the fairness guarantee)
  T2  divergence: penalizing the argmax token makes the regen pick a different one
  T3  RED reproduction: WITHOUT the persistent re-apply (the old one-shot mask that
      GenerateLogits wipes), the regen reproduces the same token (the bug)
  T4  cumulative penalty eventually demotes even a confident token within budget
  T5  no cross-prefix contamination
  T6  per-example reset (penalties never leak across instruction_text)
"""
import os, sys, math
os.environ["CSD_RECURRENCE_PENALTY"] = "0.3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from types import SimpleNamespace
import model_utils as M


def make_lm(instr="EXAMPLE-1 "):
    lm = M._TensorizedLMBase(_dafny=None, tokenizer=None,
                             tokens=["A", "B", "C"], tids=[0, 1, 2],
                             logits_device="cpu")
    lm.instruction_text = instr
    return lm


def lp(a, b, c):
    return {0: SimpleNamespace(logprob=a),
            1: SimpleNamespace(logprob=b),
            2: SimpleNamespace(logprob=c)}


P = ["x"]
P2 = ["y"]
LN03 = math.log(0.3)

# T1: byte-identical no-op
lm = make_lm()
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
before = lm._logits_tensor.clone()
lm._apply_recurrence_penalty(lm.instruction_text + lm._prefix_text(P))  # empty map
assert lm.ChooseNextToken() == "A"
assert torch.allclose(lm._logits_tensor, before), "T1 not byte-identical"
print("T1 PASS: empty-map no-op is byte-identical; argmax=A")

# T2: divergence after penalizing the argmax
lm = make_lm()
key = lm.instruction_text + lm._prefix_text(P)
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
assert lm.ChooseNextToken() == "A"
lm.PenalizeTriedTokenAt(P, "A")
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))   # regen = fresh logits
lm._apply_recurrence_penalty(key)
c = lm.ChooseNextToken()
assert c == "B", f"T2 expected B got {c}"
assert abs(lm._logits_tensor[0].item() - (-0.1 + LN03)) < 1e-4, "T2 wrong penalty amount"
print(f"T2 PASS: diverged A->B; A logprob -0.100 -> {lm._logits_tensor[0].item():.3f}")

# T3: RED reproduction (no persistent re-apply => same token)
lm = make_lm()
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
assert lm.ChooseNextToken() == "A"
lm.PenalizeTriedTokenAt(P, "A")
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))   # fresh logits = old MaskToken wiped
# deliberately DO NOT call _apply_recurrence_penalty (simulates the wipe bug)
c = lm.ChooseNextToken()
assert c == "A", f"T3 expected A (bug) got {c}"
print("T3 PASS: without persistent re-apply, regen reproduces A (the wipe bug)")

# T4: cumulative penalty eventually demotes a confident token
lm = make_lm()
key = lm.instruction_text + lm._prefix_text(P)
def fin(): lm._finalize_from_logprob_dict(lp(0.0, -3.0, -9.0))
fin(); assert lm.ChooseNextToken() == "A"
seq = []
for _ in range(3):
    lm.PenalizeTriedTokenAt(P, "A"); fin(); lm._apply_recurrence_penalty(key)
    seq.append((lm.ChooseNextToken(), round(lm._logits_tensor[0].item(), 3)))
assert [s[0] for s in seq] == ["A", "A", "B"], f"T4 unexpected {seq}"
print(f"T4 PASS: cumulative demote {seq} (A: 0 -> {seq[-1][1]} over 3 tries -> B)")

# T5: no cross-prefix contamination
lm = make_lm()
keyP2 = lm.instruction_text + lm._prefix_text(P2)
lm.PenalizeTriedTokenAt(P, "A")
lm._finalize_from_logprob_dict(lp(-0.1, -0.5, -2.0))
lm._apply_recurrence_penalty(keyP2)
assert lm.ChooseNextToken() == "A", "T5 penalty leaked to other prefix"
print("T5 PASS: penalty at P does not affect P2")

# T6: per-example reset
lm = make_lm("EXAMPLE-1 ")
lm.PenalizeTriedTokenAt(P, "A")
assert len(lm._tried_token_penalties) == 1
lm.instruction_text = "EXAMPLE-2 "
lm.PenalizeTriedTokenAt(P, "B")   # triggers reset of EXAMPLE-1 entries
keys = list(lm._tried_token_penalties.keys())
assert keys and all(k.startswith("EXAMPLE-2 ") for k in keys), f"T6 stale keys {keys}"
print("T6 PASS: penalty map reset on instruction_text change")

print("\nALL TESTS PASSED")
