#!/bin/bash
# Make CSD for FOLIO using Qwen 3B (lighter).
# Usage: bash scripts/make_csd/folio_qwen3b.sh
set -e

# FOLIO: allow multiple << formula >> windows; evaluation uses the last one. Plain text unconstrained, then constrained inside delimiters.
TASK_DESC="FOLIO: the strategy must generate plain text first (unconstrained), then one or more \" << \" formula \" >>\" segments (Prover9 grammar: {forall}, {exists}, predicates, {and}, {or}, {not}, {implies}, {iff}, {xor}). Evaluation uses only the last << >> segment. Use UnconstrainedStep for plain text; use ConstrainedStep only for the segment between the delimiters. Respect the grammar only between the delimiters. All structure instructions are in the prompt."

echo "Making FOLIO CSD (Qwen 3B)..."
echo "Task: $TASK_DESC"
echo ""

python run_synthesis.py \
  --task "$TASK_DESC" \
  --dataset folio \
  --max-iterations 10 \
  --model "Qwen/Qwen2.5-Coder-3B-Instruct" \
  --output-name "folio_csd" \
  --temperature 0.7 \
  --device auto \
  --min-accuracy 0.5 \
  --min-format-rate 0.8 \
  --min-syntax-rate 0.8 \
  --eval-sample-size 3

echo "FOLIO CSD (Qwen 3B) done. Run dir: outputs/generated-csd/runs/ (see latest_run.txt)"
