#!/bin/bash
# Make CSD for FOLIO using Qwen 3B (lighter).
# Usage: bash scripts/make_csd/folio_qwen3b.sh
set -e

# FOLIO: structure (plain text, then << FOL >>) is in the prompt; parser allows that. Pure LLM CSD.
TASK_DESC="FOLIO: generate output that may include plain text, then \" << \" then exactly one \
first-order logic formula (Prover9 grammar: {forall}, {exists}, predicates, {and}, {or}, {not}, {implies}, {iff}, {xor}), then \" >>\". \
The constrained decoder must respect the grammar only between the delimiters. \
All structure instructions are in the prompt; use your strategy to satisfy the grammar."

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
