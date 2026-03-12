#!/bin/bash
# Make CSD for FOLIO using Qwen 3B (lighter).
# Usage: bash scripts/make_csd/folio_qwen3b.sh
set -e

# FOLIO task description for FOL constrained decoding (verbose for Qwen)
TASK_DESC="Generate first-order logic (FOL) formulas for FOLIO logical reasoning. \
The parser enforces a strict FOL grammar with quantifiers, predicates, and logical connectives. \
CRITICAL RULES: \
1. Use {forall} and {exists} for quantifiers, followed by a single lowercase variable and a formula. \
2. Predicates start with uppercase (e.g., Dog(x), Likes(john, mary)) with lowercase arguments. \
3. Single lowercase letters (x, y, z) are variables; multi-character lowercase identifiers are constants. \
4. Logical connectives: {and}, {or}, {not}, {implies}, {iff}, {xor}. \
5. The constrained windows contain complete FOL statements — one per premise and one for the conclusion. \
6. Parentheses are used for grouping; operator precedence is: iff < implies < xor < or < and < not."

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
  --eval-sample-size 10

echo "FOLIO CSD (Qwen 3B) done. Run dir: outputs/generated-csd/runs/ (see latest_run.txt)"
