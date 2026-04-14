#!/bin/bash
# Make CSD for SyGuS SLIA using Qwen 7B (stronger, slower).
# Usage: bash synthesis/shell/sygus_slia_qwen7b.sh
set -e

TASK_DESC="Generate a string manipulation strategy for SyGuS SLIA problems. \
The grammar enforces a strict S-expression format using operations: str.++, str.substr, str.indexof, str.len, str.at, str.replace, str.upper, str.lower, and integer arithmetic (+, -). \
CRITICAL RULES: \
1. The constrained window is a SINGLE S-expression that computes the output string from the input variable. \
2. Variables (e.g. name, email, s, filename) are bare identifiers; string literals are double-quoted (e.g. \" \"). \
3. Integer arguments to str.substr and str.at must be int expressions (INT_LIT, str.len, str.indexof, +, -). \
4. str.indexof returns -1 when the substring is not found; use it with str.substr to split strings. \
5. Use UnconstrainedStep for any plain text before the << delimiter; use ConstrainedStep only inside the window. \
6. The expression must be complete and syntactically valid before >>."

echo "Making SyGuS SLIA CSD (Qwen 7B)..."
echo "Task: $TASK_DESC"
echo ""

python run_synthesis.py \
  --task "$TASK_DESC" \
  --dataset sygus_slia \
  --max-iterations 10 \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --output-name "sygus_slia_csd" \
  --temperature 0.7 \
  --device auto \
  --min-accuracy 0.3 \
  --min-format-rate 0.5 \
  --min-syntax-rate 0.5 \
  --eval-sample-size 5 \
  --eval-max-steps 256

echo "SyGuS SLIA CSD (Qwen 7B) done. Run dir: outputs/ (see latest_run.txt)"
