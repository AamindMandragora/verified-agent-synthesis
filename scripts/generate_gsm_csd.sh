#!/bin/bash
# Generate a GSM-specific CSD for CRANE-style math reasoning
#
# Usage: bash generate_gsm_csd.sh

set -e

# GSM task description emphasizing SHORT constrained windows with VARIABLES
TASK_DESC="Generate short symbolic mathematical expressions for GSM-Symbolic reasoning. \
The parser enforces a strict arithmetic expression grammar with VARIABLES and numeric constants. \
CRITICAL RULES: \
1. Every expression MUST contain at least one variable - pure numeric expressions like '2 * 2' are INVALID. \
    Variables may appear as letter+digits (n1, x2), split letter/digit tokens (n 1, x 2), or letter-only (n, x, foo). \
2. Variables represent problem values; numeric constants (12, 100) are for unit conversions and percentages. \
3. The constrained windows are SHORT (typically 5-20 tokens for expressions like 'n1 + n2 * 12'). \
4. The grammar is RECURSIVE but depth is BOUNDED in runtime; prefer balanced parentheses and compact expressions. \
5. The grammar includes the closing delimiter '>>' - expressions must be complete and compact. \
6. Avoid trivial outputs (e.g., just a single variable) unless the problem truly requires it; include necessary operators."

echo "Generating GSM-specific CSD for CRANE math windows..."
echo ""
echo "Task description:"
echo "  $TASK_DESC"
echo ""

# Run synthesis
python run_synthesis.py \
    --task "$TASK_DESC" \
    --dataset gsm_symbolic \
    --max-iterations 10 \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-name "gsm_crane_csd" \
    --temperature 0.7 \
    --device auto \
    --min-accuracy 0.8 \
    --min-format-rate 1.0 \
    --min-syntax-rate 1.0 \
    --eval-sample-size 10

echo ""
echo "GSM CSD generation complete!"
echo ""
echo "To use the generated CSD, run:"
echo "CUDA_VISIBLE_DEVICES=0,1 python -m evaluations.gsm_symbolic.cli \\"
echo "   --run-dir outputs/generated-csd/runs/latest \\"
echo "   --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "   --device cuda \\"
echo "   --limit 50 \\"
echo "   --max-steps 1024 \\"
echo "   --vocab-size 2000 \\"
echo "   --load-in-4bit"
