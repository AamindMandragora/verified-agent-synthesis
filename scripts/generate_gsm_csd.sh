#!/bin/bash
# Generate a GSM-specific CSD for CRANE-style math reasoning
#
# Usage: bash scripts/generate_gsm_csd.sh

set -e

# GSM task description - bare description of the task, no strategy hints
TASK_DESC="Math word problem solving where the model reasons in natural language and writes \
arithmetic expressions inside << >> delimiters. The parser validates expression syntax automatically."

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
    --min-accuracy 0.3 \
    --min-format-rate 0.5 \
    --min-syntax-rate 0.5 \
    --eval-sample-size 10 \
    --eval-max-steps 512

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
echo "   --vocab-size 3000 \\"
echo "   --load-in-4bit"
