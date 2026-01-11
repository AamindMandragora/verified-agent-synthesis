#!/bin/bash
# Generate a GSM-specific CSD for CRANE-style math reasoning
#
# Usage: bash scripts/generate_gsm_csd.sh

set -e

# GSM task description emphasizing SHORT constrained windows
TASK_DESC="Generate short mathematical expressions for GSM-Symbolic reasoning. \
The parser enforces a strict arithmetic expression grammar (addition, subtraction, multiplication, division with numbers). \
CRITICAL: The constrained windows are VERY SHORT (typically 5-20 tokens for expressions like '5 + 3 * 2'). \
The grammar does NOT include stopping delimiters - expressions must be complete but compact. \
Since the windows are short and the grammar is simple, prefer PureConstrainedGeneration or \
SpeculativeGeneration with a small window (N=3-4) to generate compact valid expressions quickly. \
DO NOT use strategies that encourage long generation - we need to generate complete expressions in under 20 tokens."

echo "🚀 Generating GSM-specific CSD for CRANE math windows..."
echo ""
echo "Task description:"
echo "  $TASK_DESC"
echo ""

# Run synthesis
python run_synthesis.py \
    --task "$TASK_DESC" \
    --max-iterations 10 \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-name "gsm_crane_csd" \
    --temperature 0.7 \
    --device auto

echo ""
echo "✅ GSM CSD generation complete!"
echo ""
echo "To use the generated CSD, run:"
echo "  python scripts/evaluate_gsm_symbolic.py \\"
echo "    --method crane-csd \\"
echo "    --run-dir outputs/generated-csd/latest \\"
echo "    --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "    --device cuda \\"
echo "    --limit 50 \\"
echo "    --max-steps 1024 \\"
echo "    --vocab-size 2000"
