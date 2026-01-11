#!/bin/bash
# Quick test of fixed Spider SQL evaluation
# Tests with just 3 examples to verify improvements

set -e

echo "🧪 Quick Test: Spider SQL Evaluation with Fixes"
echo "================================================"
echo ""
echo "Testing with:"
echo "  - Improved token selection (schema names prioritized)"
echo "  - Reduced vocab size (1000 instead of 3000)"
echo "  - Reduced timeout (120s instead of 600s)"
echo "  - Only 3 examples for quick validation"
echo ""

python scripts/evaluate_spider_sql.py \
  --spider-root /home/aadivyar/spider_data/spider_data \
  --method csd \
  --run-dir outputs/generated-csd/runs/20260105_215059_4ee3a0 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --device cuda \
  --schema-aware \
  --limit 3 \
  --max-steps 256 \
  --vocab-size 1000 \
  --timeout 120 \
  --verbose

echo ""
echo "✅ Test complete!"
echo ""
echo "Compare results:"
echo "  - OLD: 130-250s per query, 0% accuracy, hallucinated columns"
echo "  - NEW: Should be 40-80s per query, higher accuracy, valid schema names"
echo ""
echo "Next steps:"
echo "  1. If improved, run full evaluation with --limit 100"
echo "  2. For best results, generate SQL-specific CSD:"
echo "     bash scripts/generate_sql_csd.sh"
