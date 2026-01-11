#!/bin/bash
# Generate a SQL-specific CSD for Spider evaluation
#
# Usage: bash scripts/generate_sql_csd.sh

set -e

# SQL task description that emphasizes the need for schema-aware constrained generation
TASK_DESC="Generate SQL queries for the Spider text-to-SQL dataset. \
The parser enforces a strict SQL grammar with schema-specific constraints: \
table and column names must exactly match the database schema. \
The grammar is VERY STRICT - it only allows valid SQL tokens and exact schema identifiers. \
Since the parser heavily restricts the valid token set at each step, \
the strategy should prioritize FULLY CONSTRAINED generation to ensure every token is valid. \
Queries typically need 30-100 tokens. The parser validates every single token against the SQL grammar, \
so prefer PureConstrainedGeneration or strategies with minimal unconstrained steps to avoid rejection waste."

echo "🚀 Generating SQL-specific CSD..."
echo ""
echo "Task description:"
echo "  $TASK_DESC"
echo ""

# Run synthesis with appropriate settings
python run_synthesis.py \
    --task "$TASK_DESC" \
    --max-iterations 10 \
    --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --output-name "sql_csd" \
    --temperature 0.7 \
    --device auto \
    --no-save-reports

echo ""
echo "✅ SQL CSD generation complete!"
echo ""
echo "To use the generated CSD, run:"
echo "  python scripts/evaluate_spider_sql.py \\"
echo "    --spider-root /path/to/spider \\"
echo "    --method csd \\"
echo "    --run-dir outputs/generated-csd/latest \\"
echo "    --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "    --device cuda \\"
echo "    --schema-aware \\"
echo "    --limit 10 \\"
echo "    --max-steps 256 \\"
echo "    --vocab-size 1500"