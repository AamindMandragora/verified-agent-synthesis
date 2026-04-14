#!/bin/bash
# Make CSD for PDDL Blocks World using Qwen 7B (stronger, slower).
# Usage: bash synthesis/shell/pddl_qwen7b.sh
set -e

TASK_DESC="Generate a PDDL planning strategy for Blocks World problems. \
The grammar enforces a strict sequence of PDDL action applications: (pick-up X), (put-down X), (stack X Y), (unstack X Y) where X and Y are single lowercase block letters. \
CRITICAL RULES: \
1. The constrained window contains one or more actions with no other text. \
2. Actions must satisfy their preconditions: pick-up requires hand empty and block clear on table; put-down requires holding block; stack requires holding top and bottom is clear; unstack requires hand empty and top is on bottom and top is clear. \
3. The plan is evaluated by simulation - it must achieve the stated goal from the initial state. \
4. Use UnconstrainedStep for any reasoning text before <<; use ConstrainedStep only for the action sequence inside << >>. \
5. Plans are typically short (2-8 actions); prefer correct minimal plans."

echo "Making PDDL CSD (Qwen 7B)..."
echo "Task: $TASK_DESC"
echo ""

python run_synthesis.py \
  --task "$TASK_DESC" \
  --dataset pddl \
  --max-iterations 10 \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --output-name "pddl_csd" \
  --temperature 0.7 \
  --device auto \
  --min-accuracy 0.3 \
  --min-format-rate 0.5 \
  --min-syntax-rate 0.5 \
  --eval-sample-size 5 \
  --eval-max-steps 128

echo "PDDL CSD (Qwen 7B) done. Run dir: outputs/ (see latest_run.txt)"
