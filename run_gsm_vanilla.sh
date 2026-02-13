#!/bin/bash
set -e

# script to evaluate gsm symbolic using the vanilla CRANE system without any dynamic CSD generation

# 1. Ensure GeneratedCSD.dfy contains the vanilla CRANE strategy
# We look for "PureConstrainedGeneration" to see if it's already the vanilla version.
# If not, or if we want to be sure, we overwrite it.

echo "Setting up vanilla CRANE strategy in dafny/GeneratedCSD.dfy..."

cat > dafny/GeneratedCSD.dfy <<EOF
include "VerifiedAgentSynthesis.dfy"

module GeneratedCSD {
  import opened VerifiedDecoderAgent

  method MyCSDStrategy(lm: LM, parser: Parser, prompt: Prefix, maxSteps: nat) returns (generated: Prefix)
    modifies lm.Logits
    requires lm.ValidTokensIdsLogits()
    requires parser.IsValidPrefix([])
    requires forall t: Token :: t in parser.ValidNextTokens([]) ==> t in lm.Tokens
    ensures lm.ValidTokensIdsLogits()
    ensures |generated| <= maxSteps
    ensures parser.IsValidPrefix(generated)
    ensures |generated| == maxSteps || parser.IsCompletePrefix(generated)
  {
    generated := CSDHelpers.PureConstrainedGeneration(lm, parser, prompt, maxSteps);
  }
}
EOF

# 2. Compile the strategy (skip generation)
echo "Compiling strategy..."
python run_synthesis.py --task "Vanilla CRANE" --compile-only --output-name generated_csd > /dev/null

# 3. Get the run directory
RUN_DIR=$(cat outputs/generated-csd/latest_run.txt)
echo "Using compiled run: $RUN_DIR"

# 4. Run the evaluation
# Pass all arguments to this script forward to the evaluation CLI
echo "Starting evaluation..."
echo "Arguments: $@"

python -m evaluations.gsm_symbolic.cli \
  --run-dir "$RUN_DIR" \
  "$@"
