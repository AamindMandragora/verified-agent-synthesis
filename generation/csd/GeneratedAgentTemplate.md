# `generation/csd/GeneratedAgentTemplate.py` Guide

This document explains the purpose of `generation/csd/GeneratedAgentTemplate.py` and how it fits into the synthesis pipeline.

## What It Is

`GeneratedAgentTemplate.py` is the Python template used to synthesize a new strategy implementation.
The generator loads this file, replaces the strategy hole with model-produced code, and then sends the result through the transpilation and verification pipeline.

## What Must Stay Stable

- The template must continue to expose the expected class and method shape used by `synthesis/generator.py`.
- The strategy hole markers must remain intact so the generator can inject code safely.
- Imports from `VerifiedAgentSynthesis.py` must stay compatible with the temporary workspace layout used during Dafny compilation.

## How It Is Used

1. `generation/generator.py` loads `generation/csd/GeneratedAgentTemplate.py`.
2. The model writes only the strategy body, not a whole file.
3. The generated body is inserted between the hole markers.
4. The completed file is transpiled and verified against the contracts defined in `generation/csd/VerifiedAgentSynthesis.py`.

## Editing Guidance

- Keep the template small and predictable.
- Prefer changing contracts in `VerifiedAgentSynthesis.py` only when the whole pipeline truly needs it.
- If you change the template structure, also update generator assumptions and tests that validate hole replacement.
