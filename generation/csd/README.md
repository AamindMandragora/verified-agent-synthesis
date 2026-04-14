# CSD Core Files

This folder contains the Python-authored core assets for the constrained decoding stack:

- `VerifiedAgentSynthesis.py`: the verified helper library and contract surface
- `GeneratedAgentTemplate.py`: the synthesis template with the strategy hole
- `VerifiedAgentSynthesis.md`: a reference for the helper library API
- `GeneratedAgentTemplate.md`: a guide to the template structure and the strategy hole

These files are the source of truth for the Python-first synthesis pipeline.
The `dafny/` folder remains a generated or toolchain-facing intermediate and is not the main authoring surface.
