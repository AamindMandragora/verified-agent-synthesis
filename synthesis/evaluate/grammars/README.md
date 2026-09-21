# Evaluation Grammars

This directory stores Lark grammar files used to constrain token generation during synthesis evaluation.

## What These Grammars Control

- Which next tokens are valid under the current parser state.
- How syntax validity is checked for produced constrained spans.
- Dataset-specific structural contracts (math spans, SQL forms, molecule syntax, etc.).

## Active Grammar Families

- GSM-Symbolic grammar (`gsm.lark`)
- SQL grammar (`sql.lark`), copied byte-for-byte from SynCode commit
  `4d6c110a028d3fb9cbb5f1a6ac6446b696f3eac4`
  (`syncode/parsers/grammars/sql.lark`, SHA-256
  `405cb737863b63bb0fb8d832268443bcb5a89089f6691406b37d95e052a1a2ac`).
  Spider uses its `start` rule with Lark's `basic` lexer, matching SynCode's
  incremental-mask setup.
- SMILES class grammars:
  - `smiles_acrylates.lark`
  - `smiles_chain_extenders.lark`
  - `smiles_isocyanates.lark`
- Utility grammar for parser/runtime support (`json.lark`)

## Performance Note

These grammars are consumed through Syncode-backed DFA mask stores.
That path is essential for fast valid-next-token lookup and should remain the default runtime mode.
