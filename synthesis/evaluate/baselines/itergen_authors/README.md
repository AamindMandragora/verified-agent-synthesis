# IterGen authors' SQL grammar

`sql_grammar.lark` is copied unmodified from the IterGen paper's own
repository: https://github.com/structuredllm/itergen, commit
`52d42129f36d0d272336f33e4c67b1ae3f3b51a2`, file
`itergen/syncode/syncode/parsers/grammars/sql_grammar.lark`. That repo is
itself gitignored out of this worktree (`legacy/itergen`, 52MB, not tracked
by git), so this one file is pinned here instead of vendoring the whole
library.

Why we need it: IterGen's own Spider case study asks `view()`/`backward()`
for the grammar symbols `column_name` and `table_name`, which this file
defines. Our own `synthesis/evaluate/grammars/sql.lark` uses different names
(`column_ref` / `table_ref`), so passing it to the Spider IterGen adapter
made `view()` always return empty lists -- the back-up-and-retry step could
never fire, and the "IterGen" lane was really just one unconstrained decode.
`_run_itergen_legacy_adapter_inner` in `run_legacy_fixed_strategy.py` loads
this file instead, for the Spider IterGen lane only; every other Spider lane
(GCD, CRANE, unconstrained, our own method) keeps using
`synthesis/evaluate/grammars/sql.lark`.

Original file license: the file carries its own BSD-3-Clause header
(adapted from https://github.com/zbrookle/sql_to_ibis and
https://github.com/lapp0/outlines), reproduced verbatim below the header in
this copy. The IterGen repository itself is MIT-licensed (see its `LICENSE`
file).
