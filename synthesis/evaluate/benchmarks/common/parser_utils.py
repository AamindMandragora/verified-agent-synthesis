"""
Parser creation utilities for Dafny-compatible grammar parsers.

Uses syncode's DFA mask store for O(1) token validity checks instead of
O(vocab) brute-force Lark parsing.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from synthesis.evaluate.benchmarks.common.dafny_tokens import dafny_seq_to_str


# Per-component timing, shared conceptually with model_utils but kept separate
# so parser_utils doesn't depend on model_utils.
_PARSER_TIMINGS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
_PARSER_TIMINGS_ENABLED = os.environ.get("CSD_DISABLE_TIMING", "") == ""


@contextmanager
def _parser_timed(label: str):
    if not _PARSER_TIMINGS_ENABLED:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        slot = _PARSER_TIMINGS[label]
        slot[0] += elapsed
        slot[1] += 1


def print_parser_timings(header: str = "") -> None:
    if not _PARSER_TIMINGS_ENABLED or not _PARSER_TIMINGS:
        return
    total = sum(t for t, _ in _PARSER_TIMINGS.values())
    if total <= 0:
        return
    lines = [f"[PARSER_TIMING] {header} total={total:.2f}s"]
    for label in sorted(_PARSER_TIMINGS.keys(), key=lambda k: -_PARSER_TIMINGS[k][0]):
        secs, calls = _PARSER_TIMINGS[label]
        pct = 100.0 * secs / total
        avg_ms = 1000.0 * secs / max(calls, 1)
        lines.append(f"  {label:<34} {secs:7.2f}s  ({pct:5.1f}%)  calls={calls:<5} avg={avg_ms:7.2f}ms")
    print("\n".join(lines), flush=True)

_PARSER_COMPONENT_CACHE = {}
_DFA_MASK_STORE_CACHE = {}


def _tokenizer_cache_fingerprint(tokenizer) -> tuple[str, int]:
    name = getattr(tokenizer, 'name_or_path', tokenizer.__class__.__name__)
    try:
        vocab_size = len(tokenizer)
    except Exception:
        vocab_size = -1
    return str(name), int(vocab_size)


def _ensure_syncode_import_path() -> None:
    syncode_dir = Path(
        os.environ.get(
            "CSD_SYNCODE_DIR",
            str(Path(__file__).parent.parent.parent / "syncode"),
        )
    ).expanduser()
    # Vendored layout is synthesis/evaluate/syncode/syncode. We need
    # synthesis/evaluate/syncode on sys.path so imports like
    # `syncode.parsers` resolve correctly.
    candidates = [str(syncode_dir)]
    for candidate in reversed(candidates):
        if candidate in sys.path:
            sys.path.remove(candidate)
    for candidate in candidates:
        sys.path.insert(0, candidate)


def _load_grammar_text(grammar_source: str) -> str:
    if '\n' not in grammar_source and len(grammar_source) < 500:
        grammar_path = Path(grammar_source)
        if grammar_path.exists():
            return grammar_path.read_text()
    return grammar_source


def _get_parser_components(grammar_text: str, start: str, complete_start: str | None = None):
    _ensure_syncode_import_path()
    from lark import Lark
    from syncode.parsers.grammars import Grammar
    from syncode.parsers import create_base_parser

    complete_rule = complete_start if complete_start is not None else "start"
    component_key = (grammar_text, start, complete_rule)
    cached_components = _PARSER_COMPONENT_CACHE.get(component_key)
    if cached_components is None:
        grammar = Grammar(grammar_text)
        base_parser = create_base_parser(grammar)
        lark_parser = Lark(grammar_text, start=start, parser='lalr')
        # Separate parser for IsCompletePrefix / _is_complete.  The prefix/validity
        # parser above uses the CSD start rule (e.g. gsm csd_start = `any_expr ">>"`,
        # sql csd_start = `sql_stmt EOQ`), which REQUIRES the closing delimiter.  But
        # that closer (">>" / EOQ) is permanently grammar-masked during constrained
        # decoding, so checking "is this span content complete?" against csd_start can
        # never return True -> spans never close (the 0/0 wedge).  Completeness is about
        # the CONTENT rule "start" (any_expr / sql_stmt / smiles) WITHOUT the closer; the
        # strategy force-appends the closer itself via CloseConstrainedSpan.  Every
        # grammar (gsm/sql/smiles) defines a "start" rule, so this is uniform.  This only
        # changes the generation loop's completeness signal -- it does NOT touch scoring:
        # the syntax-rate metric uses a separate start="syncode" parser (eval_logic) that
        # still requires the literal "<<...>>".
        complete_lark_parser = Lark(grammar_text, start=complete_rule, parser='lalr')
        cached_components = (grammar, base_parser, lark_parser, complete_lark_parser)
        _PARSER_COMPONENT_CACHE[component_key] = cached_components
    return cached_components


def _get_cached_dfa_mask_store(grammar_text: str, grammar, tokenizer, dfa_mode: str = "grammar_mask"):
    if tokenizer is None:
        return None

    _ensure_syncode_import_path()
    from syncode.dfa_mask_store import DFAMaskStore
    import syncode.common as common

    mask_key = (grammar_text, _tokenizer_cache_fingerprint(tokenizer), dfa_mode)
    dfa_mask_store = _DFA_MASK_STORE_CACHE.get(mask_key)
    if dfa_mask_store is None:
        dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
            grammar=grammar,
            tokenizer=tokenizer,
            use_cache=True,
            logger=common.EmptyLogger(),
            mode=dfa_mode,
        )
        _DFA_MASK_STORE_CACHE[mask_key] = dfa_mask_store
    return dfa_mask_store


def create_lark_dafny_parser(
    grammar_source: str,
    VerifiedDecoderAgent,
    _dafny,
    start: str = "start",
    tokenizer=None,
    *,
    dfa_mode: str = "grammar_mask",
    apply_forbidden_token_filter: bool = True,
    constrained_span_opener: str | None = None,
    complete_start: str | None = None,
    accept_mask_backend: str = "syncode",
):
    """
    Create a Dafny-compatible parser using syncode's DFA mask store.

    Args:
        grammar_source: Either a grammar string or path to .lark file
        VerifiedDecoderAgent: The imported Dafny module
        _dafny: The Dafny runtime module
        start: Start rule name in the grammar
        tokenizer: HuggingFace tokenizer (required for DFA mask store)
        dfa_mode: SynCode DFAMaskStore mode (CRANE adaptive uses grammar_strict)
        apply_forbidden_token_filter: When False, do not hard-block { } ** tokens
        constrained_span_opener: If set (e.g. "<<"), prepend to constrained prefix
            text for mask/validity (CRANE structured_gen includes opening delimiter)
        complete_start: Lark start rule for IsCompletePrefix (default "start")
        accept_mask_backend: ``syncode`` (default) or ``llguidance`` (CARS SMILES).
            Validity/completeness still use Lark; only next-token accept masks switch.

    Returns:
        A SyncodeDafnyParser class that can be instantiated with a token list
    """
    _ensure_syncode_import_path()

    from lark.exceptions import UnexpectedCharacters, UnexpectedToken, UnexpectedEOF
    from syncode.parsers.incremental_parser import IncrementalParser

    grammar_text = _load_grammar_text(grammar_source)
    grammar, base_parser, lark_parser, complete_lark_parser = _get_parser_components(
        grammar_text, start, complete_start=complete_start
    )
    import os as _os
    _env_mode = _os.environ.get("CSD_DFA_MODE", "").strip()
    if _env_mode:
        dfa_mode = _env_mode
    _backend = (accept_mask_backend or "syncode").strip().lower()
    _env_backend = _os.environ.get("CSD_ACCEPT_MASK_BACKEND", "").strip().lower()
    if _env_backend:
        _backend = _env_backend
    dfa_mask_store = _get_cached_dfa_mask_store(
        grammar_text, grammar, tokenizer, dfa_mode=dfa_mode
    )
    llguidance_mask_store = None
    # SMILES/CARS: llguidance at empty prefix only (allows multi-atom BPE like ``CC``).
    # Non-empty prefixes keep Syncode — full-text/piece re-encode into llguidance can
    # shortfall-consume and return n=0, which trips cars.dfy stop
    # (IsCompletePrefix && ValidNextTokenCount==0) on incomplete rings.
    if _backend == "llguidance":
        if tokenizer is None:
            raise ValueError("accept_mask_backend=llguidance requires a tokenizer")
        from synthesis.evaluate.benchmarks.smiles.llguidance_mask import (
            SmilesLlguidanceMaskStore,
        )

        llguidance_mask_store = SmilesLlguidanceMaskStore(grammar_text, tokenizer)
    _span_opener = constrained_span_opener
    _use_forbidden_filter = apply_forbidden_token_filter
    # Ring-balance applies exactly when the grammar defines ring closures (SMILES).
    # Both metaDecode and GCD load the same grammar file, so this is framework-uniform.
    _enforce_ring_balance = "RING_CLOSURE" in grammar_text

    class SyncodeDafnyParser(VerifiedDecoderAgent.Parser):
        """Parser using syncode's DFA mask store for fast token validity checks."""

        def __init__(self, lm_tokens):
            super().__init__()
            self._lm_tokens = lm_tokens
            try:
                self._token_list = list(lm_tokens)
            except TypeError:
                self._token_list = [lm_tokens[i] for i in range(len(lm_tokens))]

            # Per-instance incremental parser (reset for each generation)
            self._inc_parser = IncrementalParser(base_parser, ignore_whitespace=False)
            self._dfa_mask_store = dfa_mask_store
            self._llguidance_mask_store = llguidance_mask_store
            self._UnexpectedCharacters = UnexpectedCharacters
            self._UnexpectedToken = UnexpectedToken
            self._UnexpectedEOF = UnexpectedEOF
            self._span_opener = _span_opener
            self._use_forbidden_filter = _use_forbidden_filter

            # Precompute token string -> index mapping
            self._token_str_to_idx = {}
            for idx, token in enumerate(self._token_list):
                token_str = dafny_seq_to_str(token)
                if token_str:
                    self._token_str_to_idx.setdefault(token_str, []).append(idx)

            # Hard-block mask: tokens whose string contains '{', '}', or '**'
            # are permanently forbidden regardless of the DFA over-approximation.
            # Syncode's grammar_mask mode over-approximates — e.g. whitespace-
            # prefixed tokens like ' {' slip through because '%ignore WS' makes
            # whitespace valid at any grammar state.  These characters are not
            # terminals in the GSM grammar, so blocking them is always safe.
            # The mask is True at positions that ARE allowed (i.e. not forbidden).
            import torch as _torch
            _forbidden_chars = ("{", "}", "**")
            _fb = _torch.ones(len(self._token_list), dtype=_torch.bool)
            for _idx, _token in enumerate(self._token_list):
                _ts = dafny_seq_to_str(_token)
                if any(_c in _ts for _c in _forbidden_chars):
                    _fb[_idx] = False
            self._forbidden_allow_mask: _torch.Tensor = _fb
            # Candidates for the closer rule (delimiter_hygiene.closer_allowed): only tokens with `>`.
            self._tokens_with_gt = [
                (_idx, dafny_seq_to_str(_token))
                for _idx, _token in enumerate(self._token_list)
                if ">" in dafny_seq_to_str(_token)
            ]

            # Lark parser for the rare IsValidPrefix fallback (keeps the CSD start
            # rule, e.g. csd_start, which requires the closing delimiter).
            self._lark = lark_parser
            # Separate parser for IsCompletePrefix completeness checks: its start rule
            # is the CONTENT rule "start" (any_expr / sql_stmt / smiles) WITHOUT the
            # grammar-masked closer, so span content can be recognized as complete and
            # the span can actually close.  See _get_parser_components for the why.
            self._complete_lark = complete_lark_parser
            self._valid_prefix_cache = {}
            self._complete_cache = {}
            self._valid_next_mask_cache = {}
            self._valid_next_indices_cache = {}
            # About five questions per decode step all need the same prefix
            # turned into text, and the conversion walks every token, so we
            # remember the text we already built for each prefix object
            # instead of redoing that walk. We key on id() because prefixes
            # aren't hashable, but a dead temporary's id can be handed to a
            # brand-new, unrelated object -- so each entry also holds the
            # prefix itself, and we only trust a hit when the stored object
            # is the exact same one being asked about now. Capped so a long
            # run doesn't just pile up one entry per decode step forever.
            self._prefix_text_cache = {}
            self._prefix_text_cache_limit = 1024

        def _tokens_to_text(self, tokens) -> str:
            """Convert Dafny token sequence to text."""
            key = id(tokens)
            cached = self._prefix_text_cache.get(key)
            if cached is not None and cached[0] is tokens:
                return cached[1]
            try:
                text = ''.join(dafny_seq_to_str(tokens[i]) for i in range(len(tokens)))
            except (TypeError, AttributeError, IndexError):
                text = str(tokens)
            if len(self._prefix_text_cache) >= self._prefix_text_cache_limit:
                # Dicts remember insertion order, so the first key we see is
                # the oldest one; drop it to make room for this new entry.
                self._prefix_text_cache.pop(next(iter(self._prefix_text_cache)))
            self._prefix_text_cache[key] = (tokens, text)
            return text

        def _structured_text(self, prefix) -> str:
            """Text for mask/validity; CRANE prepends opening << to expr-only prefix."""
            inner = self._tokens_to_text(prefix) if len(prefix) > 0 else ""
            if self._span_opener:
                return self._span_opener + inner
            return inner

        def _complete_text(self, prefix) -> str:
            """Text for IsCompletePrefix (expression body without delimiters)."""
            return self._tokens_to_text(prefix) if len(prefix) > 0 else ""

        def _is_valid_prefix(self, text: str) -> bool:
            """CRANE-aligned token validity (IterGen ``_is_valid``).

            Drive Syncode's IncrementalParser, then:
              - COMPLETE / MAYBE_COMPLETE remainder -> accept
              - otherwise accept iff ``dfa_mask_store.is_valid_prefix(r)``
              - parse exceptions -> reject

            Matches CRANE ``itergen`` opportunistic gating so illegal whole
            tokens (e.g. ``mx``, ``{``) are not accepted merely because the
            incremental parser returned without raising.
            """
            with _parser_timed("is_valid_prefix.total"):
                if not text:
                    return True
                if ">>" in text:
                    # The closer is written by the span helper, never part of the content.
                    return False
                cached = self._valid_prefix_cache.get(text)
                if cached is not None:
                    return cached
                try:
                    from syncode.parse_result import RemainderState

                    with _parser_timed("is_valid_prefix.inc_parser"):
                        r = self._inc_parser.get_acceptable_next_terminals(text)
                    if r.remainder_state in (
                        RemainderState.COMPLETE,
                        RemainderState.MAYBE_COMPLETE,
                    ):
                        result = True
                    elif self._dfa_mask_store is not None:
                        with _parser_timed("is_valid_prefix.dfa"):
                            result = bool(self._dfa_mask_store.is_valid_prefix(r))
                    else:
                        # No DFA store: cannot apply CRANE's second check; reject
                        # incomplete remainders rather than over-accepting.
                        result = False
                except (self._UnexpectedToken, self._UnexpectedCharacters, self._UnexpectedEOF):
                    result = False
                except Exception:
                    # CRANE ``_is_valid`` returns False on parse exceptions.
                    result = False
                self._valid_prefix_cache[text] = result
                return result

        def _is_complete(self, text: str) -> bool:
            """Check if text is a complete valid parse."""
            if not text:
                return False
            if _enforce_ring_balance:
                from synthesis.evaluate.benchmarks.common.delimiter_hygiene import (
                    ring_balance_allows_stop,
                )
                # An unclosed ring is grammar-complete but never a valid molecule;
                # forbid stopping until the ring closes. Grammar cannot express this.
                if not ring_balance_allows_stop(text):
                    return False
            cached = self._complete_cache.get(text)
            if cached is not None:
                return cached
            try:
                self._complete_lark.parse(text)
                result = True
            except Exception:
                result = False
            self._complete_cache[text] = result
            return result

        def _get_accept_mask_for_text(self, current_text: str):
            """Grammar accept mask, minus every token that would put `>>` into the span content."""
            from synthesis.evaluate.benchmarks.common.delimiter_hygiene import apply_closer_rule

            return apply_closer_rule(
                self._grammar_accept_mask_for_text(current_text), self._tokens_with_gt, current_text
            )

        def _grammar_accept_mask_for_text(self, current_text: str):
            """Get boolean accept mask using syncode's DFA mask store."""
            with _parser_timed("accept_mask.total"):
                cached = self._valid_next_mask_cache.get(current_text)
                if cached is not None:
                    _PARSER_TIMINGS["accept_mask.cache_hit"][1] += 1
                    return cached
                # llguidance only at empty prefix (CARS constrain_first surface).
                if self._llguidance_mask_store is not None and current_text == "":
                    with _parser_timed("accept_mask.llguidance"):
                        accept_mask = self._llguidance_mask_store.accept_mask_for_text(
                            current_text
                        )
                    if self._use_forbidden_filter:
                        if accept_mask.numel() == self._forbidden_allow_mask.numel():
                            accept_mask = accept_mask & self._forbidden_allow_mask
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask
                if self._dfa_mask_store is None:
                    # Fallback: brute force (slow but correct)
                    with _parser_timed("accept_mask.brute_force_fallback"):
                        import torch
                        accept_mask = torch.zeros(len(self._token_list), dtype=torch.bool)
                        for idx, token in enumerate(self._token_list):
                            token_str = dafny_seq_to_str(token)
                            if token_str and self._is_valid_prefix(current_text + token_str):
                                accept_mask[idx] = True
                    # Apply hard-block: remove forbidden-char tokens.
                    if self._use_forbidden_filter:
                        accept_mask = accept_mask & self._forbidden_allow_mask
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask

                try:
                    # Use syncode's incremental parser to get accept sequences
                    with _parser_timed("accept_mask.inc_parser_accepts"):
                        r = self._inc_parser.get_acceptable_next_terminals(current_text)
                    # Use DFA mask store to get accept mask, with sub-timers
                    # so we can see which piece of get_accept_mask dominates.
                    with _parser_timed("accept_mask.dfa_lookup"):
                        ms = self._dfa_mask_store
                        cur_incomplete_string = r.remainder
                        if cur_incomplete_string is None:
                            with _parser_timed("accept_mask.dfa_default_mask_ones"):
                                import torch as _t
                                accept_mask = _t.ones(len(ms._vocab), dtype=_t.bool)
                        else:
                            with _parser_timed("accept_mask.compute_dfa_states"):
                                cur_dfa_states = ms._dfas.compute_dfa_states(cur_incomplete_string)
                            with _parser_timed("accept_mask.lookup_next_tokens"):
                                accept_mask = ms._lookup_next_tokens(cur_dfa_states, r)
                            if ms.indentation and r.next_ac_indents is not None:
                                with _parser_timed("accept_mask.indent_intersect"):
                                    indent_ac_token = ms._lookup_table.get_indentation_tokens(r.next_ac_indents)
                                    accept_mask &= indent_ac_token
                        with _parser_timed("accept_mask.to_cpu"):
                            accept_mask = accept_mask.to(dtype=accept_mask.dtype, device='cpu')
                    # Apply hard-block: remove forbidden-char tokens that slipped
                    # through syncode's over-approximation.
                    if self._use_forbidden_filter:
                        accept_mask = accept_mask & self._forbidden_allow_mask
                    self._valid_next_mask_cache[current_text] = accept_mask
                    return accept_mask
                except Exception:
                    # Fallback on parse error
                    import torch
                    return torch.zeros(len(self._token_list), dtype=torch.bool)

        def _get_accept_mask_for_prefix(self, prefix):
            """Get boolean accept mask for a Dafny prefix sequence."""
            return self._get_accept_mask_for_text(self._structured_text(prefix))

        def _get_valid_token_indices(self, current_text: str):
            """Get list of valid token indices using a cached accept mask."""
            cached = self._valid_next_indices_cache.get(current_text)
            if cached is not None:
                return cached
            accept_mask = self._get_accept_mask_for_text(current_text)
            valid_indices = accept_mask.nonzero(as_tuple=False).flatten().tolist()
            self._valid_next_indices_cache[current_text] = valid_indices
            return valid_indices

        def is_valid_prefix(self, text: str) -> bool:
            return self._is_valid_prefix(text)

        def is_complete(self, text: str) -> bool:
            return self._is_complete(text)

        def IsValidPrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is valid."""
            with _parser_timed("IsValidPrefix.dafny"):
                if len(prefix) == 0 and not self._span_opener:
                    return True
                # CARS try_advance: reject prefixes llguidance cannot consume (stricter
                # than Lark alone). Uses per-piece ids to avoid BPE re-merge shortfall.
                if self._llguidance_mask_store is not None and len(prefix) > 0:
                    pieces = [dafny_seq_to_str(prefix[i]) for i in range(len(prefix))]
                    ids: list[int] = []
                    tok = self._llguidance_mask_store._tokenizer
                    for piece in pieces:
                        if piece:
                            ids.extend(tok.encode(piece, add_special_tokens=False))
                    store = self._llguidance_mask_store
                    store.reset()
                    if ids and store._matcher.try_consume_tokens(list(ids)) < len(ids):
                        return False
                return self._is_valid_prefix(self._structured_text(prefix))

        def IsCompletePrefix(self, prefix) -> bool:
            """Dafny interface: Check if prefix is complete."""
            with _parser_timed("IsCompletePrefix.dafny"):
                if len(prefix) == 0:
                    return False
                return self._is_complete(self._complete_text(prefix))

        def ValidNextTokens(self, prefix):
            """Dafny interface: Get valid next tokens using DFA mask store."""
            with _parser_timed("ValidNextTokens.total"):
                with _parser_timed("ValidNextTokens.tokens_to_text"):
                    current_text = self._structured_text(prefix)

                if current_text and not self._is_valid_prefix(current_text):
                    return _dafny.SeqWithoutIsStrInference([])

                with _parser_timed("ValidNextTokens.valid_indices"):
                    valid_indices = self._get_valid_token_indices(current_text)
                with _parser_timed("ValidNextTokens.materialize_dafny_seq"):
                    valid_tokens = [self._token_list[idx] for idx in valid_indices]
                    result = _dafny.SeqWithoutIsStrInference(valid_tokens)
                return result

        def ValidNextTokenCount(self, prefix):
            """Dafny interface: Count valid next tokens without materializing them."""
            with _parser_timed("ValidNextTokenCount.dafny"):
                current_text = self._structured_text(prefix)

                if current_text and not self._is_valid_prefix(current_text):
                    return 0

                accept_mask = self._get_accept_mask_for_text(current_text)
                return int(accept_mask.sum().item())

        def ValidNextToken(self, prefix, token):
            """Dafny interface: is this one token offered by the mask AND valid under the exact grammar check."""
            with _parser_timed("ValidNextToken.dafny"):
                current_text = self._structured_text(prefix)

                if current_text and not self._is_valid_prefix(current_text):
                    return False

                token_str = dafny_seq_to_str(token)
                if not token_str:
                    return False

                indices = self._token_str_to_idx.get(token_str)
                if not indices:
                    return False

                accept_mask = self._get_accept_mask_for_text(current_text)
                if not any(bool(accept_mask[idx]) for idx in indices if idx < len(accept_mask)):
                    return False
                # The mask is loose (it offers tokens the grammar rejects), so the
                # answer for one token comes from the exact check.
                return self._is_valid_prefix(current_text + token_str)

        def GroupHasValidMember(self, prefix, group):
            """Dafny interface: bulk DFA-mask check for group membership.

            Equivalent to (any t in group: ValidNextToken(prefix, t)) but does
            the DFA accept-mask lookup once and walks the group in one Python
            loop, instead of one DFA query per token in a Dafny-compiled loop.
            """
            with _parser_timed("GroupHasValidMember.dafny"):
                current_text = self._structured_text(prefix)
                if current_text and not self._is_valid_prefix(current_text):
                    return False
                accept_mask = self._get_accept_mask_for_text(current_text)
                accept_len = len(accept_mask)
                str_to_idx = self._token_str_to_idx
                for tok_dafny in group:
                    token_str = dafny_seq_to_str(tok_dafny)
                    if not token_str:
                        continue
                    indices = str_to_idx.get(token_str)
                    if not indices:
                        continue
                    if any(idx < accept_len and bool(accept_mask[idx]) for idx in indices) and (
                        self._is_valid_prefix(current_text + token_str)
                    ):
                        return True
                return False

        def CompletedSchemaSymbolCount(self, prefix):
            """Dafny interface: number of table_ref/column_ref symbols that have
            COMPLETED within `prefix`.

            Reads the IncrementalParser's SymbolPosMap side-record (IterGen's
            mechanism, ported into the vendored parser). Drives the incremental
            parser on this prefix's text first so the map reflects exactly this
            prefix (the parser caches by lexer-token prefix, so re-driving a grown
            or rolled-back prefix is cheap and restores the map to that point),
            then counts the completed schema symbols.

            PURE SIDE-RECORD: the SymbolPosMap is never read by the accept-set /
            masking path, so reading this count cannot change decode for any
            caller. Used as a mid-query unit boundary by the grounding helper:
            when the count rises, one more table/column name just finished.
            """
            with _parser_timed("CompletedSchemaSymbolCount.dafny"):
                return self.CompletedSymbolCount(prefix, "", 0)


        def _drive_symbol_map(self, prefix):
            current_text = self._structured_text(prefix)
            if not current_text:
                return None
            try:
                self._inc_parser.get_acceptable_next_terminals(current_text)
            except Exception:
                return None
            return getattr(self._inc_parser, "symbol_pos_map", None)

        def StructuredCharLength(self, prefix) -> int:
            return len(self._structured_text(prefix))

        def CompletedSymbolCount(self, prefix, unit: str, after_char: int = 0) -> int:
            """IterGen-style count of completed `unit` symbols ending after after_char.

            Empty unit "" sums table_ref+column_ref (SQL schema aggregate).
            """
            with _parser_timed("CompletedSymbolCount.dafny"):
                spm = self._drive_symbol_map(prefix)
                if spm is None:
                    return 0
                after = int(after_char)
                if not unit:
                    return int(
                        spm.get_symbol_count("table_ref", after=after)
                        + spm.get_symbol_count("column_ref", after=after)
                    )
                try:
                    return int(spm.get_symbol_count(unit, after=after))
                except Exception:
                    return 0

        def SymbolEndTokenIndex(self, prefix, unit: str, which: int) -> int:
            spm = self._drive_symbol_map(prefix)
            if spm is None:
                return len(prefix) if hasattr(prefix, "__len__") else 0
            try:
                char_pos = int(spm.get_symbol_pos_end(unit, int(which)))
            except Exception:
                return len(prefix)
            # Exclusive token index covering chars up to char_pos
            return int(self._char_pos_to_token_index_exclusive(prefix, char_pos))

        def _char_pos_to_token_index_exclusive(self, prefix, char_pos: int) -> int:
            """Smallest token index i such that chars of prefix[:i] cover char_pos in structured text."""
            structured = self._structured_text(prefix)
            opener_len = len(self._span_opener) if self._span_opener else 0
            # char_pos is in structured coordinates; map to inner prefix coords
            inner_pos = max(0, char_pos - opener_len)
            if inner_pos <= 0:
                return 0
            acc = 0
            for i in range(len(prefix)):
                piece = self._tokens_to_text([prefix[i]])
                acc += len(piece)
                if acc >= inner_pos:
                    return i + 1
            return len(prefix)

        def _char_pos_to_token_index(self, prefix, char_pos: int) -> int:
            """Map structured-text char_pos to inclusive token index in expr-only prefix."""
            opener_len = len(self._span_opener) if self._span_opener else 0
            inner_pos = max(0, int(char_pos) - opener_len)
            if inner_pos <= 0:
                return 0
            acc = 0
            for i in range(len(prefix)):
                piece = self._tokens_to_text([prefix[i]])
                next_acc = acc + len(piece)
                if next_acc > inner_pos:
                    return i
                acc = next_acc
            return len(prefix)

        def SymbolStartTokenIndex(self, prefix, unit: str, which: int) -> int:
            spm = self._drive_symbol_map(prefix)
            if spm is None:
                return 0
            try:
                char_pos = int(spm.get_symbol_pos_start(unit, int(which)))
            except Exception:
                return 0
            return int(self._char_pos_to_token_index(prefix, char_pos))

        def RenderSymbol(self, prefix, unit: str, which: int) -> str:
            spm = self._drive_symbol_map(prefix)
            if spm is None:
                return ""
            try:
                start = int(spm.get_symbol_pos_start(unit, int(which)))
                end = int(spm.get_symbol_pos_end(unit, int(which)))
            except Exception:
                return ""
            text = self._structured_text(prefix)
            if end > len(text):
                end = len(text)
            if start < 0:
                start = 0
            return text[start:end]


    return SyncodeDafnyParser


def get_builtin_grammar(format_name: str) -> str:
    """Get built-in grammar for common formats."""
    grammars = {
        "json": r'''
            start: value
            ?value: object | array | string | number | "true" -> true | "false" -> false | "null" -> null
            object: "{" [pair ("," pair)*] "}"
            pair: string ":" value
            array: "[" [value ("," value)*] "]"
            string: ESCAPED_STRING
            number: SIGNED_NUMBER
            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER
            %import common.WS
            %ignore WS
        ''',
        "sql": r'''
            start: select_stmt
            select_stmt: "SELECT"i columns "FROM"i table [where_clause]
            columns: "*" | column ("," column)*
            column: NAME
            table: NAME
            where_clause: "WHERE"i condition
            condition: NAME comp_op value
            comp_op: "=" | "!=" | "<" | ">" | "<=" | ">="
            value: NAME | NUMBER | STRING
            %import common.CNAME -> NAME
            %import common.NUMBER
            %import common.ESCAPED_STRING -> STRING
            %import common.WS
            %ignore WS
        ''',
        "math": r'''
            start: expr
            ?expr: term | expr "+" term | expr "-" term
            ?term: factor | term "*" factor | term "/" factor
            ?factor: NUMBER | "(" expr ")" | "-" factor
            %import common.NUMBER
            %import common.WS
            %ignore WS
        ''',
    }

    if format_name.lower() not in grammars:
        raise ValueError(f"Unknown format: {format_name}. Available: {list(grammars.keys())}")

    return grammars[format_name.lower()]


def _compute_unit_rollback_info(parser, tokens):
    """Find the last completed grammar unit in a token sequence.

    Inputs:
      parser   -- any object with an IsCompletePrefix(prefix: list) method
      tokens   -- a list of token objects (plain strings or Dafny sequences)

    Output:
      None if no complete unit has been generated.
      Otherwise a tuple (rollback_pos, unit_tokens) where:
        rollback_pos  -- index into tokens where the last unit began (i.e., the
                         position of the last complete point BEFORE this unit)
        unit_tokens   -- tokens[rollback_pos : last_complete_end]

    Algorithm:
      1. Walk the token list left-to-right, calling IsCompletePrefix on each
         growing prefix.
      2. Each time IsCompletePrefix transitions from True-at-i to False-at-i+1
         (or True-at-final), record the end position as a "complete boundary".
      3. The last two consecutive boundaries define the last unit.
         If only one boundary exists, the unit spans from index 0 to that boundary.
    """
    if not tokens:
        return None

    complete_ends = []
    for i in range(1, len(tokens) + 1):
        if parser.IsCompletePrefix(tokens[:i]):
            complete_ends.append(i)
        elif complete_ends and complete_ends[-1] == i - 1:
            # We just stepped past a complete point; the previous boundary stands.
            pass

    if not complete_ends:
        return None

    last_end = complete_ends[-1]
    # The unit start is right after the second-to-last complete boundary, or 0.
    if len(complete_ends) >= 2:
        rollback_pos = complete_ends[-2]
    else:
        rollback_pos = 0

    unit_tokens = tokens[rollback_pos:last_end]
    return rollback_pos, unit_tokens
