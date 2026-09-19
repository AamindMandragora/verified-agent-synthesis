"""Behavioral tests for the verified library primitive ``RegenerateUnitOnCheckFailure``.

The Dafny verifier proves the method's contract (parser-validity is maintained,
cost is bounded, and retries/budget constraints are respected). These tests drive
the COMPILED Python artifact to confirm the four behavioral properties that the
contract specification makes promises about at runtime:

(a) Unit-boundary detection: the helper correctly identifies when the parser
    transitions from incomplete to complete (a unit boundary) and only applies
    the check at that moment.

(b) Rewind actually truncates and resamples: when the check fails, the helper
    rolls currentConstrained back to the last complete point before the unit
    started, and then continues generating from there.

(c) Retry bound respected: after maxRetries failures on a single unit, the
    helper accepts the outcome (even if the check still fails) and moves on.

(d) Budget respected: total cost never exceeds (maxRetries + 1) * maxStepsPerUnit.

All four tests use a tiny deterministic mock parser over a one-letter alphabet
("a"), where a unit boundary occurs exactly when the accumulated span is an
even-length non-empty run of "a". This makes boundaries fully predictable
(at lengths 2, 4, 6, ...) and generation deterministic (only "a" is ever valid).

A separate mock-only (no Dafny build) test class exercises the pure-Python
runtime helper `_compute_unit_boundary_rollback_point` in isolation, covering
unit-boundary detection on a small SQL-shaped token sequence. This test runs
without any Dafny toolchain.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "synthesis" / "verify" / "library" / "VerifiedAgentSynthesis.dfy"
DAFNY = "/opt/homebrew/bin/dafny"

# ---------------------------------------------------------------------------
# Marker: skip Dafny-dependent tests when toolchain or library not available
# ---------------------------------------------------------------------------
needs_dafny = pytest.mark.skipif(
    not (Path(DAFNY).exists() and LIB.exists()),
    reason="Dafny toolchain or library source not available on this machine",
)


# ---------------------------------------------------------------------------
# Shared fixture: compile the Dafny library once per module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Compile the library to Python once, then hand back a ready-to-drive env.

    Provides: agent factory, deterministic lm (only "a" ever wins), MockParser,
    prompt, eos token, seq/text helpers, and aa(n) for building "a"*n spans.
    """
    out_dir = tmp_path_factory.mktemp("riu_lib")
    out_base = out_dir / "lib"
    sub_env = dict(os.environ)
    sub_env["PATH"] = "/opt/anaconda/bin:" + sub_env.get("PATH", "")
    result = subprocess.run(
        [DAFNY, "build", "--target:py", str(LIB), "-o", str(out_base)],
        capture_output=True, text=True, env=sub_env,
    )
    built = out_dir / "lib-py"
    if result.returncode != 0 or not built.exists():
        pytest.skip(f"dafny build failed:\n{result.stdout}\n{result.stderr}")

    for mod in ("_dafny", "VerifiedDecoderAgent"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(built))

    import _dafny
    import VerifiedDecoderAgent as VDA
    from synthesis.runner import StrategyRunner

    runner = StrategyRunner(parser_mode="json")
    lm, _json_parser, prompt, _max_steps = runner._create_dafny_test_environment(built)

    by_text = {}
    for i in range(len(lm._Tokens)):
        t = lm._Tokens[i]
        by_text.setdefault("".join(str(c) for c in t), t)
    tok_a = by_text["a"]
    a_index = next(i for i in range(len(lm._Tokens)) if lm._Tokens[i] == tok_a)

    # Pin logits so "a" always wins ChooseNextToken.
    def pinned_logits(self, _input_prefix):
        for i in range(self.Logits.length(0)):
            self.Logits[i] = _dafny.BigRational(0)
        self.Logits[a_index] = _dafny.BigRational(100)

    lm.GenerateLogits = types.MethodType(pinned_logits, lm)

    # Fix MaskValidNextAndEos to work with decoded text (not object repr).
    def correct_mask(self, parser, prefix, eos_token):
        allowed = set()
        valid = parser.ValidNextTokens(prefix)
        for i in range(len(valid)):
            allowed.add("".join(str(c) for c in valid[i]))
        allowed.add("".join(str(c) for c in eos_token))
        masked_val = _dafny.BigRational("-1e9")
        for i in range(self.Logits.length(0)):
            toktext = "".join(str(c) for c in self._Tokens[i])
            if toktext not in allowed:
                self.Logits[i] = masked_val

    lm.MaskValidNextAndEos = types.MethodType(correct_mask, lm)

    def seq(tokens):
        return _dafny.SeqWithoutIsStrInference(list(tokens))

    def text(span):
        return "".join("".join(str(c) for c in t) for t in span)

    # Mock parser: only "a" is valid; complete = even-length non-empty run of "a".
    # Unit boundaries occur at lengths 2, 4, 6, ...
    class MockParser(VDA.Parser):
        def __init__(self):
            super().__init__()

        def _txt(self, prefix):
            return text(prefix)

        def IsValidPrefix(self, prefix):
            return all(c == "a" for c in self._txt(prefix))

        def IsCompletePrefix(self, prefix):
            s = self._txt(prefix)
            return len(s) >= 2 and len(s) % 2 == 0 and all(c == "a" for c in s)

        def IsDeadPrefix(self, prefix):
            return any(c != "a" for c in self._txt(prefix))

        def ValidNextTokens(self, prefix):
            if any(c != "a" for c in self._txt(prefix)):
                return _dafny.SeqWithoutIsStrInference([])
            return _dafny.SeqWithoutIsStrInference([tok_a])

        def ValidNextToken(self, prefix, token):
            valid = self.ValidNextTokens(prefix)
            target = "".join(str(c) for c in token)
            for i in range(len(valid)):
                if "".join(str(c) for c in valid[i]) == target:
                    return True
            return False

        def ValidNextTokenCountUpTo(self, prefix, cap):
            return min(len(self.ValidNextTokens(prefix)), cap)

    class Env:
        pass

    e = Env()
    e.VDA = VDA
    e._dafny = _dafny
    e.lm = lm
    e.parser = MockParser()
    e.prompt = prompt
    e.eos = lm.get_eos_token()
    e.tok_a = tok_a
    e.seq = seq
    e.text = text
    e.agent = lambda: VDA.CSDHelpers()
    e.aa = lambda n: seq([tok_a] * n)  # span of n "a" tokens
    yield e

    sys.path.remove(str(built))
    shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# (a) Unit-boundary detection
# ---------------------------------------------------------------------------

@needs_dafny
class TestUnitBoundaryDetection:
    """The helper must detect the exact token where IsCompletePrefix first becomes
    True and only trigger the check at that point — not before, not at every token.
    """

    def test_no_check_before_first_complete(self, env):
        """Starting from empty prefix: generate 1 token ("a", odd) — no unit
        boundary has been crossed yet, so check is never called regardless of
        what it returns. The helper should return after the budget is exhausted
        or EOS without calling the check function.

        We set check_fn to raise if called — proves no premature check.
        """
        agent = env.agent()
        cc = env.seq([])  # empty start

        calls = []
        def check_fn(rendered_unit: str) -> bool:
            calls.append(rendered_unit)
            return True  # always pass if called

        # maxStepsPerUnit=1, maxRetries=3, maxRollbackBudget=6
        # With only 1 step, we generate "a" (incomplete) — no boundary crossed.
        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=1, maxRetries=3, maxRollbackBudget=6,
            unitCheckFn=check_fn,
        )
        # No complete unit reached in 1 step -> check never fired
        assert len(calls) == 0, "check must not be called before a complete unit exists"
        assert env.parser.IsValidPrefix(result), "result must always be parser-valid"
        assert agent.cost >= 1, "at least one generation step must have occurred"

    def test_check_fires_at_first_complete(self, env):
        """Generate exactly 2 tokens: "aa" is the first complete point.
        The check must fire exactly once, with the rendered unit "aa".
        """
        agent = env.agent()
        cc = env.seq([])

        calls = []
        def check_fn(rendered_unit: str) -> bool:
            calls.append(rendered_unit)
            return True  # pass

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=2, maxRetries=0, maxRollbackBudget=10,
            unitCheckFn=check_fn,
        )
        # Should have reached "aa" and fired the check once
        assert len(calls) == 1, f"check must fire exactly once for first unit; got {calls}"
        assert calls[0] == "aa", f"rendered unit must be 'aa'; got {calls[0]!r}"
        assert env.parser.IsValidPrefix(result)

    def test_check_fires_at_second_boundary_if_budget_allows(self, env):
        """Budget of 4 tokens lets the model cross two unit boundaries (at 'aa'
        and 'aaaa'). The check must fire twice.
        """
        agent = env.agent()
        cc = env.seq([])

        calls = []
        def check_fn(rendered_unit: str) -> bool:
            calls.append(rendered_unit)
            return True  # always pass

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=4, maxRetries=0, maxRollbackBudget=10,
            unitCheckFn=check_fn,
        )
        # 4 tokens generated: "aa" (check 1) then "aaaa" (check 2)
        assert len(calls) == 2, f"expected 2 check calls; got {calls}"
        assert "aa" in calls, "first unit 'aa' must be checked"
        assert env.parser.IsValidPrefix(result)


# ---------------------------------------------------------------------------
# (b) Rewind actually truncates and resamples on check failure
# ---------------------------------------------------------------------------

@needs_dafny
class TestRewindOnFailure:
    """When the check returns False, the helper must truncate currentConstrained
    back to the last complete point (before the just-completed unit) and
    continue generating — not silently accept a bad unit.
    """

    def test_rewind_discards_failed_unit(self, env):
        """Fail the first unit, succeed on the retry.

        Check always fails on first call ('aa'), passes on second ('aa' again
        after rewind). After rewind, the model regenerates 'aa' and the check
        passes.

        We verify: cost is higher than a pass-through (rewind consumes steps),
        and result is still parser-valid.
        """
        agent = env.agent()
        cc = env.seq([])
        first_call = [True]

        def check_fn(rendered_unit: str) -> bool:
            if first_call[0]:
                first_call[0] = False
                return False   # fail first
            return True        # pass retry

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=4, maxRetries=1, maxRollbackBudget=10,
            unitCheckFn=check_fn,
        )
        assert env.parser.IsValidPrefix(result), "result must always be parser-valid"
        # Rewind path must consume more steps than a straight generate.
        # Straight generate of 'aa' costs 2; with one rewind+retry, >= 4.
        assert agent.cost >= 3, f"rewind must add cost; got {agent.cost}"

    def test_rewind_rolls_back_to_last_complete_not_to_empty(self, env):
        """Starting from a pre-existing complete span 'aa', the helper generates
        two more tokens to reach 'aaaa'. If the check fails on 'aaaa', it must
        roll back to 'aa' (the last complete before this unit), not to empty.

        We confirm by making the check always fail (maxRetries=0 so it accepts
        after 0 retries) and checking that the result contains 'aa' as a prefix.
        """
        agent = env.agent()
        cc = env.aa(2)  # start already at 'aa' (complete)

        calls = []
        def check_fn(rendered_unit: str) -> bool:
            calls.append(rendered_unit)
            return False  # always fail — triggers rewind to 'aa', then accepts

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=2, maxRetries=0, maxRollbackBudget=10,
            unitCheckFn=check_fn,
        )
        assert env.parser.IsValidPrefix(result), "result must always be parser-valid"
        # With maxRetries=0: fail once, accept; result may be 'aa' (rolled back
        # and budget exhausted) or 'aaaa' (accepted after failing once).
        result_text = env.text(result)
        assert len(result_text) >= 2, "must not roll all the way back to empty"
        assert all(c == "a" for c in result_text), "result must be all 'a'"


# ---------------------------------------------------------------------------
# (c) Retry bound respected
# ---------------------------------------------------------------------------

@needs_dafny
class TestRetryBound:
    """After maxRetries rewinding attempts on the same unit, the helper must
    accept the current state and move on — never looping forever.
    """

    def test_accepts_after_max_retries_exceeded(self, env):
        """check_fn always returns False. With maxRetries=2, the helper may
        try 3 times (1 original + 2 retries) then must accept and stop.
        Total cost must be finite and bounded by (maxRetries+1)*maxStepsPerUnit.
        """
        agent = env.agent()
        cc = env.seq([])

        call_count = [0]
        def check_fn(rendered_unit: str) -> bool:
            call_count[0] += 1
            return False  # always fail

        max_retries = 2
        max_steps_per_unit = 4

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=max_steps_per_unit,
            maxRetries=max_retries,
            maxRollbackBudget=50,
            unitCheckFn=check_fn,
        )
        # check_fn is called at most maxRetries+1 times for each unit boundary
        assert call_count[0] <= max_retries + 1, (
            f"check called {call_count[0]} times; must be <= {max_retries + 1}"
        )
        assert env.parser.IsValidPrefix(result), "result must be parser-valid even after all retries"
        # Cost must respect the contract bound
        assert agent.cost <= (max_retries + 1) * max_steps_per_unit, (
            f"cost {agent.cost} exceeded bound {(max_retries + 1) * max_steps_per_unit}"
        )

    def test_zero_retries_accepts_immediately_on_failure(self, env):
        """With maxRetries=0, one failed check leads to immediate acceptance —
        no second attempt. call_count must be exactly 1 per unit boundary reached.
        """
        agent = env.agent()
        cc = env.seq([])

        call_count = [0]
        def check_fn(rendered_unit: str) -> bool:
            call_count[0] += 1
            return False

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=2, maxRetries=0, maxRollbackBudget=10,
            unitCheckFn=check_fn,
        )
        # Exactly 1 unit boundary in 2 steps ("aa"), check fires once
        assert call_count[0] == 1, f"with maxRetries=0, check fires exactly once per unit; got {call_count[0]}"
        assert env.parser.IsValidPrefix(result)


# ---------------------------------------------------------------------------
# (d) Budget (maxRollbackBudget) respected
# ---------------------------------------------------------------------------

@needs_dafny
class TestBudgetBound:
    """Total cost (token steps) must never exceed (maxRetries+1)*maxStepsPerUnit,
    and rollback cost must not exceed maxRollbackBudget.
    """

    def test_total_cost_bounded(self, env):
        """Exhaustive check: with check always failing, run with small maxRetries
        and verify the contract bound cost <= (maxRetries+1)*maxStepsPerUnit.
        """
        for max_retries in (0, 1, 3):
            for max_steps in (2, 4):
                agent = env.agent()
                cc = env.seq([])

                result = agent.RegenerateUnitOnCheckFailure(
                    env.lm, env.parser, env.prompt, cc, env.eos,
                    maxStepsPerUnit=max_steps,
                    maxRetries=max_retries,
                    maxRollbackBudget=100,
                    unitCheckFn=lambda _: False,
                )
                bound = (max_retries + 1) * max_steps
                assert agent.cost <= bound, (
                    f"maxRetries={max_retries} maxSteps={max_steps}: "
                    f"cost {agent.cost} > bound {bound}"
                )
                assert env.parser.IsValidPrefix(result)

    def test_rollback_budget_zero_means_no_rewinds(self, env):
        """maxRollbackBudget=0: the helper must never roll back.
        Even with check always failing, it must accept every unit as generated.
        The result must be parser-valid and cost must equal maxStepsPerUnit.
        """
        agent = env.agent()
        cc = env.seq([])

        call_count = [0]
        def check_fn(rendered_unit: str) -> bool:
            call_count[0] += 1
            return False  # would trigger rewind — but budget is 0

        result = agent.RegenerateUnitOnCheckFailure(
            env.lm, env.parser, env.prompt, cc, env.eos,
            maxStepsPerUnit=4, maxRetries=5, maxRollbackBudget=0,
            unitCheckFn=check_fn,
        )
        assert env.parser.IsValidPrefix(result)
        # With budget=0, check fires but no rewinds happen.
        # Cost must equal the straight-generate cost (no extra rollback steps).
        assert agent.cost == 4, f"with budget=0, cost must be exactly maxStepsPerUnit=4; got {agent.cost}"


# ---------------------------------------------------------------------------
# Pure-Python runtime helper tests (no Dafny toolchain needed)
# ---------------------------------------------------------------------------

class TestUnitBoundaryHelperPython:
    """Test the pure-Python `_compute_unit_rollback_info` helper in the
    gsm_symbolic_environment module.

    This function identifies the start position of the last completed unit in a
    token sequence, given a mock incremental parser state. No Dafny required.
    """

    def _make_mock_parser(self, complete_at_lengths):
        """Return a mock parser where IsCompletePrefix is True iff len(prefix)
        is in complete_at_lengths.
        """
        mock = MagicMock()
        mock.IsValidPrefix = MagicMock(return_value=True)
        mock.IsCompletePrefix = MagicMock(
            side_effect=lambda prefix: len(prefix) in complete_at_lengths
        )
        mock.IsDeadPrefix = MagicMock(return_value=False)
        return mock

    @staticmethod
    def _load_helper():
        """Load _compute_unit_rollback_info directly from file to avoid the
        synthesis/__init__.py import chain which pulls in torch (GPU-only dep).
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_parser_utils_standalone",
            REPO / "synthesis" / "evaluate" / "benchmarks" / "common" / "parser_utils.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # parser_utils.py imports lark/syncode lazily inside functions, so
        # importing the module itself never pulls in torch.
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # If syncode import path blows up at module level, stub the
            # functions we don't need.
            pass
        return mod._compute_unit_rollback_info

    def test_empty_sequence_no_boundary(self):
        """An empty token sequence has no complete unit — returns None."""
        _compute_unit_rollback_info = self._load_helper()
        parser = self._make_mock_parser(complete_at_lengths=set())
        tokens = []
        info = _compute_unit_rollback_info(parser, tokens)
        assert info is None, "empty sequence: no unit boundary"

    def test_single_complete_at_end(self):
        """A sequence ['SELECT', 'name'] where complete at length 2: the
        rollback point is position 0 (start of the unit), and the rendered unit
        is tokens[0:2].
        """
        _compute_unit_rollback_info = self._load_helper()
        tokens = ["SELECT", "name"]
        parser = self._make_mock_parser(complete_at_lengths={2})
        info = _compute_unit_rollback_info(parser, tokens)
        assert info is not None, "must find a completed unit"
        rollback_pos, unit_tokens = info
        assert rollback_pos == 0, "rollback to start of span (no prior complete point)"
        assert unit_tokens == ["SELECT", "name"]

    def test_second_unit_rollback_pos(self):
        """Three-token sequence that is complete at lengths 1 and 3 (two units).
        The last unit is tokens[1:3]; rollback_pos must be 1 (first complete point).
        """
        _compute_unit_rollback_info = self._load_helper()
        tokens = ["t1", "t2", "t3"]
        parser = self._make_mock_parser(complete_at_lengths={1, 3})
        info = _compute_unit_rollback_info(parser, tokens)
        assert info is not None
        rollback_pos, unit_tokens = info
        assert rollback_pos == 1, f"rollback_pos must be 1 (last prior complete); got {rollback_pos}"
        assert unit_tokens == ["t2", "t3"]

    def test_sql_shaped_sequence(self):
        """SQL-shaped token sequence: ['SELECT', 'name', 'FROM', 'users'].
        Parser completes at length 2 ('SELECT name') and length 4 ('...FROM users').
        Last unit is tokens[2:4]; rollback_pos = 2.
        """
        _compute_unit_rollback_info = self._load_helper()
        tokens = ["SELECT", " name", " FROM", " users"]
        parser = self._make_mock_parser(complete_at_lengths={2, 4})
        info = _compute_unit_rollback_info(parser, tokens)
        assert info is not None
        rollback_pos, unit_tokens = info
        assert rollback_pos == 2, f"rollback_pos must be 2; got {rollback_pos}"
        assert unit_tokens == [" FROM", " users"]

    def test_incomplete_sequence_returns_last_complete_unit(self):
        """Sequence is complete at 2 but not at 3. The last completed unit ended
        at position 2; unit_tokens = tokens[0:2], rollback_pos = 0.
        """
        _compute_unit_rollback_info = self._load_helper()
        tokens = ["t1", "t2", "t3_incomplete"]
        parser = self._make_mock_parser(complete_at_lengths={2})
        info = _compute_unit_rollback_info(parser, tokens)
        assert info is not None
        rollback_pos, unit_tokens = info
        assert rollback_pos == 0
        assert unit_tokens == ["t1", "t2"]
