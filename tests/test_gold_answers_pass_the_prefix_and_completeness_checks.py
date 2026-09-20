"""Property tests: the Dafny proof's prefix/completeness checks against real gold answers.

The Dafny proof in ``synthesis/verify/library/VerifiedAgentSynthesis.dfy`` trusts two
Python functions in ``synthesis/evaluate/benchmarks/common/parser_utils.py``:

- ``_is_valid_prefix(text)`` (Dafny-facing: ``IsValidPrefix``) -- "can this text still
  grow into a string the grammar accepts?"
- ``_is_complete(text)`` (Dafny-facing: ``IsCompletePrefix``) -- "is this text a whole
  grammar string?"

If either function disagrees with the grammar on real gold data, every proof built on
top of it is proving something about the wrong world. This file checks four properties
against the project's actual gold answers and actual ``.lark`` grammars, with no
tokenizer, model, or GPU: ``_is_valid_prefix``/``_is_complete`` are plain-Lark checks
that do not touch syncode's DFA mask store (that store is only used for the per-token
accept mask, which this file does not exercise), so the parser is built with
``tokenizer=None``.

(a) every character-prefix of a gold answer the grammar covers is a valid prefix.
(b) the full gold answer, for answers the grammar covers, is complete.
(c) monotone: once ``_is_valid_prefix`` is False for a prefix, it must stay False for
    every longer prefix of the same string (checked along gold answers and along
    deliberately damaged strings).
(d) damage (delete one character, insert an illegal character, append an unmatched
    closer, splice in "&gt;&gt;") must not un-reject a text: rejection is one-way, and any
    text containing "&gt;&gt;" is rejected outright (the span closer is never content).

Not every gold answer is inside the grammar's language -- e.g. 119 of the 1,034 Spider
dev gold queries end with a literal ";" in the raw gold file (EOQ punctuation, not part
of the "start" content rule this test checks) or hit an unrelated grammar gap. Coverage
is measured here, not hidden: properties (a)-(c) run only on the subset each grammar's
plain-Lark full parse actually accepts, and a regression test pins the reject count so
silent grammar drift shows up. (Double-quoted string literals -- 213 of the 1,034 golds --
were the largest cause of rejects until the 2026-09-20 grammar fix added double-quote
support to sql.lark's STRING terminal; see span-contract-fix-20260917.)

Run on focal (csd conda env has lark/regex/rdkit; no GPU, no tokenizer, no model load):

    cd ~/csd-generation-worktrees/gold-prefix-test
    /apps/conda/aadivyar/envs/csd/bin/python -m pytest -v -s \\
        tests/test_gold_answers_pass_the_prefix_and_completeness_checks.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lark import Lark  # noqa: E402

from synthesis.evaluate.benchmarks.common.parser_utils import (  # noqa: E402
    create_lark_dafny_parser,
)
from synthesis.project_defaults import (  # noqa: E402
    default_gsm_source_dir,
    default_spider_data_dir,
)

GRAMMAR_DIR = REPO_ROOT / "synthesis" / "evaluate" / "grammars"

# A character that is not a terminal or literal in any of the four grammars checked
# here (gsm.lark, sql.lark, smiles_acrylates.lark, smiles_chain_extenders.lark,
# smiles_isocyanates.lark), so inserting it is unambiguous damage rather than the
# start of some other legal token. Checked by ``grep -c '~' synthesis/evaluate/grammars/*.lark``.
ILLEGAL_CHAR = "~"

DAMAGE_SAMPLE_SIZE = 30  # how many covered gold answers get damage-tested per benchmark
DAMAGE_SEED = 20260920


# ---------------------------------------------------------------------------
# Parser construction: the same stub pattern tests/test_mask_brace_block.py uses
# for a real SyncodeDafnyParser without a model, tokenizer, or GPU.
# ---------------------------------------------------------------------------

class _FakeParserBase:
    pass


class _FakeVerifiedDecoderAgent:
    Parser = _FakeParserBase


class _FakeDafny:
    Seq = staticmethod(lambda s: s)
    SeqWithoutIsStrInference = staticmethod(lambda items: items)


def _make_parser(grammar_path: Path, **kwargs):
    """Build a real SyncodeDafnyParser for one grammar, no tokenizer/model/GPU.

    ``_is_valid_prefix``/``_is_complete`` are pure Lark (see parser_utils.py:
    ``_get_cached_dfa_mask_store`` returns None when ``tokenizer`` is None, and
    neither method touches the DFA mask store), so a real tokenizer is not needed
    to exercise them.
    """
    grammar_text = grammar_path.read_text()
    factory = create_lark_dafny_parser(
        grammar_text,
        _FakeVerifiedDecoderAgent(),
        _FakeDafny(),
        tokenizer=None,
        **kwargs,
    )
    return factory([])


def _plain_lark_full_parser(grammar_path: Path, start: str = "start") -> Lark:
    """Independent ground truth for "does the grammar cover this gold string at
    all", built without going through parser_utils, so property (b) below is not
    circular with the thing it is checking.
    """
    return Lark(grammar_path.read_text(), start=start, parser="lalr", lexer="basic")


def _split_by_grammar_coverage(lark_parser: Lark, texts: list[str]) -> tuple[list[str], list[str]]:
    covered, rejected = [], []
    for text in texts:
        try:
            lark_parser.parse(text)
            covered.append(text)
        except Exception:
            rejected.append(text)
    return covered, rejected


# ---------------------------------------------------------------------------
# Shared properties (a)-(d), applied to every benchmark below.
# ---------------------------------------------------------------------------

def _validity_profile(parser, text: str) -> list[bool]:
    """profile[n] = _is_valid_prefix(text[:n]) for n in 0..len(text)."""
    return [parser._is_valid_prefix(text[:n]) for n in range(len(text) + 1)]


def _first_true_after_false(profile: list[bool]) -> int | None:
    """Index of a prefix accepted after a shorter one was rejected, or None."""
    seen_false = False
    for i, valid in enumerate(profile):
        if not valid:
            seen_false = True
        elif seen_false:
            return i
    return None


def _assert_every_prefix_is_valid(parser, texts: list[str], label: str) -> None:
    """Property (a): every character-prefix of a grammar-covered gold answer is valid."""
    failures = []
    for text in texts:
        profile = _validity_profile(parser, text)
        if not all(profile[1:]):
            first_bad = profile.index(False, 1)
            failures.append(text[:first_bad])
        if len(failures) >= 5:
            break
    assert not failures, (
        f"{label}: {len(failures)}+ gold-covered prefixes rejected as invalid, e.g. "
        f"{failures[:5]!r}"
    )


def _assert_complete_answers_are_recognized(parser, texts: list[str], label: str) -> None:
    """Property (b): the full gold answer, for grammar-covered answers, is complete."""
    failures = [text for text in texts if not parser._is_complete(text)][:5]
    assert not failures, f"{label}: gold answers the grammar covers were not IsCompletePrefix: {failures!r}"


def _assert_monotone(parser, texts: list[str], label: str) -> None:
    """Property (c): once invalid, a prefix's longer extensions never become valid again."""
    failures = []
    for text in texts:
        profile = _validity_profile(parser, text)
        bad_index = _first_true_after_false(profile)
        if bad_index is not None:
            failures.append((text[:bad_index], text))
        if len(failures) >= 5:
            break
    assert not failures, (
        f"{label}: {len(failures)}+ strings had a shorter prefix rejected but a "
        f"longer one accepted (non-monotone), e.g. {failures[:5]!r}"
    )


def _damaged_variants(text: str, rng: random.Random) -> list[tuple[str, str]]:
    """(description, damaged_text) pairs: one character deleted, one illegal character
    inserted, an unmatched closing paren inserted, and the closer '>>' appended."""
    variants = []
    if len(text) > 1:
        i = rng.randrange(len(text))
        variants.append(("delete_char", text[:i] + text[i + 1:]))
    i = rng.randrange(len(text) + 1)
    variants.append((f"insert_{ILLEGAL_CHAR!r}", text[:i] + ILLEGAL_CHAR + text[i:]))
    i = rng.randrange(len(text) + 1)
    variants.append(("insert_unmatched_)", text[:i] + ")" + text[i:]))
    variants.append(("append_>>", text + ">>"))
    i = len(text) // 2
    variants.append(("splice_>>_in_middle", text[:i] + ">>" + text[i:]))
    return variants


def _assert_damage_is_one_way_and_closer_always_rejected(
    parser, texts: list[str], label: str
) -> None:
    """Property (d): once a damaged prefix is rejected, no longer prefix of the same
    damaged string is accepted (monotonicity again, now on strings that really can be
    rejected), and any text containing '>>' is rejected outright."""
    rng = random.Random(DAMAGE_SEED)
    sample = texts if len(texts) <= DAMAGE_SAMPLE_SIZE else rng.sample(texts, DAMAGE_SAMPLE_SIZE)

    monotone_failures = []
    closer_failures = []
    for text in sample:
        for description, damaged in _damaged_variants(text, rng):
            if ">>" in damaged:
                if parser._is_valid_prefix(damaged):
                    closer_failures.append((description, damaged))
                    continue
            profile = _validity_profile(parser, damaged)
            bad_index = _first_true_after_false(profile)
            if bad_index is not None:
                monotone_failures.append((description, damaged[:bad_index], damaged))
        if len(monotone_failures) >= 5 and len(closer_failures) >= 5:
            break

    assert not closer_failures[:5], (
        f"{label}: text containing '>>' was accepted as a valid prefix, e.g. "
        f"{closer_failures[:5]!r}"
    )
    assert not monotone_failures[:5], (
        f"{label}: a damaged string had a shorter prefix rejected but a longer one "
        f"accepted, e.g. {monotone_failures[:5]!r}"
    )


# ---------------------------------------------------------------------------
# GSM-Symbolic: gold answer expressions are the ``answer_parsed`` field of each
# CRANE gsm_symbolic JSON file (this is exactly what eval_logic.expected_answer
# returns and what the CSD grammar is meant to accept -- see
# synthesis/evaluate/benchmarks/gsm_symbolic/eval_logic.py:118).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gsm_gold_answers() -> list[str]:
    gsm_dir = default_gsm_source_dir()
    if not gsm_dir.exists():
        pytest.skip(f"GSM gold data not found at {gsm_dir}")
    answers = []
    for path in sorted(gsm_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        answer_parsed = payload.get("answer_parsed") or ""
        if answer_parsed:
            answers.append(answer_parsed)
    if not answers:
        pytest.skip(f"No answer_parsed gold answers found under {gsm_dir}")
    return answers


@pytest.fixture(scope="module")
def gsm_parser():
    return _make_parser(GRAMMAR_DIR / "gsm.lark")


@pytest.fixture(scope="module")
def gsm_plain_lark() -> Lark:
    return _plain_lark_full_parser(GRAMMAR_DIR / "gsm.lark")


@pytest.fixture(scope="module")
def gsm_gold_split(gsm_plain_lark, gsm_gold_answers):
    return _split_by_grammar_coverage(gsm_plain_lark, gsm_gold_answers)


# Measured 2026-09-20 on the 100 CRANE gsm_symbolic gold files: 3 answers are
# outside gsm.lark's language (two have an unbalanced paren in the source data,
# one uses "**" exponentiation, which the grammar does not define an operator
# for). A regression test below fails if this count grows.
GSM_MAX_KNOWN_REJECTS = 3


def test_gsm_gold_coverage_is_reported_and_does_not_regress(gsm_gold_answers, gsm_gold_split):
    covered, rejected = gsm_gold_split
    print(
        f"GSM: {len(gsm_gold_answers)} gold answers, {len(covered)} covered by "
        f"gsm.lark, {len(rejected)} rejected: {rejected}"
    )
    assert len(rejected) <= GSM_MAX_KNOWN_REJECTS, (
        f"gsm.lark now rejects {len(rejected)} gold answers (was <= {GSM_MAX_KNOWN_REJECTS}): "
        f"{rejected}"
    )


def test_gsm_every_prefix_of_a_covered_gold_answer_is_valid(gsm_parser, gsm_gold_split):
    covered, _ = gsm_gold_split
    _assert_every_prefix_is_valid(gsm_parser, covered, "GSM")


def test_gsm_covered_gold_answers_are_complete(gsm_parser, gsm_gold_split):
    covered, _ = gsm_gold_split
    _assert_complete_answers_are_recognized(gsm_parser, covered, "GSM")


def test_gsm_validity_is_monotone_along_gold_answers(gsm_parser, gsm_gold_split):
    covered, _ = gsm_gold_split
    _assert_monotone(gsm_parser, covered, "GSM")


def test_gsm_damage_is_one_way_and_closer_always_rejected(gsm_parser, gsm_gold_split):
    covered, _ = gsm_gold_split
    _assert_damage_is_one_way_and_closer_always_rejected(gsm_parser, covered, "GSM")


# ---------------------------------------------------------------------------
# Spider: gold SQL is dev_gold.sql, loaded the same way
# synthesis/evaluate/benchmarks/sql_spider/dataset.py loads it (tab-split,
# ``.strip()`` on the SQL half, one gold query per non-blank line).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spider_gold_queries() -> list[str]:
    spider_dir = default_spider_data_dir()
    gold_path = spider_dir / "dev_gold.sql"
    if not gold_path.exists():
        pytest.skip(f"Spider gold data not found at {gold_path}")
    queries = []
    for line in gold_path.read_text().splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        gold_sql, _, _gold_db_id = line.partition("\t")
        gold_sql = gold_sql.strip()
        if gold_sql:
            queries.append(gold_sql)
    return queries


@pytest.fixture(scope="module")
def spider_parser():
    return _make_parser(GRAMMAR_DIR / "sql.lark", dfa_mode="grammar_mask")


@pytest.fixture(scope="module")
def spider_plain_lark() -> Lark:
    return _plain_lark_full_parser(GRAMMAR_DIR / "sql.lark")


@pytest.fixture(scope="module")
def spider_gold_split(spider_plain_lark, spider_gold_queries):
    return _split_by_grammar_coverage(spider_plain_lark, spider_gold_queries)


# Measured 2026-09-20 against the official 1,034-row Spider dev_gold.sql, after the
# same-day fix adding double-quoted string literal support to sql.lark's STRING
# terminal (span-contract-fix-20260917): 119 gold queries are outside sql.lark's
# language (was 308 before that fix; double-quoted literals alone accounted for 213
# of the old rejects). Of the remaining 119, most end with a literal ";" in the raw
# gold file, which is EOQ punctuation (csd_start), not part of the "start" content
# rule this test checks. A regression test below fails if the reject count grows.
SPIDER_MAX_KNOWN_REJECTS = 119


def test_spider_gold_coverage_is_reported_and_does_not_regress(spider_gold_queries, spider_gold_split):
    covered, rejected = spider_gold_split
    double_quoted = sum(1 for q in rejected if '"' in q)
    print(
        f"Spider: {len(spider_gold_queries)} gold queries, {len(covered)} covered by "
        f"sql.lark, {len(rejected)} rejected ({double_quoted} of those contain a "
        f'double-quoted string)'
    )
    assert len(rejected) <= SPIDER_MAX_KNOWN_REJECTS, (
        f"sql.lark now rejects {len(rejected)} gold queries (was <= {SPIDER_MAX_KNOWN_REJECTS})"
    )


def test_spider_every_prefix_of_a_covered_gold_query_is_valid(spider_parser, spider_gold_split):
    covered, _ = spider_gold_split
    # Full 726-query covered set measured at ~7s locally, well inside the ~3
    # minute budget, so no sampling is needed for (a).
    _assert_every_prefix_is_valid(spider_parser, covered, "Spider")


def test_spider_covered_gold_queries_are_complete(spider_parser, spider_gold_split):
    covered, _ = spider_gold_split
    _assert_complete_answers_are_recognized(spider_parser, covered, "Spider")


def test_spider_validity_is_monotone_along_gold_queries(spider_parser, spider_gold_split):
    covered, _ = spider_gold_split
    _assert_monotone(spider_parser, covered, "Spider")


def test_spider_damage_is_one_way_and_closer_always_rejected(spider_parser, spider_gold_split):
    covered, _ = spider_gold_split
    _assert_damage_is_one_way_and_closer_always_rejected(spider_parser, covered, "Spider")


# ---------------------------------------------------------------------------
# SMILES: gold exemplar molecules are the "Molecule: ..." lines already baked
# into each class's prompt file (synthesis.evaluate.benchmarks.smiles.dataset).
# ---------------------------------------------------------------------------

from synthesis.evaluate.benchmarks.smiles.dataset import (  # noqa: E402
    SMILES_CLASSES,
    get_smiles_task,
)

# Measured 2026-09-20: acrylates 1/32, chain_extenders 1/11, isocyanates 1/11
# gold molecules are outside their grammar's language (each is a fused/bridged
# polycyclic ring arrangement the grammar's RING_CLOSURE/branch shape does not
# admit). A regression test below fails if any class's reject count grows.
SMILES_MAX_KNOWN_REJECTS = {"acrylates": 1, "chain_extenders": 1, "isocyanates": 1}


@pytest.fixture(scope="module", params=list(SMILES_CLASSES))
def smiles_class(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def smiles_gold_molecules(smiles_class) -> list[str]:
    task = get_smiles_task(smiles_class)
    molecules = task["prompt_exemplars"]
    if not molecules:
        pytest.skip(f"No exemplar molecules found for SMILES class {smiles_class}")
    return molecules


@pytest.fixture(scope="module")
def smiles_grammar_path(smiles_class) -> Path:
    return GRAMMAR_DIR / f"smiles_{smiles_class}.lark"


@pytest.fixture(scope="module")
def smiles_parser(smiles_grammar_path):
    return _make_parser(smiles_grammar_path)


@pytest.fixture(scope="module")
def smiles_plain_lark(smiles_grammar_path) -> Lark:
    return _plain_lark_full_parser(smiles_grammar_path)


@pytest.fixture(scope="module")
def smiles_gold_split(smiles_plain_lark, smiles_gold_molecules):
    return _split_by_grammar_coverage(smiles_plain_lark, smiles_gold_molecules)


def test_smiles_gold_coverage_is_reported_and_does_not_regress(
    smiles_class, smiles_gold_molecules, smiles_gold_split
):
    covered, rejected = smiles_gold_split
    print(
        f"SMILES/{smiles_class}: {len(smiles_gold_molecules)} gold molecules, "
        f"{len(covered)} covered by smiles_{smiles_class}.lark, {len(rejected)} "
        f"rejected: {rejected}"
    )
    max_known = SMILES_MAX_KNOWN_REJECTS[smiles_class]
    assert len(rejected) <= max_known, (
        f"smiles_{smiles_class}.lark now rejects {len(rejected)} gold molecules "
        f"(was <= {max_known}): {rejected}"
    )


def test_smiles_every_prefix_of_a_covered_gold_molecule_is_valid(smiles_class, smiles_parser, smiles_gold_split):
    covered, _ = smiles_gold_split
    _assert_every_prefix_is_valid(smiles_parser, covered, f"SMILES/{smiles_class}")


def test_smiles_covered_gold_molecules_are_complete(smiles_class, smiles_parser, smiles_gold_split):
    covered, _ = smiles_gold_split
    _assert_complete_answers_are_recognized(smiles_parser, covered, f"SMILES/{smiles_class}")


def test_smiles_validity_is_monotone_along_gold_molecules(smiles_class, smiles_parser, smiles_gold_split):
    covered, _ = smiles_gold_split
    _assert_monotone(smiles_parser, covered, f"SMILES/{smiles_class}")


def test_smiles_damage_is_one_way_and_closer_always_rejected(smiles_class, smiles_parser, smiles_gold_split):
    covered, _ = smiles_gold_split
    _assert_damage_is_one_way_and_closer_always_rejected(smiles_parser, covered, f"SMILES/{smiles_class}")
