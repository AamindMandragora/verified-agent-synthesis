"""
Tests for the IterGen-only reword of the GSM-Symbolic cot_instruct prompt.

Background (see synthesis/evaluate/baselines/crane_repo_runner.py and
CRANE_REPO_DIR/src/main.py): the itergen, crane, and unconstrained GSM lanes
all call the CRANE authors' main.py with --cot_grammar=gsm --out_grammar=gsm,
so main.py's instruct_type derivation (`instruct_type = args.cot_grammar`
when cot_grammar == out_grammar) resolves to the same yaml key, "gsm", for
every lane -- the only difference between lanes is --cot_grammar_mode
(original / adaptive / itergen), which on its own does not affect
instruct_type at all.

The authors' cot_instruct.gsm text (and the equivalent text / text-gsm
entries, which are byte-identical) contains a literal "<reasoning>"
placeholder:

    "Before answering you should reason about the problem (using the
    <reasoning> field in the response described below)."
    ...
    "Let's think step by step. <reasoning> The final answer is
    [[START]]symbolic expression[[END]]"

Under IterGen with Qwen3.5, the model copies the word "reasoning" into the
grammar-constrained answer span (see
~/modal-baseline-rerun-20260914/runs/baseline-trace-20260920h/lanes/gsm-itergen-q35-2b/stdout.log,
where 9/10 answers come back as the literal completion "<<reasoning>> ").

Fix: main.py's process_dataset_interactive() now special-cases
dataset=="gsm_symbolic", do_cot, and cot_grammar_mode=="itergen" to select a
new yaml key, "gsm-itergen", whose instruction text has the placeholder
reworded but is otherwise identical (same rules, same [[START]]/[[END]]
markers). No other lane is touched.
"""
import os
from pathlib import Path

import pytest
import yaml

CRANE_REPO_DIR = Path(os.environ.get("CRANE_REPO_DIR", os.path.expanduser("~/CRANE")))
CRANE_SRC_DIR = CRANE_REPO_DIR / "src"
YAML_PATH = CRANE_SRC_DIR / "prompt_templates" / "gsm_symbolic.yaml"
MAIN_PY_PATH = CRANE_SRC_DIR / "main.py"

pytestmark = pytest.mark.skipif(
    not YAML_PATH.exists() or not MAIN_PY_PATH.exists(),
    reason=f"CRANE copy not found at {CRANE_REPO_DIR} (set CRANE_REPO_DIR)",
)

# sha256 of the authors' ORIGINAL cot_instruct.gsm text, computed from the
# pristine CRANE copy before this change (text/gsm/text-gsm were verified
# byte-identical at that point). The crane and unconstrained lanes must keep
# producing exactly this text, byte for byte, forever.
ORIGINAL_GSM_COT_INSTRUCT_SHA256 = (
    "ff9b0416bf907acfcc3faa96df94e443faa30efb05afbc040d585735d1f7481f"
)


def _load_yaml():
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


def _resolve_instruct_type(cot_grammar_mode: str, do_cot: bool = True,
                            cot_grammar: str = "gsm", out_grammar: str = "gsm") -> str:
    """
    Re-implements the instruct_type selection call by reading the *current*
    source of main.py's process_dataset_interactive() (the function actually
    invoked by crane_repo_runner.py, which never passes --enable_dist) and
    executing just its instruct_type-selection statements against fake args.
    This keeps the test honest to the real file instead of hardcoding a copy
    of the selection logic that could silently drift out of sync, while
    avoiding the heavy GPU/model imports the rest of main.py pulls in.
    """
    src = MAIN_PY_PATH.read_text()
    lines = src.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if start is None and 'instruct_type = f"{args.cot_grammar}-{args.out_grammar}"' in line:
            # back up to the preceding "if args.cot_grammar != args.out_grammar:"
            start = i - 1
        if start is not None and end is None and "prompt_fn = BasePrompter(" in line:
            end = i
            break
    assert start is not None and end is not None, (
        "Could not locate the instruct_type selection block in main.py; "
        "the file structure changed more than this test expects."
    )
    import textwrap
    block = textwrap.dedent("\n".join(lines[start:end]))

    class Args:
        pass

    args = Args()
    args.cot_grammar = cot_grammar
    args.out_grammar = out_grammar
    args.dataset = "gsm_symbolic"
    args.do_cot = do_cot
    args.cot_grammar_mode = cot_grammar_mode

    namespace = {"args": args}
    exec(compile(block, "<main.py instruct_type block>", "exec"), namespace)
    return namespace["instruct_type"]


def test_itergen_lane_instruction_has_no_copyable_placeholder():
    instruct_type = _resolve_instruct_type(cot_grammar_mode="itergen")
    config = _load_yaml()
    text = config["cot_instruct"][instruct_type]

    assert "<reasoning>" not in text, (
        f"itergen instruct_type={instruct_type!r} still contains the literal "
        f"<reasoning> placeholder the model copies into the answer span"
    )
    # No other bare <word> placeholder either (only the [[START]]/[[END]]
    # markers and the authors' prose should remain).
    import re
    leftover_placeholders = re.findall(r"<[a-zA-Z_]+>", text)
    assert leftover_placeholders == [], (
        f"itergen instruction still has copyable placeholders: {leftover_placeholders}"
    )


def test_crane_and_unconstrained_lanes_are_byte_identical_to_authors_original():
    config = _load_yaml()

    crane_instruct_type = _resolve_instruct_type(cot_grammar_mode="adaptive")
    unconstrained_instruct_type = _resolve_instruct_type(cot_grammar_mode="original")

    # Both must resolve to the *same* authors' key ("gsm"), unaffected by our
    # itergen-only branch.
    assert crane_instruct_type == "gsm"
    assert unconstrained_instruct_type == "gsm"

    crane_text = config["cot_instruct"][crane_instruct_type]
    unconstrained_text = config["cot_instruct"][unconstrained_instruct_type]

    import hashlib

    assert hashlib.sha256(crane_text.encode()).hexdigest() == ORIGINAL_GSM_COT_INSTRUCT_SHA256
    assert hashlib.sha256(unconstrained_text.encode()).hexdigest() == ORIGINAL_GSM_COT_INSTRUCT_SHA256


def test_itergen_instruction_keeps_authors_rules_and_markers():
    instruct_type = _resolve_instruct_type(cot_grammar_mode="itergen")
    config = _load_yaml()
    text = config["cot_instruct"][instruct_type]

    assert "[[START]]" in text and "[[END]]" in text
    # The authors' allowed-operations rule sentence must survive verbatim.
    assert (
        "You are only allowed to use the following operations: "
        "+, -, /, //, %, (), and int()."
    ) in text
