"""
PDDL Blocks World dataset: embedded planning problems.

Each problem is defined by:
  - Initial state (which blocks are on table, which are on other blocks)
  - Goal state
  - A reference plan (for documentation; correctness is checked by simulation)

Evaluation: simulate the LM-generated plan from the initial state and check
if the goal is achieved.

State representation:
  on_table: set of blocks resting directly on the table
  on:       dict block -> block (top block -> bottom block)
  clear:    set of blocks with nothing on top
  holding:  block currently held by the arm (str or None)

Problems inspired by the CARS paper (Gu et al., 2510.01902) PDDL planning
benchmark.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class BlocksState:
    on_table: Set[str] = field(default_factory=set)
    on: Dict[str, str] = field(default_factory=dict)   # top -> bottom
    clear: Set[str] = field(default_factory=set)
    holding: Optional[str] = None

    def copy(self) -> "BlocksState":
        return BlocksState(
            on_table=set(self.on_table),
            on=dict(self.on),
            clear=set(self.clear),
            holding=self.holding,
        )

    @property
    def hand_empty(self) -> bool:
        return self.holding is None


def _make_state(
    on_table: List[str],
    on: List[Tuple[str, str]],       # [(top, bottom), ...]
    clear: List[str],
) -> BlocksState:
    return BlocksState(
        on_table=set(on_table),
        on={t: b for t, b in on},
        clear=set(clear),
        holding=None,
    )


# ---------------------------------------------------------------------------
# Action simulator
# ---------------------------------------------------------------------------

class ActionError(Exception):
    """Raised when an action precondition is violated."""


def apply_action(state: BlocksState, action: str, *args: str) -> BlocksState:
    """
    Apply a Blocks World action to a state, returning the new state.
    Raises ActionError if preconditions are not met.
    """
    s = state.copy()
    action = action.lower()

    if action == "pick-up":
        (b,) = args
        if not s.hand_empty:
            raise ActionError(f"pick-up {b}: arm not empty (holding {s.holding})")
        if b not in s.clear:
            raise ActionError(f"pick-up {b}: {b} not clear")
        if b not in s.on_table:
            raise ActionError(f"pick-up {b}: {b} not on table")
        s.on_table.discard(b)
        s.clear.discard(b)
        s.holding = b

    elif action == "put-down":
        (b,) = args
        if s.holding != b:
            raise ActionError(f"put-down {b}: not holding {b}")
        s.on_table.add(b)
        s.clear.add(b)
        s.holding = None

    elif action == "stack":
        b1, b2 = args
        if s.holding != b1:
            raise ActionError(f"stack {b1} {b2}: not holding {b1}")
        if b2 not in s.clear:
            raise ActionError(f"stack {b1} {b2}: {b2} not clear")
        s.on[b1] = b2
        s.clear.discard(b2)
        s.clear.add(b1)
        s.holding = None

    elif action == "unstack":
        b1, b2 = args
        if not s.hand_empty:
            raise ActionError(f"unstack {b1} {b2}: arm not empty")
        if b1 not in s.clear:
            raise ActionError(f"unstack {b1} {b2}: {b1} not clear")
        if s.on.get(b1) != b2:
            raise ActionError(f"unstack {b1} {b2}: {b1} is not on {b2}")
        del s.on[b1]
        s.clear.discard(b1)
        s.clear.add(b2)
        s.holding = b1

    else:
        raise ActionError(f"Unknown action: {action}")

    return s


def goal_achieved(state: BlocksState, goal_on: List[Tuple[str, str]], goal_on_table: List[str]) -> bool:
    """Check if all goal fluents are satisfied in state."""
    for top, bottom in goal_on:
        if state.on.get(top) != bottom:
            return False
    for b in goal_on_table:
        if b not in state.on_table:
            return False
    return True


def simulate_plan(
    initial_state: BlocksState,
    plan: List[Tuple[str, ...]],    # [(action, arg1, ...), ...]
    goal_on: List[Tuple[str, str]],
    goal_on_table: List[str],
) -> Tuple[bool, str]:
    """
    Simulate a plan from the initial state.

    Returns:
        (success, reason_string)
        success = True if the goal is achieved after all actions succeed.
    """
    state = initial_state.copy()
    for i, step in enumerate(plan):
        action, *args = step
        try:
            state = apply_action(state, action, *args)
        except ActionError as e:
            return False, f"Step {i+1} ({action} {' '.join(args)}): {e}"
    if goal_achieved(state, goal_on, goal_on_table):
        return True, "Goal achieved"
    return False, f"Plan completed but goal not satisfied. on={state.on}, on_table={state.on_table}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class PDDLExample:
    """A single Blocks World planning problem."""
    problem_id: str
    description: str
    initial_state: BlocksState
    goal_on: List[Tuple[str, str]]       # (top_block, bottom_block) pairs
    goal_on_table: List[str]             # blocks that must be on table
    reference_plan: List[Tuple[str, ...]]

    @property
    def question(self) -> str:
        lines = [self.description, ""]
        lines.append("Initial state:")
        for b in sorted(self.initial_state.on_table):
            lines.append(f"  {b} is on the table")
        for top, bot in sorted(self.initial_state.on.items()):
            lines.append(f"  {top} is on {bot}")
        for b in sorted(self.initial_state.clear):
            lines.append(f"  {b} is clear")
        lines.append("")
        lines.append("Goal:")
        for top, bot in self.goal_on:
            lines.append(f"  {top} is on {bot}")
        for b in self.goal_on_table:
            lines.append(f"  {b} is on the table")
        lines.append("")
        lines.append(
            "Write a plan as a sequence of PDDL actions. "
            "Valid actions: (pick-up X), (put-down X), (stack X Y), (unstack X Y)."
        )
        return "\n".join(lines)

    @property
    def answer(self) -> str:
        return "CORRECT"  # Checked by simulation


_PROBLEMS: List[dict] = [
    # Problem 1: stack a on b (both on table, both clear)
    {
        "problem_id": "stack-a-on-b",
        "description": "Stack block a on top of block b.",
        "initial_state": _make_state(
            on_table=["a", "b"],
            on=[],
            clear=["a", "b"],
        ),
        "goal_on": [("a", "b")],
        "goal_on_table": ["b"],
        "reference_plan": [("pick-up", "a"), ("stack", "a", "b")],
    },
    # Problem 2: unstack a from b, put both on table
    {
        "problem_id": "unstack-a-from-b",
        "description": "Unstack block a from block b so both are on the table.",
        "initial_state": _make_state(
            on_table=["b"],
            on=[("a", "b")],
            clear=["a"],
        ),
        "goal_on": [],
        "goal_on_table": ["a", "b"],
        "reference_plan": [("unstack", "a", "b"), ("put-down", "a")],
    },
    # Problem 3: swap a and b (a on b -> b on a)
    {
        "problem_id": "swap-tower",
        "description": "Currently a is on b. Rearrange so b is on a.",
        "initial_state": _make_state(
            on_table=["b"],
            on=[("a", "b")],
            clear=["a"],
        ),
        "goal_on": [("b", "a")],
        "goal_on_table": ["a"],
        "reference_plan": [
            ("unstack", "a", "b"),
            ("put-down", "a"),
            ("pick-up", "b"),
            ("stack", "b", "a"),
        ],
    },
    # Problem 4: build tower a-b-c (a on b on c) from all on table
    {
        "problem_id": "build-tower-abc",
        "description": "Build a tower with a on b on c (a at top). All blocks start on the table.",
        "initial_state": _make_state(
            on_table=["a", "b", "c"],
            on=[],
            clear=["a", "b", "c"],
        ),
        "goal_on": [("a", "b"), ("b", "c")],
        "goal_on_table": ["c"],
        "reference_plan": [
            ("pick-up", "b"),
            ("stack", "b", "c"),
            ("pick-up", "a"),
            ("stack", "a", "b"),
        ],
    },
    # Problem 5: dismantle tower c-b-a (c on top) to all on table
    {
        "problem_id": "dismantle-tower",
        "description": "Dismantle the tower (c on b on a) so all blocks are on the table.",
        "initial_state": _make_state(
            on_table=["a"],
            on=[("b", "a"), ("c", "b")],
            clear=["c"],
        ),
        "goal_on": [],
        "goal_on_table": ["a", "b", "c"],
        "reference_plan": [
            ("unstack", "c", "b"),
            ("put-down", "c"),
            ("unstack", "b", "a"),
            ("put-down", "b"),
        ],
    },
    # Problem 6: move top of one tower to another
    {
        "problem_id": "move-top-block",
        "description": "Move the top block (b, on a) to be on top of c.",
        "initial_state": _make_state(
            on_table=["a", "c"],
            on=[("b", "a")],
            clear=["b", "c"],
        ),
        "goal_on": [("b", "c")],
        "goal_on_table": ["a", "c"],
        "reference_plan": [
            ("unstack", "b", "a"),
            ("stack", "b", "c"),
        ],
    },
    # Problem 7: build separate stacks (a on b, c on d) from all on table
    {
        "problem_id": "two-stacks",
        "description": "Build two separate stacks: a on b, and c on d. All blocks start on the table.",
        "initial_state": _make_state(
            on_table=["a", "b", "c", "d"],
            on=[],
            clear=["a", "b", "c", "d"],
        ),
        "goal_on": [("a", "b"), ("c", "d")],
        "goal_on_table": ["b", "d"],
        "reference_plan": [
            ("pick-up", "a"),
            ("stack", "a", "b"),
            ("pick-up", "c"),
            ("stack", "c", "d"),
        ],
    },
    # Problem 8: reorganize: a on b (table), c on d (table) -> c on b, a on d
    {
        "problem_id": "reorganize-stacks",
        "description": (
            "Currently a is on b and c is on d. "
            "Reorganize so that c is on b and a is on d."
        ),
        "initial_state": _make_state(
            on_table=["b", "d"],
            on=[("a", "b"), ("c", "d")],
            clear=["a", "c"],
        ),
        "goal_on": [("c", "b"), ("a", "d")],
        "goal_on_table": ["b", "d"],
        "reference_plan": [
            ("unstack", "a", "b"),
            ("put-down", "a"),
            ("unstack", "c", "d"),
            ("stack", "c", "b"),
            ("pick-up", "a"),
            ("stack", "a", "d"),
        ],
    },
]


def load_pddl(
    limit: Optional[int] = None,
    random_sample: bool = False,
    seed: int = 42,
) -> List[PDDLExample]:
    """
    Load embedded PDDL Blocks World problems.

    Args:
        limit:         Maximum number of problems to return.
        random_sample: If True, sample randomly (with seed) instead of taking first N.
        seed:          Random seed for sampling.

    Returns:
        List of PDDLExample instances.
    """
    examples = [PDDLExample(**p) for p in _PROBLEMS]

    if random_sample and limit is not None:
        rng = random.Random(seed)
        examples = rng.sample(examples, min(limit, len(examples)))
    elif limit is not None:
        examples = examples[:limit]

    return examples
