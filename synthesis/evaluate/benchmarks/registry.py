"""Benchmark logic registry for evaluation delegation."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def get_logic(dataset_name: str) -> Any:
    if dataset_name == "gsm_symbolic":
        return import_module("synthesis.evaluate.benchmarks.gsm_symbolic.eval_logic")
    if dataset_name == "spider":
        return import_module("synthesis.evaluate.benchmarks.sql_spider.eval_logic")
    if dataset_name == "smiles":
        return import_module("synthesis.evaluate.benchmarks.smiles.eval_logic")
    raise ValueError(f"Unknown dataset: {dataset_name}")


def resolve_require_delimiters(dataset_name: str, cli_value: bool) -> bool:
    """Decide whether the eval loop should require a visible << >> span.

    Every benchmark's output now carries its spans as visible `<< >>` -- when
    the strategy does not open the span, the runtime does, with a literal
    `<<`. So there is nothing for the benchmark to veto and the CLI flag
    (`cli_value`) decides. `dataset_name` is still validated, so a typo'd
    dataset fails here rather than silently evaluating nothing.
    """
    get_logic(dataset_name)
    return cli_value
