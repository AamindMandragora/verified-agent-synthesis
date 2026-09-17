import ast
from pathlib import Path
from types import SimpleNamespace

import json
import hashlib
import subprocess
import threading
import sys
import pytest

from scripts.runtime import run_cold_synthesis_queue as queue


EXPECTED_CRANE_GSM_TASK = (
    "You are an expert in solving grade school math tasks. "
    "You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.\n\n"
    "Before answering you should reason about the problem (using the <reasoning> field in the response described below). "
    "Intermediate symbolic expressions generated during reasoning should be wrapped in << >>.\n\n"
    "Then, output the symbolic expression wrapped in << >> that answers the question. "
    "The expressions must use numbers as well as the variables defined in the question. "
    "You are only allowed to use the following operations: +, -, /, //, %, (), and int().\n\n"
    "You will always respond in the format described below: \n"
    "Let's think step by step. <reasoning> The final answer is <<symbolic expression>>"
)


def _job(dataset="gsm_symbolic"):
    return {
        "cell_id": "gsm-qwen35-2b",
        "task": "Solve symbolic math.",
        "dataset": dataset,
        "eval_model": "Qwen/Qwen3.5-2B",
        "max_iterations": 80,
        "eval_sample_size": 49,
        "min_accuracy": 0.265306122449,
        "min_syntax_rate": 0.9,
        "eval_max_steps": 900,
        "eval_max_seconds": 600,
        "memory_reservation_mib": 16384,
        "gpu_mem_util": 0.8,
        "output_name": "coldq_gsm-qwen35-2b_0719",
        "heldout_sample_size": 49,
        "heldout_split_file": "/repo/split.json",
        "heldout_split_name": "test",
        "heldout_output_json": "/repo/heldout.json",
    }


def _matching_run_configuration(job):
    prefix = "gsm" if job["dataset"] == "gsm_symbolic" else "spider"
    split = {
        "gsm_split_file": None,
        "gsm_split_name": None,
        "spider_split_file": None,
        "spider_split_name": None,
        "bar_split_name": "train",
    }
    if job["dataset"] in {"gsm_symbolic", "spider"}:
        split[f"{prefix}_split_file"] = str(
            Path("/repo")
            / (
                "gsm_symbolic_crane_proportional_49x49_seed123.json"
                if prefix == "gsm"
                else "spider_dev_proportional_300x300_seed334.json"
            )
        )
        split[f"{prefix}_split_name"] = "train"
    pinned_commit = job.get("launch_commit") or job.get("git_commit") or ("c" * 40)
    return {
        "task_description": job["task"],
        "output_name": job["output_name"],
        "git_commit": pinned_commit,
        "max_iterations": job["max_iterations"],
        "thresholds": {
            "min_accuracy": job["min_accuracy"],
            "min_syntax_rate": job["min_syntax_rate"],
        },
        "author_model": {
            "backend": "claude",
            "model": queue.AUTHOR_MODEL,
            "max_new_tokens": 8192,
            "reasoning_budget_tokens": 4096,
            "anthropic_thinking": "always-on",
            "anthropic_effort": "high",
            "anthropic_thinking_display": "summarized",
        },
        "evaluation": {
            "dataset": job["dataset"],
            "eval_model": job["eval_model"],
            "eval_sample_size": job["eval_sample_size"],
            "eval_max_steps": job["eval_max_steps"],
            "eval_step_token_budget": 1,
            "eval_max_seconds_per_example": job["eval_max_seconds"],
            "smiles_classes": (
                [job["smiles_class"]] if job.get("smiles_class") else None
            ),
            "split_provenance": split,
        },
    }


def test_gsm_task_matches_crane_task_and_cot_contract() -> None:
    assert queue.GSM_TASK == EXPECTED_CRANE_GSM_TASK
    assert {
        config["task"]
        for config in queue.EXPECTED_CELLS.values()
        if config["dataset"] == "gsm_symbolic"
    } == {EXPECTED_CRANE_GSM_TASK}


def test_synthesis_command_is_cold_and_preserves_the_run_limits():
    command = queue.synthesis_command(_job(), Path("/env/python"))

    assert command[:3] == ["/env/python", "-m", "synthesis.run_synthesis"]
    assert command[command.index("--synthesis-max-tokens") + 1] == "8192"
    assert command[command.index("--max-iterations") + 1] == "80"
    assert "--bar-split-name" not in command
    assert not any(flag.startswith("--initial-") for flag in command)


def test_synthesis_command_uses_the_fixed_claude_code_opus5_author():
    command = queue.synthesis_command(_job(), Path("/env/python"))

    assert command[command.index("--generation-backend") + 1] == "claude"
    assert command[command.index("--generation-model") + 1] == "claude-opus-5"


def test_bedrock_author_enables_extended_thinking(monkeypatch):
    from synthesis.generate.generator import StrategyGenerator

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    author = StrategyGenerator(
        backend="claude-bedrock",
        model_name="us.anthropic.claude-sonnet-4-6",
        max_new_tokens=8192,
        reasoning_budget_tokens=4096,
    )

    assert author.anthropic_thinking == "always-on"
    assert author.anthropic_effort == "high"
    assert author.anthropic_thinking_display == "summarized"
    assert author._bedrock_thinking_fields() == {
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }


def test_recovery_command_uses_only_the_remaining_original_calls():
    job = _job()
    job.update(
        {
            "max_iterations": 36,
            "initial_attempt_offset": 4,
            "initial_attempt_history_file": "saved-results/recovery/gsm35-2b.json",
        }
    )

    command = queue.synthesis_command(job, Path("/env/python"))

    assert command[command.index("--max-iterations") + 1] == "36"
    assert command[command.index("--initial-attempt-offset") + 1] == "4"
    assert command[command.index("--initial-attempt-history-file") + 1] == (
        "saved-results/recovery/gsm35-2b.json"
    )


def test_synthesis_command_only_uses_run_synthesis_cli_flags():
    command = queue.synthesis_command(_job(), Path(sys.executable))
    help_result = subprocess.run(
        [sys.executable, "-m", "synthesis.run_synthesis", "--help"],
        cwd=Path(__file__).parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    unsupported = [
        argument
        for argument in command
        if argument.startswith("--") and argument not in help_result.stdout
    ]

    assert unsupported == []


def test_run_synthesis_does_not_read_removed_cli_arguments():
    source_path = Path("synthesis/run_synthesis.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    parser_destinations = set()
    namespace_reads = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            explicit_dest = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "dest"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            option_strings = [
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            ]
            long_option = next(
                (option for option in option_strings if option.startswith("--")),
                option_strings[0] if option_strings else None,
            )
            if explicit_dest:
                parser_destinations.add(explicit_dest)
            elif long_option:
                parser_destinations.add(long_option.lstrip("-").replace("-", "_"))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "args"
            and isinstance(node.ctx, ast.Load)
        ):
            namespace_reads.add(node.attr)

    assert namespace_reads - parser_destinations == set()


def test_run_synthesis_supplies_the_fixed_delimiter_default_to_the_benchmark():
    source = Path("synthesis/run_synthesis.py").read_text(encoding="utf-8")

    assert (
        "require_delimiters=resolve_require_delimiters(" "args.dataset, cli_value=True)"
    ) in source


def test_synthesis_environment_names_the_isolated_cold_output():
    gpus = tuple(range(queue.POOLABLE_GPU_COUNT))
    env = queue.synthesis_environment(_job(), gpus, {"PATH": "/bin"}, Path("/repo"))
    joined = ",".join(str(gpu) for gpu in gpus)
    doubled = ",".join(str(gpu) for gpu in gpus for _ in range(2))

    assert env["CUDA_VISIBLE_DEVICES"] == joined
    # 16384 MiB reservation: two engines fit on one 40GB card -> 2 workers/GPU.
    assert env["CSD_EVAL_GPU_SLOTS"] == doubled
    assert env["CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX"] == "0.8"
    assert env["CSD_OUTPUT_NAME"] == "coldq_gsm-qwen35-2b_0719"
    assert env["CSD_OUTPUT_DIR"] == "/repo/outputs/generated/coldq_gsm-qwen35-2b_0719"


def test_synthesis_environment_pins_the_approved_current_claude_account():
    gpus = tuple(range(queue.POOLABLE_GPU_COUNT))
    env = queue.synthesis_environment(_job(), gpus, {"PATH": "/bin"}, Path("/repo"))

    assert env["CSD_CLAUDE_CONFIG_DIR"] == "/home/aadivyar/.claude-csd-ssdear"
    assert env["CSD_CLAUDE_EXPECTED_ACCOUNT"] == "ssdear@gmail.com"


def test_poolable_synthesis_environment_uses_the_reserved_gpu_bundle():
    """The bundle reaches the worker in the order it was handed out."""
    gpus = tuple(range(3, 3 - queue.POOLABLE_GPU_COUNT, -1))
    env = queue.synthesis_environment(_job(), gpus, {"PATH": "/bin"}, Path("/repo"))
    joined = ",".join(str(gpu) for gpu in gpus)
    doubled = ",".join(str(gpu) for gpu in gpus for _ in range(2))

    assert env["CUDA_VISIBLE_DEVICES"] == joined
    assert env["CSD_EVAL_GPU_SLOTS"] == doubled


def test_poolable_synthesis_environment_keeps_one_worker_per_gpu_for_big_models():
    """A 22000 MiB reservation cannot host two engines on one 40GB card."""
    job = _job()
    job["memory_reservation_mib"] = 22000
    gpus = tuple(range(queue.POOLABLE_GPU_COUNT))
    env = queue.synthesis_environment(job, gpus, {"PATH": "/bin"}, Path("/repo"))
    joined = ",".join(str(gpu) for gpu in gpus)

    assert env["CSD_EVAL_GPU_SLOTS"] == joined


def test_smiles_synthesis_environment_carries_the_job_gpu_util():
    """The span-sampling temperature is a benchmark fact now, not an env var."""
    job = _job("smiles")
    job["cell_id"] = "smiles-acrylates-qwen25-1p5b"
    job["output_name"] = "coldq_smiles-acrylates-qwen25-1p5b_20260724"
    job["gpu_mem_util"] = 0.4
    job["smiles_class"] = "acrylates"

    env = queue.synthesis_environment(job, (3,), {"PATH": "/bin"}, Path("/repo"))

    assert "CSD_CONSTRAINED_TEMPERATURE" not in env
    assert env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] == "0.4"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "CSD_EVAL_GPU_SLOTS" not in env


def test_non_smiles_synthesis_environment_does_not_force_constrained_temperature():
    env = queue.synthesis_environment(
        _job(), tuple(range(queue.POOLABLE_GPU_COUNT)), {"PATH": "/bin"}, Path("/repo")
    )

    assert "CSD_CONSTRAINED_TEMPERATURE" not in env
    assert env["CSD_VLLM_GPU_MEMORY_UTILIZATION"] == "0.8"


def test_heldout_environment_sets_no_constrained_temperature():
    env = queue.author_free_environment({"PATH": "/bin", "AWS_ACCESS_KEY_ID": "x"}, 3)

    assert "CSD_CONSTRAINED_TEMPERATURE" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "AWS_ACCESS_KEY_ID" not in env


def test_resolve_vllm_gpu_memory_utilization_prefers_env_override(monkeypatch):
    from synthesis import run_synthesis as rs

    monkeypatch.delenv("CSD_VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    assert rs._resolve_vllm_gpu_memory_utilization() == float(
        rs.VLLM_GPU_MEMORY_UTILIZATION
    )

    monkeypatch.setenv("CSD_VLLM_GPU_MEMORY_UTILIZATION", "0.4")
    assert rs._resolve_vllm_gpu_memory_utilization() == 0.4

    monkeypatch.setenv("CSD_VLLM_GPU_MEMORY_UTILIZATION", "  ")
    assert rs._resolve_vllm_gpu_memory_utilization() == float(
        rs.VLLM_GPU_MEMORY_UTILIZATION
    )


def test_resolve_vllm_gpu_memory_utilization_uses_per_model_default(monkeypatch):
    """Stale controllers may not export the env var; the child must still pick
    the per-model budget instead of the global 0.81 share
    (incident smiles-acrylates-qwen35-2b:2:memory:1784879589)."""
    from synthesis import run_synthesis as rs

    monkeypatch.delenv("CSD_VLLM_GPU_MEMORY_UTILIZATION", raising=False)
    assert rs._resolve_vllm_gpu_memory_utilization("Qwen/Qwen3.5-2B") == 0.4
    assert rs._resolve_vllm_gpu_memory_utilization("unknown/model") == float(
        rs.VLLM_GPU_MEMORY_UTILIZATION
    )

    # Explicit env override still wins.
    monkeypatch.setenv("CSD_VLLM_GPU_MEMORY_UTILIZATION", "0.5")
    assert rs._resolve_vllm_gpu_memory_utilization("Qwen/Qwen3.5-2B") == 0.5


def test_expected_runtime_gpu_mem_util_matches_shared_table():
    from synthesis.run_constants import VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL

    for model, runtime in queue.EXPECTED_RUNTIME_BY_MODEL.items():
        assert runtime["gpu_mem_util"] == VLLM_GPU_MEMORY_UTILIZATION_BY_MODEL[model]


def test_bundle_allocator_runs_poolable_cells_without_gpu_overlap():
    """Every cell gets its own GPUs, and the box runs dry rather than sharing."""
    width = queue.POOLABLE_GPU_COUNT
    snapshots = {gpu: {"used_mib": 0, "total_mib": 48_000} for gpu in range(4)}
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}
    job = _job()
    job["gpu_mem_util"] = 0.4

    handed_out: list[int] = []
    for cell in range(4 // width):
        bundle = queue.choose_gpu_bundle(job, snapshots, reservations, baseline)
        assert bundle is not None, f"cell {cell} got no bundle"
        assert len(bundle) == width
        assert not set(bundle) & set(handed_out), "a GPU was handed to two cells"
        handed_out.extend(bundle)
        for gpu in bundle:
            reservations[gpu][f"cell{cell}"] = queue.synthesis_required_memory_mib(
                job, snapshots[gpu]["total_mib"]
            )

    assert sorted(handed_out) == list(range((4 // width) * width))
    assert queue.choose_gpu_bundle(job, snapshots, reservations, baseline) is None
    assert queue.required_gpu_count(_job("smiles")) == 1


def test_heldout_command_binds_result_to_cell_commit_and_strategy():
    job = _job()
    job["git_commit"] = "a" * 40
    csd = Path("/repo/GeneratedCSD.py")

    command = queue.heldout_command(job, Path("/env/python"), csd)

    assert command[command.index("--provenance-cell-id") + 1] == job["cell_id"]
    assert (
        command[command.index("--provenance-manifest-commit") + 1] == job["git_commit"]
    )


def test_heldout_environment_removes_paid_author_credentials():
    env = queue.author_free_environment(
        {
            "PATH": "/bin",
            "AWS_BEARER_TOKEN_BEDROCK": "secret",
            "OPENAI_API_KEY": "secret",
            "CSD_EVAL_GPU_SLOTS": "0,1",
            "CSD_EVAL_POOL_SIZE": "2",
        },
        1,
        gpu_memory_utilization_max=0.4,
    )

    assert env == {
        "PATH": "/bin",
        "CUDA_VISIBLE_DEVICES": "1",
        "CSD_VLLM_GPU_MEMORY_UTILIZATION_MAX": "0.4",
    }


def test_full_memory_gate_requires_an_idle_unreserved_gpu():
    job = _job("smiles")
    job["requires_exclusive_gpu"] = True
    snapshots = {
        0: {"used_mib": 500, "total_mib": 48_000},
        1: {"used_mib": 1_500, "total_mib": 48_000},
    }
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {0: {}, 1: {}}

    assert queue.choose_gpu_bundle(job, snapshots, reservations, baseline) == (0,)

    reservations[0]["other-cell"] = 1
    assert queue.choose_gpu_bundle(job, snapshots, reservations, baseline) is None


def test_pinned_heldout_csd_rejects_hash_drift(tmp_path: Path):
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# accepted\n", encoding="utf-8")
    job = {
        "heldout_csd_path": str(csd),
        "heldout_csd_sha256": hashlib.sha256(csd.read_bytes()).hexdigest(),
    }
    assert queue.pinned_heldout_csd(job, tmp_path) == csd
    csd.write_text("# changed\n", encoding="utf-8")
    with pytest.raises(queue.ConfigError, match="held-out CSD hash"):
        queue.pinned_heldout_csd(job, tmp_path)


def test_dispatch_preserves_manifest_priority_on_one_gpu():
    first = _job("smiles")
    first["cell_id"] = "first"
    second = _job("smiles")
    second["cell_id"] = "second"
    started = []

    def worker(job, gpus):
        started.append((job["cell_id"], gpus))
        return 0

    queue.dispatch(
        [first, second],
        snapshot=lambda: {0: {"used_mib": 0, "total_mib": 48_000}},
        worker=worker,
        poll_seconds=0.001,
    )

    assert started == [("first", (0,)), ("second", (0,))]


def test_dispatch_finishes_one_queue_phase_before_starting_the_next():
    first = _job("smiles")
    first.update({"cell_id": "phase-one", "queue_phase": 1})
    second = _job("smiles")
    second.update({"cell_id": "phase-two", "queue_phase": 2})
    second_started = threading.Event()
    started: list[str] = []

    def worker(job, _gpus):
        started.append(job["cell_id"])
        if job["cell_id"] == "phase-one":
            assert not second_started.wait(timeout=0.05)
        else:
            second_started.set()
        return 0

    queue.dispatch(
        [first, second],
        snapshot=lambda: {
            0: {"used_mib": 0, "total_mib": 48_000},
            1: {"used_mib": 0, "total_mib": 48_000},
        },
        worker=worker,
        poll_seconds=0.001,
    )

    assert started == ["phase-one", "phase-two"]
    assert second_started.is_set()


def test_synthesis_reservation_matches_what_all_workers_will_actually_take():
    """Reserve every pooled worker's vLLM budget, not just one worker.

    synthesis_environment starts two workers per physical GPU for small
    poolable models. The allocator must reserve both workers before placing
    the job beside another user's process.
    """
    job = _job()
    job["gpu_mem_util"] = 0.4

    # Two workers, each max(16384, ceil(0.4 * 48000)=19200).
    assert queue.synthesis_required_memory_mib(job, 48_000) == 38_400

    # The per-worker floor still wins when the fraction lands under it.
    job["gpu_mem_util"] = 0.1
    assert queue.synthesis_required_memory_mib(job, 48_000) == 32_768


def test_qwen35_2b_two_workers_fit_the_30672_mib_free_memory_boundary():
    """The settled Qwen3.5-2B floor admits both GPUs at the exact boundary."""
    job = _job("spider")
    job["gpu_mem_util"] = 0.35
    job["memory_reservation_mib"] = queue.EXPECTED_RUNTIME_BY_MODEL[
        job["eval_model"]
    ]["memory_reservation_mib"]
    total = 40_960
    snapshots = {
        1: {"used_mib": total - 32_859, "free_mib": 32_859, "total_mib": total},
        2: {"used_mib": total - 40_432, "free_mib": 40_432, "total_mib": total},
    }
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}

    assert job["eval_model"] == "Qwen/Qwen3.5-2B"
    assert queue.EXPECTED_RUNTIME_BY_MODEL[job["eval_model"]][
        "memory_reservation_mib"
    ] == 14_336
    assert queue.eval_workers_per_gpu(job) == 2
    assert queue.required_gpu_count(job) == 2
    assert queue.synthesis_required_memory_mib(job, total) == 28_672
    assert tuple(sorted(queue.choose_gpu_bundle(job, snapshots, reservations, baseline))) == (1, 2)

    for low_gpu in snapshots:
        below_boundary = {
            gpu: dict(snapshot) for gpu, snapshot in snapshots.items()
        }
        below_boundary[low_gpu]["free_mib"] = 30_671
        below_boundary[low_gpu]["used_mib"] = total - 30_671
        below_baseline = {
            gpu: dict(snapshot) for gpu, snapshot in below_boundary.items()
        }
        assert (
            queue.choose_gpu_bundle(job, below_boundary, reservations, below_baseline)
            is None
        ), f"GPU {low_gpu} below 30672 MiB was incorrectly admitted"


def test_poolable_double_workers_reject_a_partly_occupied_gpu():
    job = _job("spider")
    job["gpu_mem_util"] = 0.35
    total = 40_960
    snapshots = {
        0: {"used_mib": 14_900, "free_mib": 26_060, "total_mib": total},
        2: {"used_mib": 0, "free_mib": total, "total_mib": total},
    }
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}

    env = queue.synthesis_environment(job, (0, 2), {}, Path("/repo"))
    assert env["CSD_EVAL_GPU_SLOTS"] == "0,0,2,2"
    assert queue.synthesis_required_memory_mib(job, total) == 32_768
    assert (
        queue.choose_gpu_bundle(job, snapshots, reservations, baseline) is None
    )


def test_two_cells_share_one_card_without_overfilling_it():
    """Stack two cells on one GPU in a single pass, and stop at two.

    This is what the live spider run does on GPU 3. The allocator sees one
    fixed memory reading for the whole pass, so the only thing telling it the
    first cell is already there is the reservation it just wrote down. That
    running total has to be added to the next cell's demand, and the pair has
    to still leave the safety margin free.
    """
    total = 40_960
    snapshots = {0: {"used_mib": 0, "total_mib": total}}
    baseline = {0: dict(snapshots[0])}
    reservations = {0: {}}

    def spider_cell(cell_id, util, floor):
        job = _job("smiles")
        job["cell_id"] = cell_id
        job["gpu_mem_util"] = util
        job["memory_reservation_mib"] = floor
        return job

    # The two cells actually sharing GPU 3 right now.
    first = spider_cell("spider-qwen35-4b", 0.45, 19_000)
    second = spider_cell("spider-qwen25-1p5b", 0.4, 16_000)
    third = spider_cell("spider-qwen35-2b", 0.4, 16_384)

    bundle = queue.choose_gpu_bundle(first, snapshots, reservations, baseline)
    assert bundle == (0,)
    first_need = queue.synthesis_required_memory_mib(first, total)
    reservations[0]["spider-qwen35-4b"] = first_need

    # Same pass, same memory reading - only the reservation has changed.
    bundle = queue.choose_gpu_bundle(second, snapshots, reservations, baseline)
    assert bundle == (0,), "second cell was refused a card that has room for it"
    second_need = queue.synthesis_required_memory_mib(second, total)
    reservations[0]["spider-qwen25-1p5b"] = second_need

    assert (
        first_need + second_need + queue.GPU_SAFETY_MIB <= total
    ), "the pair the allocator admitted does not actually fit with the margin"

    # A third cell of the same size must be turned away, not squeezed in.
    assert queue.choose_gpu_bundle(third, snapshots, reservations, baseline) is None


def test_dispatch_starts_another_cell_as_soon_as_a_gpu_frees_up():
    """A card emptying mid-run has to pull the next cell in, not sit idle.

    GPU 0 has room for exactly one cell. GPU 1 starts full with someone else's
    job and empties partway through. The second cell must land on GPU 1 once it
    empties, rather than waiting for the first cell to finish.
    """
    first = _job("smiles")
    first["cell_id"] = "first"
    first["gpu_mem_util"] = 0.4
    second = _job("smiles")
    second["cell_id"] = "second"
    second["gpu_mem_util"] = 0.4

    started: list[tuple[str, tuple[int, ...]]] = []
    second_started = threading.Event()

    def worker(job, gpus):
        started.append((str(job["cell_id"]), gpus))
        if job["cell_id"] == "first":
            # Hold GPU 0 so the queue cannot just reuse it for the second cell.
            assert second_started.wait(timeout=5), "second cell never started"
        else:
            second_started.set()
        return 0

    polls = {"count": 0}

    def snapshot():
        polls["count"] += 1
        if polls["count"] > 40:
            raise AssertionError(f"dispatch never placed both cells; started={started}")
        # Room for one 19200 MiB cell plus the 2000 MiB safety margin, not two.
        gpu1_used = 47_000 if polls["count"] < 4 else 20_000
        return {
            0: {"used_mib": 20_000, "total_mib": 48_000},
            1: {"used_mib": gpu1_used, "total_mib": 48_000},
        }

    queue.dispatch(
        [first, second], snapshot=snapshot, worker=worker, poll_seconds=0.001
    )

    assert [cell for cell, _ in started] == ["first", "second"]
    assert dict(started)["first"] == (0,)
    assert dict(started)["second"] == (1,), "second cell did not take the freed card"
    assert polls["count"] >= 4, "second cell started before GPU 1 was free"


def test_compiled_csd_uses_best_threshold_candidate_after_exhaustion(tmp_path):
    output_name = "coldq_gsm-qwen35-2b_0719"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "failed-run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / output_name / "latest_run.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    weaker = run_dir / "compiled-weaker"
    closest = run_dir / "compiled-closest"
    weaker.mkdir()
    closest.mkdir()
    (weaker / "GeneratedCSD.py").write_text("# weaker\n", encoding="utf-8")
    (closest / "GeneratedCSD.py").write_text("# closest\n", encoding="utf-8")
    job = _job()
    job["git_commit"] = "a" * 40
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
        "attempts": [
            {
                "attempt_number": 1,
                "compilation": {"success": True, "output_dir": str(weaker)},
                "evaluation": {"accuracy": 0.6, "syntax_rate": 0.7, "num_examples": 49},
            },
            {
                "attempt_number": 2,
                "compilation": {"success": True, "output_dir": str(closest)},
                "evaluation": {
                    "accuracy": 0.58,
                    "syntax_rate": 0.9,
                    "num_examples": 49,
                },
            },
        ],
    }
    (results_dir / "failure_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )

    selected = queue.compiled_csd(
        tmp_path, output_name, min_accuracy=0.65, min_syntax_rate=0.9, job=job
    )

    assert selected == closest / "GeneratedCSD.py"


def test_compiled_csd_treats_a_truncated_report_as_incomplete(tmp_path):
    output_name = "coldq_gsm-qwen35-2b_0719"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "interrupted-run"
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / output_name / "latest_run.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    (results_dir / "success_report.json").write_text("{", encoding="utf-8")

    assert (
        queue.compiled_csd(
            tmp_path,
            output_name,
            min_accuracy=0.65,
            min_syntax_rate=0.9,
            job={**_job(), "git_commit": "a" * 40},
        )
        is None
    )


def test_exhaustion_report_must_match_the_exact_cold_invocation():
    job = _job()
    job["git_commit"] = "a" * 40
    job["launch_commit"] = "b" * 40
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
    }

    assert queue.report_matches_job(report, job, require_exhausted=True)

    report["run_configuration"]["evaluation"]["eval_model"] = "wrong-model"
    assert not queue.report_matches_job(report, job, require_exhausted=True)
    report["run_configuration"]["evaluation"]["eval_model"] = job["eval_model"]
    report["total_attempts"] -= 1
    assert not queue.report_matches_job(report, job, require_exhausted=True)

    report["total_attempts"] = job["max_iterations"]
    report["run_configuration"]["author_model"]["reasoning_budget_tokens"] = 2048
    assert not queue.report_matches_job(report, job, require_exhausted=True)

    smiles_job = _job("smiles")
    smiles_job.update(
        {
            "git_commit": "a" * 40,
            "smiles_class": "isocyanates",
            "task": queue.SMILES_TASK,
        }
    )
    smiles_report = {
        "total_attempts": smiles_job["max_iterations"],
        "run_configuration": _matching_run_configuration(smiles_job),
    }
    assert queue.report_matches_job(smiles_report, smiles_job, require_exhausted=True)
    smiles_report["run_configuration"]["evaluation"]["smiles_classes"] = ["acrylates"]
    assert not queue.report_matches_job(
        smiles_report, smiles_job, require_exhausted=True
    )


def test_smiles_report_accepts_the_runtime_string_class_shape():
    job = _job("smiles")
    job.update(
        {
            "git_commit": "a" * 40,
            "smiles_class": "acrylates",
            "task": queue.SMILES_TASK,
        }
    )
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
    }
    report["run_configuration"]["author_model"].update(
        {"backend": "claude", "model": queue.AUTHOR_MODEL}
    )
    report["run_configuration"]["evaluation"]["smiles_classes"] = "acrylates"

    assert queue.report_matches_job(report, job, require_exhausted=True)


def test_report_matches_when_job_omits_launch_and_git_commit():
    """Corrected-campaign jobs ship with null commits; success reports still pin HEAD."""
    job = _job("smiles")
    job.update(
        {
            "git_commit": None,
            "launch_commit": None,
            "smiles_class": "acrylates",
            "task": queue.SMILES_TASK,
            "max_iterations": 40,
        }
    )
    report = {
        "total_attempts": 3,
        "run_configuration": _matching_run_configuration(
            {**job, "git_commit": "d" * 40}
        ),
    }

    assert queue.report_matches_job(report, job, require_exhausted=False)

    job["launch_commit"] = "b" * 40
    assert not queue.report_matches_job(report, job, require_exhausted=False)
    report["run_configuration"]["git_commit"] = "b" * 40
    assert queue.report_matches_job(report, job, require_exhausted=False)



def test_stamp_job_commit_from_report_fills_null_commits(tmp_path):
    output_name = "coldq_smiles-acrylates-qwen35-2b"
    run_dir = tmp_path / "outputs" / "generated" / output_name / "run-1"
    results = run_dir / "results"
    results.mkdir(parents=True)
    (tmp_path / "outputs" / "generated" / output_name / "latest_run.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    commit = "e" * 40
    (results / "success_report.json").write_text(
        json.dumps({"run_configuration": {"git_commit": commit}}),
        encoding="utf-8",
    )
    job = _job("smiles")
    job.update(
        {
            "git_commit": None,
            "launch_commit": None,
            "smiles_class": "acrylates",
            "output_name": output_name,
        }
    )
    stamped = queue.stamp_job_commit_from_report(job, tmp_path, output_name)
    assert stamped["git_commit"] == commit
    assert job["git_commit"] is None

def test_non_smiles_report_ignores_the_irrelevant_runtime_smiles_default():
    job = _job()
    job["git_commit"] = "a" * 40
    report = {
        "total_attempts": job["max_iterations"],
        "run_configuration": _matching_run_configuration(job),
    }
    report["run_configuration"]["author_model"].update(
        {"backend": "claude", "model": queue.AUTHOR_MODEL}
    )
    report["run_configuration"]["evaluation"][
        "smiles_classes"
    ] = "acrylates,chain_extenders,isocyanates"

    assert queue.report_matches_job(report, job, require_exhausted=True)


def test_corrected_launch_guard_requires_an_independent_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(queue, "require_synthesis_unblocked", lambda _repo: None)

    with pytest.raises(queue.ConfigError, match="independent corrected approval"):
        queue.validate_corrected_launch(
            tmp_path,
            tmp_path / "manifest.json",
            None,
        )


@pytest.mark.parametrize(
    "gpus",
    [None, (0, 2), (1,), (2, 3, 0), (1, 2, 3)],
)
def test_corrected_campaign_rejects_any_gpu_scope_except_approved(gpus):
    with pytest.raises(queue.ConfigError, match="exactly GPUs 0,2,3"):
        queue.validate_corrected_gpu_scope(gpus)


def test_corrected_campaign_accepts_only_approved_gpu_scopes():
    queue.validate_corrected_gpu_scope((0, 2, 3))
    queue.validate_corrected_gpu_scope((0, 1, 2, 3))


def test_spider_only_resume_requires_exact_prefixes_and_gpu_scope():
    queue.validate_corrected_resume_scope((0, 1, 2, 3), ["gsm-", "smiles-"])

    with pytest.raises(queue.ConfigError, match="requires exactly GPUs 0,1,2,3"):
        queue.validate_corrected_resume_scope((0, 2, 3), ["gsm-", "smiles-"])
    with pytest.raises(queue.ConfigError, match="requires exactly exclusions"):
        queue.validate_corrected_resume_scope((0, 1, 2, 3), ["gsm-"])
    with pytest.raises(queue.ConfigError, match="requires exactly exclusions"):
        queue.validate_corrected_resume_scope((0, 1, 2, 3), ["spider-"])

    with pytest.raises(queue.ConfigError, match="requires exactly exclusions"):
        queue.validate_corrected_resume_scope((0, 1, 2, 3), [])


def test_corrected_campaign_filters_only_after_full_launch_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    commit = "f" * 40
    events = []
    jobs = [
        {"cell_id": "gsm-qwen25-1p5b"},
        {"cell_id": "smiles-acrylates-qwen25-1p5b"},
        {"cell_id": "spider-qwen25-1p5b"},
        {"cell_id": "spider-qwen25-7b"},
        {"cell_id": "spider-qwen35-2b"},
        {"cell_id": "spider-qwen35-4b"},
    ]

    def validate_launch(repo, manifest, approval):
        events.append(("validate", repo, manifest, approval))

    def load_manifest(path):
        events.append(("load", path))
        return commit, [dict(job) for job in jobs]

    monkeypatch.setattr(queue, "validate_corrected_launch", validate_launch)
    monkeypatch.setattr(queue, "load_manifest", load_manifest)
    monkeypatch.setattr(
        queue,
        "verify_repo_version",
        lambda _repo, _commit, _attestation: commit,
    )
    monkeypatch.setattr(queue, "required_gpu_count", lambda _job: 2)
    monkeypatch.setattr(
        queue,
        "synthesis_command",
        lambda job, python: [str(python), job["cell_id"]],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cold_synthesis_queue.py",
            "--repo",
            str(tmp_path),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--python",
            sys.executable,
            "--lock-file",
            str(tmp_path / "controller.lock"),
            "--state-dir",
            str(tmp_path / "state"),
            "--corrected-approval",
            str(tmp_path / "approval.json"),
            "--campaign-profile",
            "full-baseline-corrected-20260805",
            "--gpus",
            "0,1,2,3",
            "--exclude-cell-prefix",
            "gsm-",
            "--exclude-cell-prefix",
            "smiles-",
            "--dry-run",
        ],
    )

    with caplog.at_level("WARNING"):
        assert queue.main() == 0

    assert [event[0] for event in events] == ["validate", "load"]
    assert "remaining=4" in caplog.text
    assert "dry-run cell=spider-qwen25-1p5b" in caplog.text
    assert "dry-run cell=spider-qwen25-7b" in caplog.text
    assert "dry-run cell=spider-qwen35-2b" in caplog.text
    assert "dry-run cell=spider-qwen35-4b" in caplog.text
    assert "dry-run cell=gsm-" not in caplog.text
    assert "dry-run cell=smiles-" not in caplog.text


def test_recovery_exhaustion_counts_restored_and_remaining_attempts():
    job = _job()
    job.update(
        {
            "git_commit": "a" * 40,
            "max_iterations": 36,
            "initial_attempt_offset": 4,
            "initial_completed_evaluations": 3,
        }
    )
    configuration = _matching_run_configuration(job)
    configuration["author_model"].update(
        {"backend": "claude", "model": queue.AUTHOR_MODEL}
    )
    report = {
        "total_attempts": 39,
        "run_configuration": configuration,
    }

    assert queue.report_matches_job(report, job, require_exhausted=True)

    report["total_attempts"] = 38
    assert not queue.report_matches_job(report, job, require_exhausted=True)


def test_exhausted_synthesis_still_runs_heldout_for_best_attempt(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    best_csd = tmp_path / "GeneratedCSD.py"
    best_csd.write_text("# best failed attempt\n", encoding="utf-8")
    selected = iter([None, best_csd])
    calls = []
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: next(selected))
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args, **kwargs: True)
    monkeypatch.setattr(queue, "heldout_is_complete", lambda *args: len(calls) >= 2)

    def run(command, **kwargs):
        calls.append(command)
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(queue, "_run_command_teeing_stdout", run)

    status = queue.run_job(
        job,
        (0, 1),
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 0
    assert calls[1][1:3] == ["-m", "synthesis.scripts.reevaluate_compiled_csd"]
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "complete_loss"


def test_unexpected_exit_one_is_an_error_not_a_scientific_loss(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: None)
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args, **kwargs: False)
    monkeypatch.setattr(queue, "_run_command_teeing_stdout", lambda *args, **kwargs: 1)

    status = queue.run_job(
        job,
        (0, 1),
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 1
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "error"


def test_existing_heldout_must_be_complete_before_restart_skips_cell(
    tmp_path, monkeypatch
):
    job = _job()
    job["git_commit"] = "a" * 40
    heldout = tmp_path / "heldout.json"
    job["heldout_output_json"] = str(heldout)
    heldout.write_text("{", encoding="utf-8")
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# strategy\n", encoding="utf-8")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: csd)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        heldout.write_text(
            json.dumps(
                {
                    "accuracy": 0.5,
                    "syntax_rate": 1.0,
                    "metrics": {"num_examples": 49},
                    "answers": [{} for _ in range(49)],
                    "eval_split": {
                        "gsm_split_file": job["heldout_split_file"],
                        "gsm_split_name": "test",
                        "spider_split_file": None,
                        "spider_split_name": None,
                        "bar_split_name": None,
                    },
                    "reevaluation_provenance": {
                        "cell_id": job["cell_id"],
                        "manifest_commit": job["git_commit"],
                        "dataset": job["dataset"],
                        "eval_model": job["eval_model"],
                        "compiled_csd_path": str(csd.resolve()),
                        "compiled_csd_sha256": hashlib.sha256(
                            csd.read_bytes()
                        ).hexdigest(),
                        "sample_size": job["heldout_sample_size"],
                        "max_steps": job["eval_max_steps"],
                        "step_token_budget": 1,
                        "smiles_class": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(queue, "_run_command_teeing_stdout", run)

    assert (
        queue.run_job(
            job,
            (0, 1),
            repo=tmp_path,
            python=Path("/env/python"),
            state_dir=tmp_path / "state",
        )
        == 0
    )
    assert len(calls) == 1
    assert queue.heldout_is_complete(heldout, job)
    payload = json.loads(heldout.read_text())
    payload["reevaluation_provenance"]["eval_model"] = "wrong-model"
    heldout.write_text(json.dumps(payload), encoding="utf-8")
    assert not queue.heldout_is_complete(heldout, job)


def _complete_heldout_payload(job, csd, manifest_commit):
    return {
        "accuracy": 0.5,
        "syntax_rate": 1.0,
        "metrics": {"num_examples": job["heldout_sample_size"]},
        "answers": [{} for _ in range(job["heldout_sample_size"])],
        "eval_split": {
            "gsm_split_file": None,
            "gsm_split_name": None,
            "spider_split_file": job["heldout_split_file"],
            "spider_split_name": "test",
            "bar_split_name": None,
        },
        "reevaluation_provenance": {
            "cell_id": job["cell_id"],
            "manifest_commit": manifest_commit,
            "dataset": job["dataset"],
            "eval_model": job["eval_model"],
            "compiled_csd_path": str(csd.resolve()),
            "compiled_csd_sha256": hashlib.sha256(csd.read_bytes()).hexdigest(),
            "sample_size": job["heldout_sample_size"],
            "max_steps": job["eval_max_steps"],
            "step_token_budget": 1,
            "smiles_class": None,
        },
    }


def test_prior_heldout_requires_exact_manifest_commit_and_artifact_hash(tmp_path):
    job = _job("spider")
    current_commit = hashlib.sha1(b"current manifest").hexdigest()
    prior_commit = hashlib.sha1(b"prior manifest").hexdigest()
    other_commit = hashlib.sha1(b"other manifest").hexdigest()
    job["git_commit"] = current_commit
    job["heldout_manifest_commit"] = prior_commit
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# strategy\n", encoding="utf-8")
    heldout = tmp_path / "heldout.json"
    payload = _complete_heldout_payload(job, csd, prior_commit)
    raw = json.dumps(payload).encode("utf-8")
    heldout.write_bytes(raw)
    job["heldout_output_json_sha256"] = hashlib.sha256(raw).hexdigest()

    assert queue.heldout_is_complete(heldout, job)

    missing_commit = dict(job)
    missing_commit.pop("heldout_manifest_commit")
    assert not queue.heldout_is_complete(heldout, missing_commit)

    missing_hash = dict(job)
    missing_hash.pop("heldout_output_json_sha256")
    assert not queue.heldout_is_complete(heldout, missing_hash)

    mismatched_commit = dict(job)
    mismatched_commit["heldout_manifest_commit"] = other_commit
    assert not queue.heldout_is_complete(heldout, mismatched_commit)

    mismatched_hash = dict(job)
    mismatched_hash["heldout_output_json_sha256"] = "0" * 64
    assert not queue.heldout_is_complete(heldout, mismatched_hash)

    heldout.write_bytes(raw + b"\n")
    assert not queue.heldout_is_complete(heldout, job)


def test_current_commit_heldout_remains_valid_without_prior_binding(tmp_path):
    job = _job("spider")
    current_commit = hashlib.sha1(b"current manifest").hexdigest()
    job["git_commit"] = current_commit
    csd = tmp_path / "GeneratedCSD.py"
    csd.write_text("# strategy\n", encoding="utf-8")
    heldout = tmp_path / "heldout.json"
    heldout.write_text(
        json.dumps(_complete_heldout_payload(job, csd, current_commit)),
        encoding="utf-8",
    )

    assert queue.heldout_is_complete(heldout, job)


def test_restart_preserves_exhausted_state_while_running_heldout(tmp_path, monkeypatch):
    job = _job()
    job["git_commit"] = "a" * 40
    job["heldout_output_json"] = str(tmp_path / "heldout.json")
    best_csd = tmp_path / "GeneratedCSD.py"
    best_csd.write_text("# best failed attempt\n", encoding="utf-8")
    monkeypatch.setattr(queue, "compiled_csd", lambda *args, **kwargs: best_csd)
    monkeypatch.setattr(queue, "synthesis_was_exhausted", lambda *args: True)
    calls = []
    monkeypatch.setattr(queue, "heldout_is_complete", lambda *args: len(calls) >= 1)

    def run(command, **kwargs):
        calls.append(command)
        return 0

    monkeypatch.setattr(queue, "_run_command_teeing_stdout", run)

    status = queue.run_job(
        job,
        (0, 1),
        repo=tmp_path,
        python=Path("/env/python"),
        state_dir=tmp_path / "state",
    )

    assert status == 0
    assert len(calls) == 1
    state = json.loads((tmp_path / "state" / "gsm-qwen35-2b.json").read_text())
    assert state["status"] == "complete_loss"


def test_cold_queue_service_uses_work_env_and_kills_its_process_group():
    unit = (
        Path(__file__).parents[2]
        / "deploy"
        / "focal"
        / "systemd"
        / "csd-cold-synthesis-queue.service"
    ).read_text()

    assert (
        "ExecStart=/apps/conda/aadivyar/envs/csd/bin/python "
        "-m scripts.runtime.run_cold_synthesis_queue "
    ) in unit
    assert "/scripts/runtime/run_cold_synthesis_queue.py" not in unit
    assert "EnvironmentFile=/home/aadivyar/csd-generation/.env" in unit
    assert "2026-07-19-exhaustive-cold-queue-manifest.json" in unit
    assert "KillMode=control-group" in unit
    assert "Restart=on-failure" in unit
    assert "initial-strategy" not in unit
    assert "warm" not in unit.lower()


def test_repo_version_allows_manifest_only_commit_but_rejects_code_drift(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    script = tmp_path / "scripts" / "runtime" / "worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "scripts/runtime/worker.py"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "code"], cwd=tmp_path, check=True)
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    saved = tmp_path / "saved-results" / "manifest.json"
    saved.parent.mkdir()
    saved.write_text("{}\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "saved-results/manifest.json"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "manifest"], cwd=tmp_path, check=True)

    queue.verify_repo_version(tmp_path, code_commit)

    script.write_text("VERSION = 2\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "uncommitted code changes" in str(error)
    else:
        raise AssertionError("dirty runtime code must block launch")

    script.write_text("VERSION = 1\n", encoding="utf-8")
    untracked = tmp_path / "synthesis" / "new_path.py"
    untracked.parent.mkdir()
    untracked.write_text("ENABLED = True\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "untracked code changes" in str(error)
    else:
        raise AssertionError("untracked synthesis code must block launch")

    untracked.unlink()
    gsm_source = tmp_path / "legacy" / "CRANE" / "src" / "gsm_symbolic" / "parser.py"
    gsm_source.parent.mkdir(parents=True)
    gsm_source.write_text("CHANGED = True\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, code_commit)
    except queue.ConfigError as error:
        assert "untracked code changes" in str(error)
    else:
        raise AssertionError("untracked GSM source code must block launch")


def test_verified_repair_attestation_allows_only_its_exact_dirty_code(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    script = tmp_path / "scripts" / "runtime" / "worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", str(script.relative_to(tmp_path))], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "commit", "-qm", "code"], cwd=tmp_path, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    script.write_text("VERSION = 2\n", encoding="utf-8")
    attestation = tmp_path / "repair.json"
    attestation.write_text(
        json.dumps(
            {
                "base_commit": commit,
                "files": {
                    "scripts/runtime/worker.py": hashlib.sha256(
                        script.read_bytes()
                    ).hexdigest()
                },
                "verifier_exit": 0,
            }
        ),
        encoding="utf-8",
    )

    queue.verify_repo_version(tmp_path, commit, attestation)

    script.write_text("VERSION = 3\n", encoding="utf-8")
    try:
        queue.verify_repo_version(tmp_path, commit, attestation)
    except queue.ConfigError as error:
        assert "repair attestation" in str(error)
    else:
        raise AssertionError("code changed after attestation must block launch")


def test_saved_exhaustive_manifest_matches_the_approved_call_budget():
    repo = Path(__file__).parents[2]
    manifest = repo / "saved-results" / "2026-07-19-exhaustive-cold-queue-manifest.json"

    commit, jobs = queue.load_manifest(manifest)
    # The July artifact preserves the task text used at launch. The live
    # Spider task changed later, so this test checks the frozen budget only.

    assert len(commit) == 40
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo,
        check=True,
    )
    assert len(jobs) == 21
    assert sum(job["max_iterations"] for job in jobs) == 834
    assert sum(job["interrupted_author_calls"] for job in jobs) == 8
    assert (
        sum(job["max_iterations"] + job["interrupted_author_calls"] for job in jobs)
        == queue.APPROVED_AUTHOR_CALL_CAP
        == 842
    )


def test_qwen35_9b_isocyanates_is_an_approved_cold_queue_cell():
    spec = queue.EXPECTED_CELLS["smiles-qwen35-9b-isocyanates"]

    assert spec == {
        "dataset": "smiles",
        "eval_model": "Qwen/Qwen3.5-9B",
        "max_iterations": 40,
        "interrupted_author_calls": 0,
        "eval_sample_size": 50,
        "heldout_sample_size": 100,
        "eval_max_steps": 400,
        "task": queue.SMILES_TASK,
        "smiles_class": "isocyanates",
    }
    assert queue.EXPECTED_RUNTIME_BY_MODEL[spec["eval_model"]] == {
        "memory_reservation_mib": 25000,
        "gpu_mem_util": 0.6,
    }


def test_focal_service_launches_only_the_remaining_six_with_shared_caches():
    service = Path("deploy/focal/systemd/csd-cold-synthesis-queue.service").read_text(
        encoding="utf-8"
    )

    assert "Description=Dynamic CSD remaining six-cell cold synthesis queue" in service
    assert "Environment=CSD_CACHE_ROOT=/home/aadivyar/csd-generation/cache" in service
    assert "Environment=SYNCODE_CACHE=/home/aadivyar/csd-generation/cache/" in service
    assert "--gpus 0,2,3" in service
    assert (
        "--lock-file /home/aadivyar/csd-generation/.context/remaining_six_20260729/controller.lock"
        in service
    )
    assert (
        "--state-dir /home/aadivyar/csd-generation/.context/remaining_six_20260729/state"
        in service
    )
    assert service.count("--exclude-cell-prefix") == 5
    for prefix in (
        "spider-",
        "smiles-acrylates-qwen25-",
        "smiles-acrylates-qwen35-2b",
        "smiles-chain_extenders-",
        "smiles-isocyanates-",
    ):
        assert f"--exclude-cell-prefix {prefix}" in service


def test_exhaustive_campaign_requires_all_twenty_one_exact_cells_and_unique_outputs(
    tmp_path, monkeypatch
):
    expected_ids = {
        "gsm-qwen25-1p5b",
        "gsm-qwen25-7b",
        "gsm-qwen35-2b",
        "gsm-qwen35-4b",
        "spider-qwen25-1p5b",
        "spider-qwen25-7b",
        "spider-qwen35-2b",
        "spider-qwen35-4b",
        "smiles-acrylates-qwen25-1p5b",
        "smiles-acrylates-qwen25-7b",
        "smiles-acrylates-qwen35-2b",
        "smiles-acrylates-qwen35-4b",
        "smiles-chain_extenders-qwen25-1p5b",
        "smiles-chain_extenders-qwen25-7b",
        "smiles-chain_extenders-qwen35-2b",
        "smiles-chain_extenders-qwen35-4b",
        "smiles-isocyanates-qwen25-1p5b",
        "smiles-isocyanates-qwen25-7b",
        "smiles-isocyanates-qwen35-2b",
        "smiles-isocyanates-qwen35-4b",
        "smiles-qwen35-9b-isocyanates",
    }
    assert set(queue.EXPECTED_CELLS) == expected_ids
    interrupted = {
        cell: spec["interrupted_author_calls"]
        for cell, spec in queue.EXPECTED_CELLS.items()
        if spec["interrupted_author_calls"]
    }
    assert interrupted == {
        "gsm-qwen25-1p5b": 3,
        "gsm-qwen25-7b": 3,
        "gsm-qwen35-2b": 2,
    }
    assert sum(spec["max_iterations"] for spec in queue.EXPECTED_CELLS.values()) == 834
    assert (
        sum(
            spec["max_iterations"] + spec["interrupted_author_calls"]
            for spec in queue.EXPECTED_CELLS.values()
        )
        == queue.APPROVED_AUTHOR_CALL_CAP
        == 842
    )
    jobs = []
    for cell, spec in queue.EXPECTED_CELLS.items():
        job = _job(spec["dataset"])
        job.update(spec)
        job.update(queue.EXPECTED_RUNTIME_BY_MODEL[job["eval_model"]])
        job["eval_max_seconds"] = 600
        job["cell_id"] = cell
        job["baseline_num_correct"] = 0
        job["baseline_num_examples"] = job["eval_sample_size"]
        job["baseline_source"] = f"outputs/baselines/{cell}.json"
        job["min_accuracy"] = queue.accuracy_bar(0, job["baseline_num_examples"]).min_accuracy
        job["output_name"] = f"coldq_{cell}_0719"
        job["log_file"] = f"outputs/generated/{job['output_name']}/run.log"
        job["heldout_output_json"] = f"outputs/reeval/exhaustive_0719/{cell}.json"
        if job["dataset"] == "gsm_symbolic":
            job[
                "heldout_split_file"
            ] = "/repo/gsm_symbolic_crane_proportional_49x49_seed123.json"
        elif job["dataset"] == "spider":
            job[
                "heldout_split_file"
            ] = "/repo/spider_dev_proportional_300x300_seed334.json"
        else:
            job.pop("heldout_split_file", None)
        jobs.append(job)

    queue.validate_exhaustive_campaign(jobs)

    monkeypatch.setattr(queue, "APPROVED_AUTHOR_CALL_CAP", 841)
    try:
        queue.validate_exhaustive_campaign(jobs)
    except queue.ConfigError as error:
        assert "author-call accounting must total 841, got 842" in str(error)
    else:
        raise AssertionError("a campaign above the approved call cap must be rejected")
    monkeypatch.setattr(queue, "APPROVED_AUTHOR_CALL_CAP", 842)

    try:
        queue.validate_exhaustive_campaign(jobs, repo=tmp_path)
    except queue.ConfigError as error:
        assert "baseline_source file is missing" in str(error)
    else:
        raise AssertionError("missing baseline evidence files must block launch")

    try:
        queue.validate_exhaustive_campaign(jobs[:-1])
    except queue.ConfigError as error:
        assert "exactly the 21 approved cells" in str(error)
    else:
        raise AssertionError("a missing cell must block the exhaustive launch")

    jobs[-1]["heldout_output_json"] = jobs[0]["heldout_output_json"]
    try:
        queue.validate_exhaustive_campaign(jobs)
    except queue.ConfigError as error:
        assert "heldout_output_json values must be unique" in str(error)
    else:
        raise AssertionError("shared held-out outputs must block launch")


def test_heldout_split_must_resolve_to_the_canonical_repo_file(tmp_path):
    canonical = (
        tmp_path
        / "environment"
        / "benchmark_splits"
        / "gsm_symbolic_crane_proportional_49x49_seed123.json"
    )
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"train": [], "test": []}\n', encoding="utf-8")
    job = _job()
    job["heldout_split_file"] = str(canonical)

    queue.validate_heldout_split(job["cell_id"], job, tmp_path)

    wrong = tmp_path / "other" / canonical.name
    wrong.parent.mkdir()
    wrong.write_text(canonical.read_text(), encoding="utf-8")
    job["heldout_split_file"] = str(wrong)
    try:
        queue.validate_heldout_split(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "canonical heldout split" in str(error)
    else:
        raise AssertionError(
            "a same-named split outside the repo path must be rejected"
        )


def test_baseline_evidence_binds_counts_model_train_side_and_source_hash(tmp_path):
    split_file = (
        tmp_path
        / "environment"
        / "benchmark_splits"
        / "gsm_symbolic_crane_proportional_49x49_seed123.json"
    )
    split_file.parent.mkdir(parents=True)
    split_file.write_text('{"train": [], "test": []}\n', encoding="utf-8")
    raw_artifact = tmp_path / "raw-baseline.json"
    raw_artifact.write_text('{"accuracy": 0.4081632653}\n', encoding="utf-8")
    artifact = tmp_path / "baseline.json"
    job = _job()
    job["baseline_num_correct"] = 20
    job["baseline_num_examples"] = 49
    normalized = {
        "dataset": job["dataset"],
        "eval_model": job["eval_model"],
        "split_name": "train",
        "baseline_strategy": "crane",
        "num_correct": 20,
        "num_examples": 49,
        "split_file": str(split_file.relative_to(tmp_path)),
        "split_file_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "raw_source_artifact": str(raw_artifact),
        "raw_source_sha256": hashlib.sha256(raw_artifact.read_bytes()).hexdigest(),
    }
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    job["baseline_source"] = str(evidence)
    evidence.write_text(
        json.dumps(
            {
                "cells": {
                    job["cell_id"]: {
                        "dataset": job["dataset"],
                        "eval_model": job["eval_model"],
                        "split_name": "train",
                        "baseline_strategy": "crane",
                        "num_correct": 20,
                        "num_examples": 49,
                        "source_artifact": str(artifact),
                        "source_sha256": hashlib.sha256(
                            artifact.read_bytes()
                        ).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)

    payload = json.loads(evidence.read_text())
    normalized["baseline_strategy"] = "itergen"
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["baseline_strategy"] = "itergen"
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "wrong baseline strategy" in str(error)
    else:
        raise AssertionError("the wrong dataset comparator must block launch")

    normalized["baseline_strategy"] = "crane"
    wrong_split = tmp_path / "other-train.json"
    wrong_split.write_text('{"train": []}\n', encoding="utf-8")
    normalized["split_file"] = str(wrong_split.relative_to(tmp_path))
    normalized["split_file_sha256"] = hashlib.sha256(
        wrong_split.read_bytes()
    ).hexdigest()
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["baseline_strategy"] = "crane"
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "wrong canonical train split" in str(error)
    else:
        raise AssertionError("a different train split must block launch")

    normalized["split_file"] = str(split_file.relative_to(tmp_path))
    normalized["split_file_sha256"] = hashlib.sha256(
        split_file.read_bytes()
    ).hexdigest()
    normalized["num_correct"] = 19
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "source artifact does not support manifest measurement" in str(error)
    else:
        raise AssertionError("unsupported baseline counts in source must block launch")

    normalized["num_correct"] = 20
    artifact.write_text(json.dumps(normalized), encoding="utf-8")
    payload["cells"][job["cell_id"]]["source_sha256"] = hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    payload["cells"][job["cell_id"]]["num_correct"] = 19
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    try:
        queue.validate_baseline_evidence(job["cell_id"], job, tmp_path)
    except queue.ConfigError as error:
        assert "does not match manifest counts" in str(error)
    else:
        raise AssertionError(
            "baseline evidence with different counts must block launch"
        )


def test_choose_gpu_bundle_stays_inside_the_allowed_gpu_set():
    """--gpus 3,0 has to keep the queue off cards other people are using."""
    snapshots = {gpu: {"used_mib": 0, "total_mib": 48_000} for gpu in range(4)}
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}
    job = _job()
    job["gpu_mem_util"] = 0.4

    # Allow exactly the GPUs one cell needs, taken from the high end so a
    # bundle drawn from anywhere else stands out.
    allowed = tuple(range(4 - queue.POOLABLE_GPU_COUNT, 4))
    bundle = queue.choose_gpu_bundle(
        job, snapshots, reservations, baseline, allowed_gpus=allowed
    )
    assert bundle is not None
    assert set(bundle) == set(allowed)

    single = queue.choose_gpu_bundle(
        _job("smiles"), snapshots, reservations, baseline, allowed_gpus=(3,)
    )
    assert single == (3,)

    # One GPU short of what the cell needs: refuse, rather than quietly
    # borrowing a card outside the allowed set from whoever is on it.
    assert (
        queue.choose_gpu_bundle(
            job, snapshots, reservations, baseline, allowed_gpus=allowed[1:]
        )
        is None
    )

    # Left unrestricted it behaves exactly as before, so existing callers and
    # the no-flag launch path are unaffected.
    assert queue.choose_gpu_bundle(job, snapshots, reservations, baseline) == tuple(
        range(queue.POOLABLE_GPU_COUNT)
    )


def test_choose_gpu_bundle_ignores_allowed_gpus_the_machine_does_not_have():
    """A typo in --gpus must not silently widen the run onto every card."""
    snapshots = {gpu: {"used_mib": 0, "total_mib": 48_000} for gpu in range(4)}
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}

    bundle = queue.choose_gpu_bundle(
        _job(), snapshots, reservations, baseline, allowed_gpus=(8, 9)
    )

    assert bundle is None


def test_parse_gpu_list_reads_a_comma_separated_set():
    assert queue.parse_gpu_list("3,0") == (0, 3)
    assert queue.parse_gpu_list("2") == (2,)
    assert queue.parse_gpu_list(" 3 , 0 ") == (0, 3)


def test_parse_gpu_list_rejects_junk_rather_than_running_everywhere():
    for junk in ["", "  ", "3,3", "-1", "abc", "3,", "3,abc"]:
        try:
            queue.parse_gpu_list(junk)
        except queue.ConfigError:
            continue
        raise AssertionError(f"parse_gpu_list accepted {junk!r}")


def test_a_poolable_cell_gets_two_gpus_so_sharded_eval_can_form():
    """Spider and GSM evaluate statelessly, so they can split a batch across two
    workers. One GPU means the pool is a pool of one and the sharding is dead
    code. Three needs three near-empty cards, which never happens on a shared
    box -- two is what actually forms."""
    assert queue.required_gpu_count(_job(dataset="spider")) == 2
    assert queue.required_gpu_count(_job(dataset="gsm_symbolic")) == 2


def test_a_stateful_cell_still_gets_exactly_one_gpu():
    """SMILES carries state between examples, so it must not be split."""
    assert queue.required_gpu_count(_job(dataset="smiles")) == 1


def test_spider_decoding_cap_leaves_room_for_the_longest_gold_query():
    """Spider strategies treat eval_max_steps as a budget they carve a reserve
    out of, not as an output length limit -- the generated Dafny does
    `mainLimit := maxSteps - closeReserve`, and the largest reserve any attempt
    has chosen so far is 40. The longest gold query in the benchmark is 130
    tokens. So the cap has to clear 130 + 40, or the hardest examples become
    unreachable no matter how good the strategy is.

    200 was looser than it needed to be; a runaway draft burns the whole 200 on
    every one of 300 examples. 176 bounds that damage while still leaving 136
    usable steps after the worst-case reserve.
    """
    longest_gold_tokens = 130
    largest_observed_reserve = 40

    for cell, spec in queue.EXPECTED_CELLS.items():
        if spec["dataset"] != "spider":
            continue
        cap = spec["eval_max_steps"]
        assert cap == 176, f"{cell} caps at {cap}, expected 176"
        assert cap - largest_observed_reserve > longest_gold_tokens, (
            f"{cell} leaves only {cap - largest_observed_reserve} usable steps, "
            f"which cannot reach a {longest_gold_tokens}-token gold query"
        )


def test_saved_manifest_agrees_with_the_spider_decoding_cap():
    """The launcher reads the manifest JSON at runtime, not EXPECTED_CELLS, so
    a cap changed in only one of the two places would be silently ignored."""
    manifest = json.loads(
        Path("saved-results/2026-07-19-exhaustive-cold-queue-manifest.json").read_text()
    )
    spider_jobs = [j for j in manifest["jobs"] if j["dataset"] == "spider"]
    assert spider_jobs, "manifest has no spider jobs"
    for job in spider_jobs:
        assert (
            job["eval_max_steps"]
            == queue.EXPECTED_CELLS[job["cell_id"]]["eval_max_steps"]
        ), (
            f"{job['cell_id']}: manifest says {job['eval_max_steps']}, "
            f"code says {queue.EXPECTED_CELLS[job['cell_id']]['eval_max_steps']}"
        )


def test_saved_manifest_agrees_with_the_gsm_task_text():
    """The GSM task text lives in three places -- the code constant, the test
    constant above, and the saved manifest -- and the launcher refuses to start
    when any two of them disagree, for every cell, even cells the run excludes.

    They drifted by exactly one trailing newline once already, which is
    invisible on screen and stopped the whole campaign from launching. Compare
    the raw strings so the failure names the character instead of printing two
    walls of identical-looking text.
    """
    manifest = json.loads(
        Path("saved-results/2026-07-19-exhaustive-cold-queue-manifest.json").read_text()
    )
    gsm_jobs = [j for j in manifest["jobs"] if j["dataset"] == "gsm_symbolic"]
    assert gsm_jobs, "manifest has no gsm_symbolic jobs"
    for job in gsm_jobs:
        assert job["task"] == queue.GSM_TASK, (
            f"{job['cell_id']}: manifest task ends {job['task'][-40:]!r}, "
            f"code task ends {queue.GSM_TASK[-40:]!r}"
        )


def test_spider_qwen25_1p5b_two_workers_fit_the_26576_mib_free_memory_boundary():
    """The settled Qwen2.5-1.5B floor admits both GPUs at the exact boundary."""
    job = _job("spider")
    job["eval_model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    job["gpu_mem_util"] = 0.30
    job["memory_reservation_mib"] = queue.EXPECTED_RUNTIME_BY_MODEL[
        job["eval_model"]
    ]["memory_reservation_mib"]
    total = 40_960
    snapshots = {
        1: {"used_mib": total - 32_859, "free_mib": 32_859, "total_mib": total},
        2: {"used_mib": total - 40_432, "free_mib": 40_432, "total_mib": total},
    }
    baseline = {gpu: dict(snapshot) for gpu, snapshot in snapshots.items()}
    reservations = {gpu: {} for gpu in snapshots}

    assert job["eval_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert queue.EXPECTED_RUNTIME_BY_MODEL[job["eval_model"]][
        "memory_reservation_mib"
    ] == 12_288
    assert queue.eval_workers_per_gpu(job) == 2
    assert queue.required_gpu_count(job) == 2
    assert queue.synthesis_required_memory_mib(job, total) == 24_576
    assert tuple(sorted(queue.choose_gpu_bundle(job, snapshots, reservations, baseline))) == (1, 2)

    for low_gpu in snapshots:
        below_boundary = {
            gpu: dict(snapshot) for gpu, snapshot in snapshots.items()
        }
        below_boundary[low_gpu]["free_mib"] = 26_575
        below_boundary[low_gpu]["used_mib"] = total - 26_575
        below_baseline = {
            gpu: dict(snapshot) for gpu, snapshot in below_boundary.items()
        }
        assert (
            queue.choose_gpu_bundle(job, below_boundary, reservations, below_baseline)
            is None
        ), f"GPU {low_gpu} below 26576 MiB was incorrectly admitted"
