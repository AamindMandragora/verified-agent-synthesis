from pathlib import Path
import re
import subprocess
import sys

import pytest

from synthesis.evaluate.feedback_loop import SynthesisPipeline


def _dummy_pipeline(**kwargs):
    return SynthesisPipeline(
        evaluator=object(),
        generator=object(),
        verifier=object(),
        compiler=object(),
        **kwargs,
    )


def test_pipeline_defaults_use_ucb_budget_and_refinement_beam():
    pipeline = _dummy_pipeline()

    assert pipeline.helper_selection_policy == "bandit"
    assert pipeline.eval_max_seconds_per_example == 90.0
    assert pipeline.min_examples_before_threshold_stop == 15
    assert pipeline.refinement_beam_size == 2


class _UnloadCountingEvaluator:
    def __init__(self):
        self.unload_calls = 0

    def unload_runtime(self):
        self.unload_calls += 1


class _BackendGenerator:
    def __init__(self, backend):
        self.backend = backend


def test_pipeline_keeps_evaluator_runtime_warm_for_hosted_author_models():
    evaluator = _UnloadCountingEvaluator()
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=_BackendGenerator("anthropic"),
        verifier=object(),
        compiler=object(),
    )

    pipeline._unload_evaluator_runtime_before_refinement()

    assert evaluator.unload_calls == 0


def test_pipeline_keeps_evaluator_runtime_warm_for_claude_code():
    evaluator = _UnloadCountingEvaluator()
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=_BackendGenerator("claude"),
        verifier=object(),
        compiler=object(),
    )

    pipeline._unload_evaluator_runtime_before_refinement()

    assert evaluator.unload_calls == 0


def test_pipeline_unloads_evaluator_runtime_for_local_author_models():
    evaluator = _UnloadCountingEvaluator()
    pipeline = SynthesisPipeline(
        evaluator=evaluator,
        generator=_BackendGenerator("vllm"),
        verifier=object(),
        compiler=object(),
    )

    pipeline._unload_evaluator_runtime_before_refinement()

    assert evaluator.unload_calls == 1


def test_run_synthesis_help_advertises_ucb_budget_and_beam_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "synthesis.run_synthesis", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    help_text = result.stdout
    # Surviving science flags keep their documented defaults.
    assert re.search(r"default:\s+600", help_text)
    assert re.search(r"default:\s+90", help_text)
    assert "claude" in help_text
    assert "claude-bedrock" in help_text
    assert "anthropic" in help_text
    assert "direct Anthropic API" in " ".join(help_text.split())
    # New provider-agnostic thinking budget replaced the anthropic-* flags.
    assert "--synthesizer-reasoning-budget" in help_text
    assert "Default: 4096." in " ".join(help_text.split())
    # The single fair helper-selection contract is intentionally public again.
    assert "--helper-selection-policy {bandit}" in " ".join(help_text.split())
    assert "--eval-min-examples-before-threshold-stop" in help_text
    # 2026-07-17 simplification: these knobs are constants/env now, not flags.
    for gone in (
        "--claude-config-dir",
        "--claude-expected-account",
        "--restart-after-stuck-iters",
        "--anthropic-thinking",
        "--output-name",
        "--temperature",
        "--eval-backend",
        "--allow-small-author-model",
        "--require-delimiters",
        "--gsm-split-file",
        "--spider-split-file",
    ):
        assert gone not in help_text, gone


def test_run_synthesis_help_names_the_fixed_claude_code_model():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "synthesis.run_synthesis", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "claude-opus-5" in " ".join(result.stdout.split())


def test_root_run_synthesis_keeps_the_fixed_claude_code_model():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "run_synthesis.py").read_text()

    assert "Claude Code uses the fixed claude-opus-5 model" in source
    assert 'args.generation_model = "claude-opus-5"' in source


def test_run_synthesis_resolves_author_model_before_printing_banner():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "synthesis" / "run_synthesis.py").read_text()

    assert source.index("if args.generation_model is None:") < source.index(
        'f"  AUTHOR MODEL : {args.generation_model!r} "'
    )


def test_launchers_pin_ucb_budget_and_beam_flags():
    repo_root = Path(__file__).resolve().parents[1]
    launchers = [
        repo_root / "run_ab_thinking.sh",
        repo_root / "scripts" / "launch_ab_norestart_20260522.sh",
        repo_root / "scripts" / "launch_thinking_effort_high_logio_20260522.sh",
    ]

    for launcher in launchers:
        text = launcher.read_text()
        assert "--helper-selection-policy bandit" in text
        assert "--refinement-beam-size 2" in text
        assert "--eval-max-steps 600" in text
        assert "--eval-max-seconds-per-example 90" in text
        assert "--eval-min-examples-before-threshold-stop 15" in text


def test_full_test_runner_uses_ucb_and_refinement_beam_two():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "run_all_tests.py").read_text()

    assert 'benchmark, ablation_model, "2", mask_flag, "bandit", smiles_class' in text
    assert "utility" not in text


def test_full_test_runner_defaults_main_metadecode_to_40_iterations(tmp_path):
    import run_all_tests as matrix

    assert matrix.DEFAULT_MAIN_SYNTH_ITERS == "40"
    assert "900" in matrix.csv_list(matrix.DEFAULT_STEP_BUDGETS)

    runner = _matrix_runner(tmp_path)
    runner.config.synth_iters = matrix.csv_list(matrix.DEFAULT_SYNTH_ITERS)
    runner.config.models = ["Qwen/Qwen2.5-1.5B-Instruct"]
    runner.config.benchmarks = ["gsm"]
    runner.config.strategies = ["metadecode"]
    runner.config.token_budgets = ["1"]
    runner.config.gen_models = ["opus4.7"]
    calls = []

    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append((args, kwargs))

    runner.run_main_matrix()

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[3] == "40"
    assert args[5] == "900"
    assert kwargs["phase"] == "main_matrix"


def test_full_test_runner_uses_gsm_max_steps_900_without_changing_spider(tmp_path):
    runner = _matrix_runner(tmp_path)

    assert runner.eval_max_steps_for("gsm_symbolic") == "900"
    assert runner.eval_max_steps_for("spider") == "600"


def _matrix_runner(tmp_path, *, dry_run=True):
    import run_all_tests as matrix

    gsm_split = tmp_path / "gsm_split.json"
    gsm_split.write_text('{"train_indices": [1], "eval_indices": [2]}\n')
    spider_split = tmp_path / "spider_split.json"
    spider_split.write_text('{"train_indices": [1], "eval_indices": [2]}\n')

    config = matrix.Config(
        models=["Qwen/Qwen2.5-Coder-7B-Instruct"],
        benchmarks=["gsm", "spider"],
        strategies=["metadecode"],
        token_budgets=["1", "2", "4"],
        synth_iters=["3", "5", "10", "30"],
        gen_models=["opus4.7", "gpt5.5", "gemini"],
        step_budgets=["256", "512", "900", "1024"],
        smiles_classes=["acrylates", "chain_extenders"],
        eval_backend="vllm",
        device="auto",
        generation_sample_size="52",
        eval_sample_size="100",
        gsm_generation_sample_size="51",
        gsm_eval_sample_size="50",
        eval_max_steps="600",
        eval_max_steps_gsm="900",
        eval_max_steps_smiles="400",
        eval_max_seconds_per_example="90",
        accuracy_win_margin=0.03,
        synthesis_max_tokens="32768",
        vllm_gpu_memory_utilization="0.80",
        vllm_tensor_parallel_size=1,
        dafny_path="/tmp/dafny",
        generated_output_dir=tmp_path / "generated",
        baseline_output_dir=tmp_path / "baselines",
        ablation_output_dir=tmp_path / "ablations",
        baseline_cache_mode="reuse",
        gsm_split_file=str(gsm_split),
        spider_split_file=str(spider_split),
        dry_run=dry_run,
        skip_main=False,
        skip_ablations=False,
        ablation_sections=set(matrix.VALID_ABLATION_SECTIONS),
        conda_env_path=tmp_path / "conda",
        cuda_devices="0",
        cuda_oom_fallback="",
        free_gpu_max_used_mb=1024,
        gpu_wait_seconds=1,
        gpu_wait_timeout_seconds=0,
        main_synthesis_iterations="40",
        gpu3_retry_queue=tmp_path / "gpu3_retry_queue.jsonl",
        gpu3_retry_enabled=True,
    )
    return matrix.Runner(
        config=config,
        env={
            "ANTHROPIC_OPUS_MODEL": "claude-opus-4-7",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_GENERATION_MODEL": "gpt-5.5",
        },
    )


def test_full_test_runner_rejects_bedrock_generation_profiles(tmp_path):
    runner = _matrix_runner(tmp_path)

    for profile in ("bedrock", "bedrock:anthropic.claude-3-5-sonnet", "gemini-pro"):
        with pytest.raises(ValueError, match="Bedrock"):
            runner.resolve_gen_profile(profile)

    with pytest.raises(ValueError, match="Unknown generation profile"):
        runner.resolve_gen_profile("anthropic.claude-3-5-sonnet-20241022-v2:0")


def test_sonnet_profile_migrates_to_claude_with_explicit_old_routes(tmp_path):
    runner = _matrix_runner(tmp_path)

    with pytest.warns(FutureWarning, match="sonnet4.6"):
        assert runner.resolve_gen_profile("sonnet4.6") == (
            "claude",
            "claude-opus-5",
        )
    assert runner.resolve_gen_profile("anthropic-sonnet4.6") == (
        "anthropic",
        "claude-sonnet-4-6",
    )
    with pytest.raises(ValueError, match="Bedrock"):
        runner.resolve_gen_profile("claude-bedrock-sonnet4.6")


# NOTE (2026-07-18): test_sonnet_profile_forwards_only_claude_code_controls was
# removed here. It checked that --claude-executable/--claude-config-dir/
# --claude-expected-account/--claude-timeout-seconds were forwarded into the
# run_synthesis command. Those flags and their Config fields were deleted as
# dead code in commit fabd206d; the backend/model migration behavior it also
# touched on is already covered by test_sonnet_profile_migrates_to_claude_with_explicit_old_routes.


def test_claude_access_marker_is_classified_as_author_quota():
    import run_all_tests as matrix

    assert matrix.QUOTA_RE.search(
        "[claude-author-access] Claude Code subscription limit reached"
    )


def test_full_test_runner_resolves_direct_gemini_profile(tmp_path):
    runner = _matrix_runner(tmp_path)
    runner.env["GEMINI_GENERATION_MODEL"] = "gemini-3-pro-preview"

    assert runner.resolve_gen_profile("gemini") == ("gemini", "gemini-3-pro-preview")


def test_full_test_runner_resolves_gpt56_sol_through_codex(tmp_path):
    runner = _matrix_runner(tmp_path)

    assert runner.resolve_gen_profile("gpt5.6-sol") == ("codex", "gpt-5.6-sol")


def test_openai_generation_profile_is_skipped_when_api_key_is_missing(tmp_path):
    runner = _matrix_runner(tmp_path)
    runner.env.pop("OPENAI_API_KEY", None)
    calls = []

    runner.ensure_csd_target_baselines = lambda *args: calls.append("baseline")
    runner.run_cmd = lambda *args, **kwargs: calls.append("run") or True

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "30",
        "gpt5.5",
        "600",
    )

    assert calls == []


def test_openai_generation_profile_quota_failure_does_not_abort_matrix(tmp_path):
    runner = _matrix_runner(tmp_path)
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    calls = []

    def fake_run_cmd(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return False

    runner.run_cmd = fake_run_cmd

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "30",
        "gpt5.5",
        "600",
    )

    assert len(calls) == 1
    assert calls[0][1]["abort_on_quota"] is False


def test_sonnet_generation_profile_quota_failure_is_skipped_not_requeued(tmp_path):
    runner = _matrix_runner(tmp_path, dry_run=False)
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    calls = []

    def fake_run_cmd(cmd, **kwargs):
        calls.append((cmd, kwargs))
        runner.last_failure_was_author_access = True
        return False

    runner.run_cmd = fake_run_cmd

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "40",
        "sonnet4.6",
        "900",
        phase="ablation_synthesizer_model",
    )

    assert len(calls) == 1
    assert calls[0][1]["abort_on_quota"] is False
    assert "--generation-model" in calls[0][0]
    assert "claude-opus-5" in calls[0][0]
    assert not runner.config.gpu3_retry_queue.exists()


def test_metadecode_failure_on_non_gpu3_is_queued_for_gpu3_retry(tmp_path):
    import run_all_tests as matrix

    runner = _matrix_runner(tmp_path, dry_run=False)
    runner.env["CUDA_VISIBLE_DEVICES"] = "1"
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: False

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "40",
        "opus4.7",
        "600",
        phase="main_matrix",
    )

    queue_lines = runner.config.gpu3_retry_queue.read_text().splitlines()
    assert len(queue_lines) == 1
    record = matrix.json.loads(queue_lines[0])
    assert record["reason"] == "synthesis_subprocess_failed"
    assert record["case"]["phase"] == "main_matrix"
    assert record["case"]["synth_iter"] == "40"
    assert record["cmd"][:3] == ["python", "-m", "synthesis.run_synthesis"]


def test_metadecode_failure_on_gpu3_is_not_requeued(tmp_path):
    runner = _matrix_runner(tmp_path, dry_run=False)
    runner.env["CUDA_VISIBLE_DEVICES"] = "3"
    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: False

    assert runner.run_metadecode_case(
        "gsm_symbolic",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "40",
        "opus4.7",
        "600",
        phase="main_matrix",
    )

    assert not runner.config.gpu3_retry_queue.exists()


def test_full_test_runner_ablation_c_includes_gemini_profile(tmp_path):
    import run_all_tests as matrix

    assert "gemini" in matrix.csv_list(matrix.DEFAULT_GEN_MODELS)

    runner = _matrix_runner(tmp_path)
    runner.config.benchmarks = ["gsm"]
    runner.config.step_budgets = []
    runner.config.token_budgets = ["1"]
    runner.config.synth_iters = ["10"]
    runner.config.gen_models = ["opus4.7", "gpt5.5", "gemini"]
    calls = []

    runner.run_fixed_strategy_cases = lambda *args, **kwargs: None
    runner.run_ablation_e_case = lambda *args, **kwargs: None
    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append(args)

    runner.run_ablations()

    assert any(call[4] == "gemini" for call in calls)


def test_ablation_sections_cli_normalizes_and_rejects_unknown():
    import run_all_tests as matrix

    assert matrix.normalize_ablation_sections("c,e") == {"C", "E"}
    assert matrix.normalize_ablation_sections(" A, b ") == {"A", "B"}

    with pytest.raises(SystemExit, match="Invalid ablation section"):
        matrix.normalize_ablation_sections("A,Z")

    with pytest.raises(SystemExit, match="At least one ablation section"):
        matrix.normalize_ablation_sections("")


def test_full_test_runner_filters_ablation_sections(tmp_path):
    runner = _matrix_runner(tmp_path)
    runner.config.benchmarks = ["gsm"]
    runner.config.step_budgets = ["256"]
    runner.config.token_budgets = ["1", "2"]
    runner.config.synth_iters = ["3", "30"]
    runner.config.gen_models = ["opus4.7", "gemini"]
    runner.config.ablation_sections = {"C"}
    calls = []

    runner.run_fixed_strategy_cases = lambda *args, **kwargs: calls.append(("fixed", args, kwargs))
    runner.run_ablation_e_case = lambda *args, **kwargs: calls.append(("helper_mask", args, kwargs))
    runner.run_metadecode_cases = lambda *args, **kwargs: calls.append(("metadecode", args, kwargs))

    runner.run_ablations()

    assert [(kind, kwargs["phase"]) for kind, _args, kwargs in calls] == [
        ("metadecode", "ablation_synthesizer_model"),
        ("metadecode", "ablation_synthesizer_model"),
    ]
    assert [args[4] for _kind, args, _kwargs in calls] == ["opus4.7", "gemini"]


def test_matrix_result_json_annotation_records_ablation_provenance(tmp_path):
    import run_all_tests as matrix

    runner = _matrix_runner(tmp_path, dry_run=False)
    result_json = tmp_path / "result.json"
    result_json.write_text('{"accuracy": 0.5, "syntax_rate": 0.9, "answers": []}\n')

    runner.annotate_result_json(
        result_json,
        runner.matrix_case_metadata(
            phase="ablation_synthesizer_model",
            strategy="metadecode",
            benchmark="spider",
            eval_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            token_budget="1",
            max_steps="600",
            command=["python", "-m", "synthesis.run_synthesis"],
            synth_iter="10",
            gen_profile="gemini",
            generation_backend="gemini",
            generation_model="gemini-3-pro-preview",
            required_accuracy=0.53,
            required_syntax=0.9,
            target_accuracy_strategy="crane",
            target_accuracy_path="/tmp/crane.json",
            target_syntax_strategy="itergen",
            target_syntax_path="/tmp/itergen.json",
        ),
    )

    payload = matrix.json.loads(result_json.read_text())
    metadata = payload["matrix_metadata"]
    assert metadata["phase"] == "ablation_synthesizer_model"
    assert metadata["strategy"] == "metadecode"
    assert metadata["generation_profile"] == "gemini"
    assert metadata["generation_backend"] == "gemini"
    assert metadata["generation_model"] == "gemini-3-pro-preview"
    assert metadata["synthesis_iterations"] == 10
    assert metadata["eval_max_steps"] == 600
    assert metadata["eval_step_token_budget"] == 1
    assert metadata["thresholds"]["min_accuracy"] == 0.53
    assert metadata["thresholds"]["min_syntax_rate"] == 0.9
    assert metadata["synthesis_controls"]["max_tokens"] == 32768
    assert metadata["splits"]["spider"]["split_name"] == "eval"
    assert metadata["command"] == "python -m synthesis.run_synthesis"


def _flag_value(cmd, flag):
    assert flag in cmd, f"missing {flag} in {cmd}"
    index = cmd.index(flag)
    assert index + 1 < len(cmd), f"{flag} has no value in {cmd}"
    return cmd[index + 1]


def _assert_flag_values(cmd, expected):
    for flag, value in expected.items():
        assert _flag_value(cmd, flag) == str(value)


# 2026-07-18: dropped the `dataset_flags` case (--gsm-split-file/--spider-split-file,
# via add_generation_split_flags) plus --output-name, --output-dir,
# --eval-backend, --adaptive-helper-mask, --vllm-gpu-memory-utilization,
# --vllm-tensor-parallel-size, and the anthropic-*/restart/beam/helper-policy
# flags below. run_metadecode_case no longer forwards any of these (dead flags
# removed in commit fabd206d); the settled values now live hard-coded inside
# synthesis.run_synthesis, not as CLI knobs this launcher controls.
@pytest.mark.parametrize(
    ("benchmark", "smiles_class", "max_steps", "sample_size", "expected_task"),
    [
        (
            "gsm_symbolic",
            "",
            "900",
            "51",
            "Solve math word problems step by step, wrapping intermediate symbolic expressions and the final answer inside << >> delimiters.",
        ),
        (
            "spider",
            "",
            "600",
            "52",
            "Generate a single valid SQL query using only the provided schema context. Only output the SQL query.",
        ),
    ],
)
def test_full_test_runner_metadecode_command_forwards_complete_launch_contract(
    tmp_path,
    benchmark,
    smiles_class,
    max_steps,
    sample_size,
    expected_task,
):
    runner = _matrix_runner(tmp_path)
    captured = []

    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.42,
        "crane",
        "/tmp/crane.json",
        "42.0%",
        0.88,
        "itergen",
        "/tmp/itergen.json",
        "88.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: captured.append(cmd) or True

    assert runner.run_metadecode_case(
        benchmark,
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "1",
        "10",
        "opus4.7",
        max_steps,
        smiles_class,
    )

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:3] == ["python", "-m", "synthesis.run_synthesis"]
    assert "--temperature" not in cmd
    _assert_flag_values(
        cmd,
        {
            "--task": expected_task,
            "--dataset": benchmark,
            "--generation-model": "claude-opus-4-7",
            "--generation-backend": "anthropic",
            "--eval-model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--max-iterations": "10",
            "--min-accuracy": "0.45",
            "--min-syntax-rate": "0.88",
            "--eval-sample-size": sample_size,
            "--eval-max-steps": max_steps,
            "--eval-step-token-budget": "1",
            "--eval-max-seconds-per-example": "90",
            "--max-tokens": "32768",
            "--device": "auto",
            "--dafny-path": "/tmp/dafny",
        },
    )


def test_full_test_runner_ablation_e_command_forwards_complete_launch_contract(tmp_path):
    runner = _matrix_runner(tmp_path)
    captured = []

    runner.ensure_csd_target_baselines = lambda *args: None
    runner.best_csd_baseline_targets = lambda *args: (
        0.5,
        "crane",
        "/tmp/crane.json",
        "50.0%",
        0.9,
        "itergen",
        "/tmp/itergen.json",
        "90.0%",
    )
    runner.run_cmd = lambda cmd, **kwargs: captured.append((cmd, kwargs)) or True

    runner.run_ablation_e_case(
        "spider",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "2",
        "--adaptive-helper-mask",
        "bandit",
    )

    assert len(captured) == 1
    cmd, run_kwargs = captured[0]
    assert run_kwargs["abort_on_quota"] is False
    assert cmd[:3] == ["python", "-m", "synthesis.run_synthesis"]
    assert "--temperature" not in cmd
    assert "--anthropic-thinking" not in cmd
    # 2026-07-18: dropped --eval-backend, --adaptive-helper-mask,
    # --eval-min-examples-before-threshold-stop, --restart-after-stuck-iters,
    # --refinement-beam-size, --helper-selection-policy,
    # --vllm-gpu-memory-utilization, --vllm-tensor-parallel-size, --output-dir,
    # and --spider-split-file/--spider-split-name — none of these are forwarded
    # by run_ablation_e_case anymore (dead flags removed in commit fabd206d).
    _assert_flag_values(
        cmd,
        {
            "--task": "Generate a single valid SQL query using only the provided schema context. Only output the SQL query.",
            "--dataset": "spider",
            "--generation-backend": "openai",
            "--generation-model": "gpt-5.5",
            "--eval-model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--max-iterations": "30",
            "--min-accuracy": "0.53",
            "--min-syntax-rate": "0.9",
            "--eval-sample-size": "52",
            "--eval-max-steps": "600",
            "--eval-step-token-budget": "1",
            "--eval-max-seconds-per-example": "90",
            "--max-tokens": "32768",
            "--device": "auto",
            "--dafny-path": "/tmp/dafny",
        },
    )


def test_full_test_runner_default_matrix_sections_cover_gsm_and_sql_not_cars_dataset(tmp_path):
    import run_all_tests as matrix

    assert matrix.DEFAULT_BENCHMARKS == "gsm,spider"

    runner = _matrix_runner(tmp_path)
    seen = []

    runner.config.step_budgets = ["256"]
    runner.config.token_budgets = ["1"]
    runner.config.synth_iters = ["3"]
    runner.config.gen_models = ["opus4.7"]
    runner.config.smiles_classes = ["acrylates"]
    runner.run_fixed_strategy_cases = lambda strategy, benchmark, *args, **kwargs: seen.append(
        ("fixed", strategy, benchmark)
    )
    runner.run_metadecode_cases = lambda benchmark, *args, **kwargs: seen.append(
        ("metadecode", benchmark)
    )
    runner.run_ablation_e_case = lambda benchmark, *args, **kwargs: seen.append(
        ("helper_mask", benchmark)
    )

    runner.run_main_matrix()
    runner.run_ablations()

    assert {entry[-1] for entry in seen} == {"gsm_symbolic", "spider"}
    assert not any(entry[-1] == "smiles" for entry in seen)
    assert any(entry == ("metadecode", "gsm_symbolic") for entry in seen)
    assert any(entry == ("metadecode", "spider") for entry in seen)


def test_full_test_runner_spider_task_names_require_bare_sql():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "run_all_tests.py").read_text()

    assert "Only output the SQL query." in text
    assert "SQL: <<YOUR QUERY>>" not in text
    assert "You may optionally reason" not in text
