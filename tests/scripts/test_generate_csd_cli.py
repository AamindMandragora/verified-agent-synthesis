from pathlib import Path

from synthesis.cli.generate_csd import PROJECT_ROOT, build_synthesis_command


def test_build_synthesis_command_uses_dataset_preset_defaults():
    command = build_synthesis_command(
        dataset="gsm_symbolic",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    )

    assert command[0]
    assert command[1] == str(PROJECT_ROOT / "run_synthesis.py")
    assert "--dataset" in command
    assert command[command.index("--dataset") + 1] == "gsm_symbolic"
    assert command[command.index("--output-name") + 1] == "gsm_crane_csd"
    assert command[command.index("--model") + 1] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert command[command.index("--min-accuracy") + 1] == "0.5"
    assert command[command.index("--min-format-rate") + 1] == "1.0"
    assert command[command.index("--min-syntax-rate") + 1] == "1.0"


def test_build_synthesis_command_accepts_override_thresholds():
    command = build_synthesis_command(
        dataset="gsm_symbolic",
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        min_accuracy=0.6,
        min_format_rate=0.9,
        min_syntax_rate=0.95,
        eval_sample_size=3,
        eval_max_steps=512,
    )

    assert command[command.index("--min-accuracy") + 1] == "0.6"
    assert command[command.index("--min-format-rate") + 1] == "0.9"
    assert command[command.index("--min-syntax-rate") + 1] == "0.95"
    assert command[command.index("--eval-sample-size") + 1] == "3"
    assert command[command.index("--eval-max-steps") + 1] == "512"


def test_shell_preset_wrappers_exist_for_supported_model_dataset_pairs():
    wrappers = {
        "folio_qwen3b.sh": ("folio", "qwen3b"),
        "folio_qwen7b.sh": ("folio", "qwen7b"),
        "gsm_symbolic_qwen3b.sh": ("gsm_symbolic", "qwen3b"),
        "gsm_symbolic_qwen7b.sh": ("gsm_symbolic", "qwen7b"),
        "pddl_qwen3b.sh": ("pddl", "qwen3b"),
        "pddl_qwen7b.sh": ("pddl", "qwen7b"),
        "sygus_slia_qwen3b.sh": ("sygus_slia", "qwen3b"),
        "sygus_slia_qwen7b.sh": ("sygus_slia", "qwen7b"),
    }

    preset_dir = Path("synthesis/shell")
    actual = {path.name for path in preset_dir.glob("*.sh")}
    assert set(wrappers).issubset(actual)

    for filename, (dataset, model_preset) in wrappers.items():
        script = (preset_dir / filename).read_text(encoding="utf-8")
        assert "python -m synthesis.cli.generate_csd" in script
        assert f"generate_csd {dataset}" in script
        assert f"--model-preset {model_preset}" in script
        assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"' in script
