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


def test_shell_preset_wrappers_exist_for_supported_model_dataset_pairs():
    wrappers = {
        "folio_qwen3b.sh",
        "folio_qwen7b.sh",
        "gsm_symbolic_qwen3b.sh",
        "gsm_symbolic_qwen7b.sh",
        "pddl_qwen3b.sh",
        "pddl_qwen7b.sh",
        "sygus_slia_qwen3b.sh",
        "sygus_slia_qwen7b.sh",
    }

    preset_dir = Path("synthesis/shell")
    actual = {path.name for path in preset_dir.glob("*.sh")}
    assert wrappers.issubset(actual)
