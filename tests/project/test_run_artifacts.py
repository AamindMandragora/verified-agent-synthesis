from pathlib import Path

import pytest

from evaluation.common.run_artifacts import find_compiled_module_dir, resolve_run_dir


def test_resolve_run_dir_reads_latest_run_pointer(tmp_path: Path):
    actual_run = tmp_path / "runs" / "20260414_123456"
    actual_run.mkdir(parents=True)
    latest_txt = actual_run.parent / "latest_run.txt"
    latest_txt.write_text(str(actual_run))

    assert resolve_run_dir(actual_run.parent / "latest") == actual_run


def test_find_compiled_module_dir_prefers_named_subdirectories(tmp_path: Path):
    run_dir = tmp_path / "runs" / "example"
    module_dir = run_dir / "gsm_crane_csd"
    module_dir.mkdir(parents=True)
    (module_dir / "GeneratedCSD.py").write_text("# compiled")

    assert find_compiled_module_dir(run_dir) == module_dir


def test_find_compiled_module_dir_raises_for_missing_artifacts(tmp_path: Path):
    run_dir = tmp_path / "runs" / "missing"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        find_compiled_module_dir(run_dir)


def test_resolve_run_dir_rewrites_legacy_generated_csd_layout(tmp_path: Path):
    actual_run = tmp_path / "outputs" / "20260414_123456"
    actual_run.mkdir(parents=True)

    legacy_run = tmp_path / "outputs" / "generated-csd" / "runs" / "20260414_123456"
    assert resolve_run_dir(legacy_run) == actual_run
