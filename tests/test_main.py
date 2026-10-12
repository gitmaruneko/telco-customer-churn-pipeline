from pathlib import Path

import pytest

import main as application
from telco_churn import train


def test_root_main_delegates_arguments(monkeypatch) -> None:
    received_arguments = None

    def fake_run_cli(arguments) -> None:
        nonlocal received_arguments
        received_arguments = arguments

    monkeypatch.setattr(application, "run_cli", fake_run_cli)

    application.main(["--input", "customers.csv"])

    assert received_arguments == ["--input", "customers.csv"]


def test_train_main_passes_input_and_output_paths_to_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "customers.csv"
    output_dir = tmp_path / "results"
    received_paths = None

    def fake_run_pipeline(path, output_path):
        nonlocal received_paths
        received_paths = (path, output_path)

    monkeypatch.setattr(train, "run_pipeline", fake_run_pipeline)

    train.main(["--input", str(input_path), "--output", str(output_dir)])

    assert received_paths == (input_path, output_dir)


def test_train_main_uses_default_output_directory(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "customers.csv"
    received_output = None

    def fake_run_pipeline(path, output_dir):
        nonlocal received_output
        received_output = output_dir

    monkeypatch.setattr(train, "run_pipeline", fake_run_pipeline)

    train.main(["--input", str(input_path)])

    assert received_output == train.DEFAULT_OUTPUT_DIR


def test_train_main_reports_missing_input_without_success(monkeypatch, capsys) -> None:
    def fake_run_pipeline(path, output_dir):
        raise FileNotFoundError(path)

    monkeypatch.setattr(train, "run_pipeline", fake_run_pipeline)

    with pytest.raises(SystemExit) as error:
        train.main(["--input", "missing.csv"])

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert "Pipeline failed: missing.csv" in captured.err
    assert "Pipeline completed successfully." not in captured.out
