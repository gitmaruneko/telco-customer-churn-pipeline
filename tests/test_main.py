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


def test_train_main_passes_input_path_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    input_path = tmp_path / "customers.csv"
    received_input = None

    def fake_run_pipeline(path, output_dir=train.DEFAULT_OUTPUT_DIR):
        nonlocal received_input
        received_input = path

    monkeypatch.setattr(train, "run_pipeline", fake_run_pipeline)

    train.main(["--input", str(input_path)])

    assert received_input == input_path


def test_train_main_reports_missing_input_without_success(monkeypatch, capsys) -> None:
    def fake_run_pipeline(path, output_dir=train.DEFAULT_OUTPUT_DIR):
        raise FileNotFoundError(path)

    monkeypatch.setattr(train, "run_pipeline", fake_run_pipeline)

    with pytest.raises(SystemExit) as error:
        train.main(["--input", "missing.csv"])

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert "Pipeline failed: missing.csv" in captured.err
    assert "Pipeline completed successfully." not in captured.out
