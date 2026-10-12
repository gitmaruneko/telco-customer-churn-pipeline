from pathlib import Path

import pytest

from telco_churn.config import PROJECT_ROOT, load_model_settings


def write_config(path: Path, *, test_size: float = 0.25, max_iter: int = 250) -> None:
    path.write_text(
        f"""
[split]
test_size = {test_size}
random_state = 7

[models.logistic_regression]
max_iter = {max_iter}

[models.decision_tree]
random_state = 11

[output]
directory = "custom-output"
predictions_filename = "customer_predictions.csv"
comparison_filename = "metrics.csv"
""".strip(),
        encoding="utf-8",
    )


def test_load_model_settings_reads_typed_values(tmp_path: Path) -> None:
    config_path = tmp_path / "model.toml"
    write_config(config_path)

    settings = load_model_settings(config_path)

    assert settings.test_size == 0.25
    assert settings.split_random_state == 7
    assert settings.logistic_regression_max_iter == 250
    assert settings.decision_tree_random_state == 11
    assert settings.output_directory == PROJECT_ROOT / "custom-output"
    assert settings.predictions_filename == "customer_predictions.csv"
    assert settings.comparison_filename == "metrics.csv"


@pytest.mark.parametrize(
    ("test_size", "max_iter", "message"),
    [
        (0, 250, "split.test_size must be between 0 and 1"),
        (0.25, 0, "models.logistic_regression.max_iter must be positive"),
    ],
)
def test_load_model_settings_rejects_invalid_values(
    tmp_path: Path,
    test_size: float,
    max_iter: int,
    message: str,
) -> None:
    config_path = tmp_path / "model.toml"
    write_config(config_path, test_size=test_size, max_iter=max_iter)

    with pytest.raises(ValueError, match=message):
        load_model_settings(config_path)