import pandas as pd

from telco_churn.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_model_settings,
)
from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.train import build_argument_parser, split_train_test


def test_model_settings_load_from_project_configuration() -> None:
    settings = load_model_settings()

    assert DEFAULT_CONFIG_PATH.is_file()
    assert settings.test_size == 0.20
    assert settings.split_random_state == 42
    assert settings.logistic_regression_max_iter == 1000
    assert settings.decision_tree_random_state == 42
    assert settings.dataset_path == PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
    assert settings.output_directory == PROJECT_ROOT / "output"
    assert settings.predictions_filename == "predictions.csv"
    assert settings.comparison_filename == "model_comparison.csv"


def test_model_settings_resolve_configured_output_directory(tmp_path) -> None:
    config_path = tmp_path / "model.toml"
    config_path.write_text(
        """
[split]
test_size = 0.25
random_state = 13

[models.logistic_regression]
max_iter = 500

[models.decision_tree]
random_state = 17

[input]
dataset_path = "data/custom.csv"

[output]
directory = "reports"
predictions_filename = "test_predictions.csv"
comparison_filename = "metrics.csv"
""".lstrip(),
        encoding="utf-8",
    )

    settings = load_model_settings(config_path)

    assert settings.test_size == 0.25
    assert settings.split_random_state == 13
    assert settings.logistic_regression_max_iter == 500
    assert settings.decision_tree_random_state == 17
    assert settings.dataset_path == PROJECT_ROOT / "data" / "custom.csv"
    assert settings.output_directory == PROJECT_ROOT / "reports"
    assert settings.predictions_filename == "test_predictions.csv"
    assert settings.comparison_filename == "metrics.csv"


def test_argument_parser_uses_settings_defaults_and_accepts_path_overrides(
    tmp_path,
) -> None:
    settings = load_model_settings()
    parser = build_argument_parser(settings)

    defaults = parser.parse_args([])
    overrides = parser.parse_args(
        [
            "--input",
            str(tmp_path / "customers.csv"),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert defaults.input == settings.dataset_path
    assert defaults.output_dir == settings.output_directory
    assert overrides.input == tmp_path / "customers.csv"
    assert overrides.output_dir == tmp_path / "results"


def test_train_test_split_is_reproducible_and_stratified() -> None:
    features = pd.DataFrame({"value": range(20)})
    target = pd.Series([0] * 10 + [1] * 10)

    first_split = split_train_test(features, target)
    second_split = split_train_test(features, target)

    for first, second in zip(first_split, second_split, strict=True):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)

    _, X_test, _, y_test = first_split
    assert len(X_test) == 4
    assert y_test.value_counts().to_dict() == {0: 2, 1: 2}


def test_both_model_builders_fit_and_predict() -> None:
    features = pd.DataFrame(
        {
            "service": ["A", "B"] * 6,
            "tenure": [1, 12, 3, 24, 5, 36, 7, 48, 9, 60, 11, 72],
        }
    )
    target = pd.Series([0, 1] * 6)

    logistic_model = build_logistic_regression_model(features, max_iter=2000)
    tree_model = build_decision_tree_model(features, random_state=7)

    assert logistic_model.named_steps["classifier"].max_iter == 2000
    assert tree_model.named_steps["classifier"].random_state == 7

    for model in (logistic_model, tree_model):
        model.fit(features, target)

        assert model.predict(features).shape == target.shape
        assert model.predict_proba(features).shape == (len(features), 2)
