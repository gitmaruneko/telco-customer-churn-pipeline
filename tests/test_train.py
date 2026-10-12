import pandas as pd

from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.train import (
    DEFAULT_INPUT_PATH,
    build_predictions_output,
    run_pipeline,
    split_train_test,
)


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

    for build_model in (build_logistic_regression_model, build_decision_tree_model):
        model = build_model(features)
        model.fit(features, target)
        predictions = model.predict(features)

        assert predictions.shape == target.shape
        assert set(predictions).issubset({0, 1})
        assert model.predict_proba(features).shape == (len(features), 2)


def test_prediction_output_contains_required_columns() -> None:
    predictions = build_predictions_output(
        customer_ids=pd.Series(["0001-AAAAA", "0002-BBBBB"]),
        actual_churn=pd.Series([0, 1]),
        predictions_by_model={
            "Logistic Regression": [0, 1],
            "Decision Tree": [0, 1],
        },
        probabilities_by_model={
            "Logistic Regression": [0.1, 0.9],
            "Decision Tree": [0.2, 0.8],
        },
    )

    assert predictions.columns.tolist() == [
        "customerID",
        "actual_churn",
        "predicted_churn",
        "churn_probability",
        "decision_tree_predicted_churn",
        "decision_tree_churn_probability",
    ]


def test_run_pipeline_reports_progress_and_writes_outputs(tmp_path, capsys) -> None:
    metrics, predictions = run_pipeline(DEFAULT_INPUT_PATH, tmp_path)

    output = capsys.readouterr().out
    steps = [
        "[1] Loading data...",
        "[2] Validating data...",
        "[3] Cleaning data...",
        "[4] Training models...",
        "[5] Evaluating models...",
        "[6] Writing predictions...",
    ]

    assert [output.index(step) for step in steps] == sorted(output.index(step) for step in steps)
    assert "Rows loaded: 7043" in output
    assert "Pipeline completed successfully." in output
    assert metrics["Model"].tolist() == ["Logistic Regression", "Decision Tree"]
    assert set(predictions["predicted_churn"]).issubset({0, 1})
    assert set(predictions["decision_tree_predicted_churn"]).issubset({0, 1})
    assert (tmp_path / "model_comparison.csv").is_file()
    assert (tmp_path / "predictions.csv").is_file()
