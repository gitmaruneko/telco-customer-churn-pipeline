from io import BytesIO

import pandas as pd

import telco_churn.data as data_module
import telco_churn.train as train_module
from telco_churn.config import ModelSettings
from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.train import (
    build_predictions_output,
    print_evaluation_report,
    run_pipeline,
    split_train_test,
)


def make_pipeline_data(rows: int = 20) -> pd.DataFrame:
    """Build a small valid labeled dataset for pipeline integration tests."""
    return pd.DataFrame(
        {
            "customerID": [f"customer-{index:03d}" for index in range(rows)],
            "gender": ["Female" if index % 2 == 0 else "Male" for index in range(rows)],
            "SeniorCitizen": [index % 2 for index in range(rows)],
            "Partner": ["Yes" if index % 2 == 0 else "No" for index in range(rows)],
            "Dependents": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
            "tenure": [index + 1 for index in range(rows)],
            "PhoneService": ["Yes"] * rows,
            "MultipleLines": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
            "InternetService": [
                "DSL" if index % 2 == 0 else "Fiber optic" for index in range(rows)
            ],
            "OnlineSecurity": ["Yes" if index % 2 == 0 else "No" for index in range(rows)],
            "OnlineBackup": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
            "DeviceProtection": ["Yes" if index % 2 == 0 else "No" for index in range(rows)],
            "TechSupport": ["Yes" if index % 2 == 0 else "No" for index in range(rows)],
            "StreamingTV": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
            "StreamingMovies": ["Yes" if index % 2 == 0 else "No" for index in range(rows)],
            "Contract": [
                "One year" if index % 2 == 0 else "Month-to-month" for index in range(rows)
            ],
            "PaperlessBilling": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
            "PaymentMethod": [
                "Mailed check" if index % 2 == 0 else "Electronic check" for index in range(rows)
            ],
            "MonthlyCharges": [50.0 + index for index in range(rows)],
            "TotalCharges": [str((index + 1) * 50.0) for index in range(rows)],
            "Churn": ["No" if index % 2 == 0 else "Yes" for index in range(rows)],
        }
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


def test_model_builders_use_configured_hyperparameters() -> None:
    features = pd.DataFrame({"tenure": [1, 2]})

    logistic_model = build_logistic_regression_model(features, max_iter=250)
    tree_model = build_decision_tree_model(features, random_state=7)

    assert logistic_model.named_steps["classifier"].max_iter == 250
    assert tree_model.named_steps["classifier"].random_state == 7


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


def test_evaluation_report_is_compact_and_preserves_metrics(capsys) -> None:
    metrics = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "Train Accuracy": 0.8056,
                "Test Accuracy": 0.8055,
                "Precision": 0.6572,
                "Recall": 0.5588,
                "F1": 0.6040,
            },
            {
                "Model": "Decision Tree",
                "Train Accuracy": 0.9980,
                "Test Accuracy": 0.7225,
                "Precision": 0.4781,
                "Recall": 0.4973,
                "F1": 0.4875,
            },
        ]
    )
    original_metrics = metrics.copy(deep=True)
    matrices = {
        "Logistic Regression": [[926, 109], [165, 209]],
        "Decision Tree": [[832, 203], [188, 186]],
    }

    print_evaluation_report(metrics, matrices)

    output = capsys.readouterr().out
    performance, confusion = output.split("Confusion counts (positive = churn)")
    for _, row in metrics.iterrows():
        line = next(line for line in performance.splitlines() if row["Model"] in line)
        assert line.split() == row["Model"].split() + [
            f"{row[column]:.2%}" for column in metrics.columns if column != "Model"
        ]
    assert "TN/TP: correct no churn/churn | FP/FN: false alarm/missed churn" in confusion
    header = next(line for line in confusion.splitlines() if "Model" in line)
    assert header.split() == ["Model", "TN", "FP", "FN", "TP"]
    for name, matrix in matrices.items():
        line = next(line for line in confusion.splitlines() if name in line)
        assert line.split() == name.split() + [str(value) for row in matrix for value in row]
    assert len(output.splitlines()) == 11
    pd.testing.assert_frame_equal(metrics, original_metrics)


def test_run_pipeline_reports_progress_and_writes_outputs(tmp_path, capsys) -> None:
    input_path = tmp_path / "customers.csv"
    output_dir = tmp_path / "output"
    make_pipeline_data().to_csv(input_path, index=False)

    settings = ModelSettings(
        test_size=0.20,
        split_random_state=7,
        logistic_regression_max_iter=250,
        decision_tree_random_state=11,
        output_directory=output_dir,
        predictions_filename="customer_predictions.csv",
        comparison_filename="metrics.csv",
    )

    metrics, predictions = run_pipeline(input_path, output_dir, settings)

    output = capsys.readouterr().out
    steps = [
        "Data: 20 rows",
        "Validation: OK",
        "Cleaning: OK",
        "Models: 2 trained",
        "Model performance (%)",
        "Confusion counts (positive = churn)",
        "Predictions :",
        "Comparison  :",
        "Pipeline completed successfully.",
    ]

    assert [output.index(step) for step in steps] == sorted(output.index(step) for step in steps)
    assert (
        "Data: 20 rows | Validation: OK | Cleaning: OK | Models: 2 trained" in output.splitlines()
    )
    assert f"Predictions : {output_dir / 'customer_predictions.csv'}" in output
    assert f"Comparison  : {output_dir / 'metrics.csv'}" in output
    assert metrics["Model"].tolist() == ["Logistic Regression", "Decision Tree"]
    assert set(predictions["predicted_churn"]).issubset({0, 1})
    assert set(predictions["decision_tree_predicted_churn"]).issubset({0, 1})
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "customer_predictions.csv").is_file()
    pd.testing.assert_frame_equal(pd.read_csv(output_dir / "metrics.csv"), metrics)
    pd.testing.assert_frame_equal(pd.read_csv(output_dir / "customer_predictions.csv"), predictions)


def test_run_pipeline_downloads_missing_default_dataset(tmp_path, capsys, monkeypatch) -> None:
    input_path = tmp_path / "data" / "raw" / "Telco-Customer-Churn.csv"
    output_dir = tmp_path / "output"
    payload = make_pipeline_data().to_csv(index=False).encode("utf-8")
    requested_urls = []

    def fake_urlopen(url: str, timeout: int):
        requested_urls.append((url, timeout))
        return BytesIO(payload)

    monkeypatch.setattr(train_module, "DEFAULT_INPUT_PATH", input_path)
    monkeypatch.setattr(data_module, "urlopen", fake_urlopen)

    settings = ModelSettings(
        test_size=0.20,
        split_random_state=7,
        logistic_regression_max_iter=250,
        decision_tree_random_state=11,
        output_directory=output_dir,
        predictions_filename="customer_predictions.csv",
        comparison_filename="metrics.csv",
    )
    run_pipeline(input_path, output_dir, settings)

    output = capsys.readouterr().out
    assert "Dataset not found" in output
    assert input_path.is_file()
    assert requested_urls == [(data_module.DATASET_URL, 30)]
