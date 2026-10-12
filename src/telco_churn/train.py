import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from telco_churn.data import load_raw_dataset
from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.preprocessing import clean_data, split_features_target
from telco_churn.validation import DatasetValidationError, validate_raw_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def split_train_test(features, target):
    """Split features and target into reproducible training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def build_predictions_output(
    customer_ids,
    actual_churn,
    predictions_by_model,
    probabilities_by_model,
) -> pd.DataFrame:
    """Build the customer-level prediction output table."""
    return pd.DataFrame(
        {
            "customerID": customer_ids.to_numpy(),
            "actual_churn": actual_churn.to_numpy(),
            "predicted_churn": predictions_by_model["Logistic Regression"],
            "churn_probability": probabilities_by_model["Logistic Regression"],
            "decision_tree_predicted_churn": predictions_by_model["Decision Tree"],
            "decision_tree_churn_probability": probabilities_by_model["Decision Tree"],
        }
    )


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the training pipeline and return metrics and customer predictions."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    print("=" * 40)
    print("TELCO CUSTOMER CHURN ANALYSIS")
    print("=" * 40)

    print("[1] Loading data...")
    data = load_raw_dataset(input_path)
    print(f"Rows loaded: {len(data)}")

    print("[2] Validating data...")
    validate_raw_dataset(data)
    print("Validation complete.")

    print("[3] Cleaning data...")
    cleaned_data = clean_data(data)
    customer_ids, features, target = split_features_target(cleaned_data)
    X_train, X_test, y_train, y_test = split_train_test(features, target)
    print("Cleaning complete.")

    print("[4] Training models...")
    models = {
        "Logistic Regression": build_logistic_regression_model(X_train),
        "Decision Tree": build_decision_tree_model(X_train),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    print("Models trained.")

    print("[5] Evaluating models...")
    predictions_by_model = {}
    probabilities_by_model = {}
    metric_rows = []
    confusion_matrices = {}

    for name, model in models.items():
        predictions = model.predict(X_test)
        churn_class_index = list(model.classes_).index(1)
        churn_probabilities = model.predict_proba(X_test)[:, churn_class_index]
        train_predictions = model.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

        predictions_by_model[name] = predictions
        probabilities_by_model[name] = churn_probabilities
        confusion_matrices[name] = [[int(tn), int(fp)], [int(fn), int(tp)]]
        metric_rows.append(
            {
                "Model": name,
                "Train Accuracy": accuracy_score(y_train, train_predictions),
                "Test Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for name, matrix in confusion_matrices.items():
        print(f"{name} confusion matrix (labels=[0, 1]): {matrix}")

    test_customer_ids = customer_ids.loc[X_test.index]
    predictions_df = build_predictions_output(
        test_customer_ids,
        y_test,
        predictions_by_model,
        probabilities_by_model,
    )
    print("[6] Writing predictions...")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "model_comparison.csv"
    predictions_df.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved: {predictions_path}")
    print(f"Saved: {metrics_path}")
    print("Pipeline completed successfully.")
    return metrics, predictions_df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Run the Telco customer churn analysis.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the labeled Telco customer churn CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Directory for generated CSV files (default: project output directory).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line training application."""
    args = parse_args(argv)
    try:
        run_pipeline(args.input, args.output)
    except (OSError, DatasetValidationError) as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
