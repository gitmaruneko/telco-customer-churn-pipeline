import argparse
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

from telco_churn.config import ModelSettings, load_model_settings
from telco_churn.data import load_raw_dataset
from telco_churn.models import build_decision_tree_model, build_logistic_regression_model
from telco_churn.preprocessing import clean_data, split_features_target
from telco_churn.validation import validate_raw_dataset


def build_argument_parser(settings: ModelSettings) -> argparse.ArgumentParser:
    """Create command-line options, using model settings as their defaults."""
    parser = argparse.ArgumentParser(
        description="Train and compare churn classification models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.dataset_path,
        help="Path to the input customer CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.output_directory,
        help="Directory for prediction and model comparison CSV files.",
    )
    return parser


def split_train_test(features, target, *, test_size: float = 0.20, random_state: int = 42):
    """Split features and target into reproducible training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return X_train, X_test, y_train, y_test


def main(argv: list[str] | None = None) -> None:
    settings = load_model_settings()
    args = build_argument_parser(settings).parse_args(argv)
    data_path = args.input
    data = load_raw_dataset(data_path)
    validate_raw_dataset(data)
    cleaned_data = clean_data(data)
    customer_ids, features, target = split_features_target(cleaned_data)
    X_train, X_test, y_train, y_test = split_train_test(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.split_random_state,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("\nTraining target ratio:")
    print(y_train.value_counts(normalize=True))
    print("\nTest target ratio:")
    print(y_test.value_counts(normalize=True))

    models = {
        "Logistic Regression": build_logistic_regression_model(
            X_train,
            max_iter=settings.logistic_regression_max_iter,
        ),
        "Decision Tree": build_decision_tree_model(
            X_train,
            random_state=settings.decision_tree_random_state,
        ),
    }
    predictions_by_model = {}
    probabilities_by_model = {}
    metric_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        churn_class_index = list(model.classes_).index(1)
        churn_probabilities = model.predict_proba(X_test)[:, churn_class_index]
        train_predictions = model.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

        predictions_by_model[name] = predictions
        probabilities_by_model[name] = churn_probabilities
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

        print(f"\n{name} confusion matrix (rows=actual, columns=predicted; labels=[0, 1]):")
        print([[tn, fp], [fn, tp]])

    metrics = pd.DataFrame(metric_rows)
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    test_customer_ids = customer_ids.loc[X_test.index]
    predictions_df = pd.DataFrame(
        {
            "customerID": test_customer_ids.to_numpy(),
            "actual_churn": y_test.to_numpy(),
            "predicted_churn": predictions_by_model["Logistic Regression"],
            "churn_probability": probabilities_by_model["Logistic Regression"],
            "decision_tree_predicted_churn": predictions_by_model["Decision Tree"],
            "decision_tree_churn_probability": probabilities_by_model["Decision Tree"],
        }
    )
    print("\nPredictions:")
    print(predictions_df.head())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / settings.predictions_filename
    comparison_path = args.output_dir / settings.comparison_filename
    predictions_df.to_csv(predictions_path, index=False)
    metrics.to_csv(comparison_path, index=False)
    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Model comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()
