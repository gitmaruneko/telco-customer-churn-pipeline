from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from telco_churn.data import load_raw_dataset
from telco_churn.preprocessing import build_preprocessor, clean_data, split_features_target
from telco_churn.validation import validate_raw_dataset


# Task 7 — Train/test split
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

# Task 8 — Build the classifier
# X_train
#    ↓
# preprocessor
#    ↓
# categorical columns → OneHotEncoder
# numeric columns     → StandardScaler
#    ↓
# Logistic Regression

def build_model(features):
    """Build the preprocessing and Logistic Regression pipeline."""
    preprocessor = build_preprocessor(features)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            # → 先處理資料
            # → categorical features 做 One-Hot Encoding
            # → numeric features 保留
            ("classifier", LogisticRegression(max_iter=1000)),
            # → 再把處理好的數值資料交給 Logistic Regression
        ]
    )
    return model

def main() -> None:
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco-Customer-Churn.csv"
    data = load_raw_dataset(data_path)
    validate_raw_dataset(data)
    cleaned_data = clean_data(data)
    _customer_ids, features, target = split_features_target(cleaned_data)
    X_train, X_test, y_train, y_test = split_train_test(features, target)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("\nTraining target ratio:")
    print(y_train.value_counts(normalize=True))

    print("\nTest target ratio:")
    print(y_test.value_counts(normalize=True))

    model = build_model(X_train)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\npredictions[:10]: {predictions[:10]}")

# Task 9 — Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print("\nAccuracy:", accuracy)

    precision = precision_score(y_test, predictions)
    print("\nPrecision:", precision)

    recall = recall_score(y_test, predictions)

    print("\nRecall:", recall)

    f1 = f1_score(y_test, predictions)
    print("\nF1 Score:", f1)

    conf_matrix = confusion_matrix(y_test, predictions)

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nChurn Probabilities:")
    churn_probabilities = model.predict_proba(X_test)[:, 1]
    print(churn_probabilities[:10])

    test_customer_ids = _customer_ids.loc[X_test.index]

    # print(test_customer_ids.head())

    predictions_df = pd.DataFrame({
        "customerID": test_customer_ids.to_numpy(),
        "actual_churn": y_test.to_numpy(),
        "predicted_churn": predictions,
        "churn_probability": churn_probabilities,
    })

    print(predictions_df.head())

    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "predictions.csv"
    predictions_df.to_csv(output_path, index=False)




if __name__ == "__main__":
    main()
