from pathlib import Path

import pandas as pd

from telco_churn.data import load_raw_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Task 4 — Clean the data

# [x] Keep raw data unchanged
# [x] Convert TotalCharges to numeric
# [x] Handle missing TotalCharges values
# [x] Convert Churn to binary
# [x] Separate customerID from model features
# [x] Prepare categorical encoding approach
# [x] Document cleaning decisions in README

def clean_data(data):
    """Clean the raw Telco churn dataset without modifying the original data."""
    cleaned_data = data.copy()

    cleaned_data["TotalCharges"] = pd.to_numeric(
        cleaned_data["TotalCharges"],
        errors="coerce"
    )

    cleaned_data["TotalCharges"] = cleaned_data["TotalCharges"].fillna(0)

    # print("Raw TotalCharges dtype:", data["TotalCharges"].dtype)
    # print("Cleaned TotalCharges dtype:", cleaned_data["TotalCharges"].dtype)
    # print("Cleaned TotalCharges missing:", cleaned_data["TotalCharges"].isna().sum())
    # print("Raw blank TotalCharges:", (data["TotalCharges"] == " ").sum())
    # print("Cleaned blank TotalCharges:", (cleaned_data["TotalCharges"] == " ").sum())
    cleaned_data["Churn"] = cleaned_data["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    print(cleaned_data["Churn"].value_counts())
    print(cleaned_data["Churn"].dtype)
    return cleaned_data

# X / y
def split_features_target(data):
    """Separate customer IDs, features, and target."""
    customer_ids = data["customerID"].copy()
    features = data.drop(columns=["customerID", "Churn"])
    target = data["Churn"].copy()
    return customer_ids, features, target

# encoding
# Machine learning models cannot directly use text categories such as
# "Month-to-month" or "Fiber optic".
# Convert categorical features into numeric columns using one-hot encoding.
#
# Contract
# ---------
# Month-to-month
# Two year
# One year

# convert to

# Contract_Month-to-month   Contract_One year   Contract_Two year
# 1                         0                   0
# 0                         0                   1
# 0                         1                   0

# cleaned_data
#       ↓
# features + target
#       ↓
# train_test_split()
#       ↓
# ┌─────────────┬─────────────┐
# X_train       X_test
# y_train       y_test
#       ↓
# preprocessor.fit(X_train)
#       ↓
# learning training data  preprocessing rule
#       ↓
# transform(X_train)
# transform(X_test)

def build_preprocessor(features):
    """Build preprocessing rules for categorical features."""
    categorical_features = features.select_dtypes(include="object").columns.tolist()
    encoder = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", encoder, categorical_features)
        ],
        remainder="passthrough"
    )
    return preprocessor

def main() -> None:
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco-Customer-Churn.csv"
    data = load_raw_dataset(data_path)
    cleaned_data = clean_data(data)
    customer_ids, features, target = split_features_target(cleaned_data)
    # categorical_features = features.select_dtypes(include="object").columns.tolist()
    # # print(categorical_features)
    # encoder = OneHotEncoder(handle_unknown="ignore")
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ("categorical", encoder, categorical_features)
    #     ],
    #     remainder="passthrough"
    # )
    numerical_features = features.select_dtypes(include="number").columns.tolist()
    # print(numerical_features)


if __name__ == "__main__":
    main()


