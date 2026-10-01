from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = frozenset(
    {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }
)


class DatasetValidationError(ValueError):
    """Raised when the input does not satisfy the raw dataset contract."""


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV without hiding data-quality issues from exploration."""
    return pd.read_csv(path, keep_default_na=False)


def validate_raw_dataset(data: pd.DataFrame) -> None:
    """Validate the minimum structural contract of the IBM sample dataset."""
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(f"Missing required columns: {missing}")

    if data.empty:
        raise DatasetValidationError("Dataset must contain at least one customer")

    duplicate_ids = data["customerID"].duplicated().sum()
    if duplicate_ids:
        raise DatasetValidationError(f"Found {duplicate_ids} duplicate customerID values")
