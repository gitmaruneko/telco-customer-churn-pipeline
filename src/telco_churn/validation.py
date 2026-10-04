from pathlib import Path

import pandas as pd

from telco_churn.data import load_raw_dataset

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

EXPECTED_CATEGORIES = {
    "gender": {"Female", "Male"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}


class DatasetValidationError(ValueError):
    """Raised when the input does not satisfy the raw dataset contract."""


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


def check_duplicate_customer_ids(data: pd.DataFrame) -> int:
    """Return the number of duplicate customer IDs."""
    return int(data["customerID"].duplicated().sum())


def check_missing_values(data: pd.DataFrame) -> pd.Series:
    """Count null and blank values in each column."""
    missing = data.isna()
    for column in data.select_dtypes(include="object").columns:
        missing[column] |= data[column].astype("string").str.strip().eq("").fillna(False)
    return missing.sum()


def check_data_types(data: pd.DataFrame) -> pd.Series:
    """Return the actual data type of each column."""
    return data.dtypes.astype(str)


def check_columns_and_rows(data: pd.DataFrame) -> tuple[int, int]:
    """Print and return the dataset dimensions."""
    rows, columns = data.shape
    print(f"Rows: {rows}")
    print(f"Columns: {columns}")
    return rows, columns


def check_required_columns(data: pd.DataFrame) -> None:
    """Check if all required columns are present in the dataset."""
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(f"Missing required columns: {missing}")


def check_total_charges_conversion(data: pd.DataFrame) -> int:
    """Count TotalCharges values that cannot be converted to numeric."""
    converted_numeric = pd.to_numeric(data["TotalCharges"], errors="coerce")
    return int(converted_numeric.isna().sum())


def check_tenure_range(data: pd.DataFrame) -> None:
    invalid_tenure = (data["tenure"] < 0) | (data["tenure"] > 72)
    if invalid_tenure.any():
        raise DatasetValidationError(f"Invalid tenure values found:\n{data[invalid_tenure]}")


def check_monthly_charges_non_negative(data: pd.DataFrame) -> None:
    invalid_monthly_charges = data["MonthlyCharges"] < 0
    if invalid_monthly_charges.any():
        raise DatasetValidationError(
            f"Invalid MonthlyCharges values found:\n{data[invalid_monthly_charges]}"
        )


def check_total_charges_non_negative(data: pd.DataFrame) -> None:
    converted_numeric = pd.to_numeric(data["TotalCharges"], errors="coerce")
    invalid_total_charges = converted_numeric < 0
    if invalid_total_charges.any():
        raise DatasetValidationError(
            f"Invalid TotalCharges values found:\n{data[invalid_total_charges]}"
        )


def check_churn_valid_values(data: pd.DataFrame) -> None:
    invalid_churn = ~data["Churn"].isin({"Yes", "No"})
    if invalid_churn.any():
        raise DatasetValidationError(f"Invalid Churn values found:\n{data[invalid_churn]}")


def check_categorical_values(data: pd.DataFrame) -> None:
    """Check categorical columns for unexpected values."""
    for column, valid_values in EXPECTED_CATEGORIES.items():
        if column not in data.columns:
            continue

        invalid_values = ~data[column].isin(valid_values)
        if invalid_values.any():
            raise DatasetValidationError(
                f"Invalid values found in column '{column}':\n"
                f"{data.loc[invalid_values, ['customerID', column]]}"
            )


def create_validation_report(data: pd.DataFrame) -> str:
    """Build a readable summary of dataset structure and quality checks."""
    rows, columns = data.shape
    missing_columns = sorted(EXPECTED_COLUMNS.difference(data.columns))
    missing_values = check_missing_values(data)
    missing_fields = [
        f"{column} ({count})" for column, count in missing_values.items() if count > 0
    ]

    duplicate_count = (
        check_duplicate_customer_ids(data) if "customerID" in data.columns else None
    )
    total_conversion_issues = (
        check_total_charges_conversion(data) if "TotalCharges" in data.columns else None
    )

    invalid_tenure_count: int | None = None
    if "tenure" in data.columns:
        tenure = pd.to_numeric(data["tenure"], errors="coerce")
        invalid_tenure_count = int((tenure.isna() | (tenure < 0) | (tenure > 72)).sum())

    monthly_negative_count: int | None = None
    if "MonthlyCharges" in data.columns:
        monthly_charges = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
        monthly_negative_count = int(monthly_charges.lt(0).sum())

    total_negative_count: int | None = None
    if "TotalCharges" in data.columns:
        total_charges = pd.to_numeric(data["TotalCharges"], errors="coerce")
        total_negative_count = int(total_charges.lt(0).sum())

    invalid_target_count = (
        int((~data["Churn"].isin({"Yes", "No"})).sum()) if "Churn" in data.columns else None
    )

    unexpected_categories: list[str] = []
    for column, valid_values in EXPECTED_CATEGORIES.items():
        if column not in data.columns:
            continue
        invalid_values = data.loc[~data[column].isin(valid_values), column]
        for value, count in invalid_values.value_counts(dropna=False).items():
            label = "<missing>" if pd.isna(value) else repr(value)
            unexpected_categories.append(f"{column}: {label} ({count})")

    data_types = check_data_types(data)
    review_needed = (
        bool(missing_columns)
        or rows == 0
        or bool(missing_fields)
        or bool(duplicate_count)
        or bool(total_conversion_issues)
        or bool(invalid_tenure_count)
        or bool(monthly_negative_count)
        or bool(total_negative_count)
        or bool(invalid_target_count)
        or bool(unexpected_categories)
    )

    lines = [
        "DATA VALIDATION REPORT",
        "----------------------",
        f"Rows: {rows}",
        f"Columns: {columns}",
        f"Duplicate customer IDs: {duplicate_count if duplicate_count is not None else 'Unavailable'}",
        f"Missing-value fields: {', '.join(missing_fields) if missing_fields else 'None'}",
        f"Missing required columns: {', '.join(missing_columns) if missing_columns else 'None'}",
        "Data types:",
    ]
    lines.extend(f"  {column}: {dtype}" for column, dtype in data_types.items())
    lines.extend(
        [
            (
                "TotalCharges conversion issues: "
                f"{total_conversion_issues if total_conversion_issues is not None else 'Unavailable'}"
            ),
            (
                "Invalid tenure values (expected 0-72): "
                f"{invalid_tenure_count if invalid_tenure_count is not None else 'Unavailable'}"
            ),
            (
                "Negative numeric values: "
                f"MonthlyCharges={monthly_negative_count if monthly_negative_count is not None else 'Unavailable'}, "
                f"TotalCharges={total_negative_count if total_negative_count is not None else 'Unavailable'}"
            ),
            (
                "Invalid target values: "
                f"{invalid_target_count if invalid_target_count is not None else 'Unavailable'}"
            ),
            (
                "Unexpected categorical values: "
                f"{'; '.join(unexpected_categories) if unexpected_categories else 'None'}"
            ),
            f"Overall status: {'REVIEW' if review_needed else 'PASS'}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco-Customer-Churn.csv"
    data = load_raw_dataset(data_path)
    print(create_validation_report(data))


if __name__ == "__main__":
    main()
