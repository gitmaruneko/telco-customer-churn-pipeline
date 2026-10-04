import pandas as pd
import pytest

from telco_churn.validation import (
    EXPECTED_COLUMNS,
    DatasetValidationError,
    validate_raw_dataset,
)


def make_valid_data() -> pd.DataFrame:
    row = {column: "value" for column in EXPECTED_COLUMNS}
    row["customerID"] = "0001-AAAAA"
    return pd.DataFrame([row])


def test_validate_raw_dataset_accepts_expected_columns() -> None:
    validate_raw_dataset(make_valid_data())


def test_validate_raw_dataset_rejects_missing_columns() -> None:
    data = make_valid_data().drop(columns="Churn")

    with pytest.raises(DatasetValidationError, match="Churn"):
        validate_raw_dataset(data)


def test_validate_raw_dataset_rejects_duplicate_customer_ids() -> None:
    data = pd.concat([make_valid_data(), make_valid_data()], ignore_index=True)

    with pytest.raises(DatasetValidationError, match="1 duplicate"):
        validate_raw_dataset(data)
