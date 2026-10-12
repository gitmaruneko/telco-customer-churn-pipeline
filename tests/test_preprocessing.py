import pandas as pd
from pandas.api.types import is_numeric_dtype

from telco_churn.preprocessing import clean_data


def test_clean_data_converts_total_charges_to_numeric() -> None:
    data = pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB"],
            "MonthlyCharges": [50.5, 75.0],
            "TotalCharges": ["100.0", " "],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned_data = clean_data(data)

    assert is_numeric_dtype(cleaned_data["TotalCharges"])
    assert cleaned_data["TotalCharges"].tolist() == [100.0, 0.0]


def test_cleaned_data_has_no_negative_charges() -> None:
    data = pd.DataFrame(
        {
            "customerID": ["0001-AAAAA", "0002-BBBBB"],
            "MonthlyCharges": [50.5, 75.0],
            "TotalCharges": ["100.0", "225.0"],
            "Churn": ["No", "Yes"],
        }
    )

    cleaned_data = clean_data(data)

    assert cleaned_data[["MonthlyCharges", "TotalCharges"]].ge(0).all().all()
