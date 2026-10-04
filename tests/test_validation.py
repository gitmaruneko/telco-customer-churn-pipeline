import pandas as pd

from telco_churn.validation import create_validation_report


def make_valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "customerID": "0001-AAAAA",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "Yes",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.5,
                "TotalCharges": "100.0",
                "Churn": "No",
            }
        ]
    )


def test_validation_report_shows_required_summary_and_pass_status() -> None:
    report = create_validation_report(make_valid_data())

    assert "Rows: 1" in report
    assert "Columns: 21" in report
    assert "Duplicate customer IDs: 0" in report
    assert "Missing-value fields: None" in report
    assert "gender: object" in report
    assert "TotalCharges conversion issues: 0" in report
    assert "Invalid target values: 0" in report
    assert "Unexpected categorical values: None" in report
    assert "Overall status: PASS" in report


def test_validation_report_marks_data_quality_issues_for_review() -> None:
    data = make_valid_data()
    invalid_row = data.iloc[0].copy()
    invalid_row["tenure"] = 73
    invalid_row["MonthlyCharges"] = -1.0
    invalid_row["TotalCharges"] = " "
    invalid_row["Churn"] = "Maybe"
    invalid_row["gender"] = "Unknown"
    data = pd.concat([data, invalid_row.to_frame().T], ignore_index=True)

    report = create_validation_report(data)

    assert "Rows: 2" in report
    assert "Duplicate customer IDs: 1" in report
    assert "Missing-value fields: TotalCharges (1)" in report
    assert "TotalCharges conversion issues: 1" in report
    assert "Invalid tenure values (expected 0-72): 1" in report
    assert "MonthlyCharges=1" in report
    assert "Invalid target values: 1" in report
    assert "gender: 'Unknown' (1)" in report
    assert "Churn: 'Maybe' (1)" in report
    assert "Overall status: REVIEW" in report
