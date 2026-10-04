# Telco Customer Churn Pipeline

This project builds a reproducible machine learning pipeline for predicting whether a customer will churn using the IBM Telco Customer Churn dataset.

## Project Goal

The goal is to create an end-to-end binary classification workflow that includes:

- data ingestion
- validation and cleaning
- exploratory data analysis
- preprocessing
- modeling and evaluation
- customer churn risk output
- automated testing
- CI/CD automation

## Business Problem

The model predicts whether a customer is likely to:

- `Churn`
- `Not Churn`

This helps identify at-risk customers and supports retention actions.

## Project Roadmap

1. **Environment setup**: create a virtual environment and install project dependencies.
2. **Data acquisition**: download the IBM CSV dataset.
3. **EDA**: inspect dataset structure, distributions, quality issues, and churn relationships.
4. **Data quality validation**: detect missing values, duplicates, invalid values, and schema issues.
5. **Preprocessing**: handle cleaning, feature encoding, and train/test splits.
6. **Modeling**: compare baseline and candidate classifiers.
7. **Evaluation**: assess model performance using classification metrics.
8. **Automation**: add tests and GitHub Actions for reproducibility.
9. **Presentation**: summarize findings and business impact.

## Data Cleaning and Preprocessing Decisions

- The raw dataset is not modified directly. `clean_data` creates a copy so the original values remain available for comparison and reproducibility.
- `TotalCharges` is stored as text in the raw CSV, so it is converted to numeric values for analysis and model use. Values that cannot be converted are treated as missing and filled with `0`. In this dataset, 11 blank `TotalCharges` values were observed; they correspond to customers with `tenure = 0`, so those customer records are kept rather than removed.
- `Churn` is mapped from `Yes`/`No` to `1`/`0` so it can be used as a binary classification target.
- `customerID` is excluded from the model features because it identifies a customer rather than describing customer behavior. It is returned separately so predictions can be associated with the corresponding customer.
- Categorical feature columns are one-hot encoded because their values are labels, not quantities with a meaningful numeric order. `handle_unknown="ignore"` allows the encoder to process categories not seen when it was fitted.
- The `ColumnTransformer` defines one-hot encoding for categorical columns and passes non-categorical columns through unchanged. To prevent data leakage, fit the preprocessor on the training data only, then use it to transform both training and test data.

## Project Structure

```text
data/               Raw and processed datasets
notebooks/          Exploratory notebooks and experiments
outputs/            Reports, predictions, and model artifacts
src/telco_churn/    Reusable production modules
tests/              Unit tests for project logic
```

## Setup

```powershell
python -m pip install -e ".[dev]"
```

## Validation

```powershell
python -m telco_churn.validation
pytest
ruff check .
```

## Data Source

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Notes

This README is intended as the formal project overview for repository users and reviewers. Detailed personal learning notes live in [Note.md](Note.md).
