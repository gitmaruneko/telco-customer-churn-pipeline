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
pytest
ruff check .
```

## Data Source

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Notes

This README is intended as the formal project overview for repository users and reviewers. Detailed personal learning notes live in [Note.md](Note.md).
