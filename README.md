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
- Numerical feature columns are standardized so they are on comparable scales, which helps Logistic Regression converge.
- The `ColumnTransformer` applies one-hot encoding to categorical columns and standardization to numerical columns. To prevent data leakage, fit the preprocessor on the training data only, then use it to transform both training and test data.

## Train/Test Split

- The data is split into 80% training data and 20% test data (`test_size=0.20`).
- `random_state=42` is fixed so the same split can be reproduced across runs.
- The split is stratified by the `Churn` target (`stratify=target`) to preserve the churn-class proportions in both sets.

## Model Comparison and Evaluation

`python -m telco_churn.train` trains Logistic Regression and a Decision Tree
using the same stratified 80/20 split (`random_state=42`). Both pipelines use
the same preprocessing steps, and the script writes the metrics and test
predictions to `output/model_comparison.csv` and `output/predictions.csv`.

Results on the provided dataset:

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8056 | 0.8055 | 0.6572 | 0.5588 | 0.6040 |
| Decision Tree | 0.9980 | 0.7225 | 0.4781 | 0.4973 | 0.4875 |

Confusion matrices (rows are actual labels and columns are predicted labels;
label order is `0` = no churn and `1` = churn):

| Model | True Negative | False Positive | False Negative | True Positive |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 926 | 109 | 165 | 209 |
| Decision Tree | 832 | 203 | 188 | 186 |

In this baseline run, Logistic Regression performs better on every reported
test metric and has similar training and test accuracy. The default Decision
Tree reaches 0.9980 training accuracy but only 0.7225 test accuracy, a large
generalization gap that suggests overfitting. A tree can express nonlinear
rules and can be easier to turn into decision rules, but this unpruned tree
does not generalize as well; limiting tree depth or tuning it with
cross-validation would be a reasonable next step. Logistic Regression is the
stronger candidate in this comparison, but these holdout results alone do not
establish a universally best or deployment-ready model.

The metrics represent different costs: higher recall catches more customers
who will churn, while higher precision means fewer retention efforts are spent
on customers who would not churn. The better model for deployment depends on
the relative cost of missed churn versus unnecessary interventions, as well as
interpretability and validation on additional data.

## Project Structure

```text
data/               Raw and processed datasets
notebooks/          Exploratory notebooks and experiments
output/             Model comparison metrics and customer predictions
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
python -m telco_churn.train
pytest
ruff check .
```

## Data Source

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Notes

This README is intended as the formal project overview for repository users and reviewers. Detailed personal learning notes live in [Note.md](Note.md).
