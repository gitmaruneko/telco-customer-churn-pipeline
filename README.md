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

## Dataset Overview

The project uses the IBM Telco Customer Churn dataset, which contains 7,043
customer records and 21 columns. The binary target is `Churn`: 26.5% of
customers churned and 73.5% were retained.

The columns describe:

- customer demographics, such as gender, senior citizen status, partner, and dependents
- account history, including tenure and contract type
- phone, internet, streaming, security, backup, and support services
- billing choices, monthly charges, and total charges
- the customer identifier and churn target

### Download the Dataset

The raw CSV is not stored in Git. Download the IBM dataset and save it as
`data/raw/Telco-Customer-Churn.csv` before running the application:

```powershell
New-Item -ItemType Directory -Force data/raw
Invoke-WebRequest `
	-Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
	-OutFile "data/raw/Telco-Customer-Churn.csv"
```

Keep this file unchanged. Cleaning creates a separate DataFrame in memory and
does not overwrite the raw CSV.

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

## Project Architecture

The production workflow is organized by responsibility:

```text
data/raw/Telco-Customer-Churn.csv
	-> data.py             Load the raw CSV while preserving quality issues
	-> validation.py       Validate the schema and report data-quality issues
	-> preprocessing.py    Clean values, split features/target, encode, and scale
	-> models.py           Build Logistic Regression and Decision Tree pipelines
	-> train.py            Split data, train models, evaluate, and create outputs
	-> output/             Write model_comparison.csv and predictions.csv
```

## Validation Checks

Before training, the raw-data gate verifies that all required columns exist,
the dataset is not empty, and `customerID` values are unique. The detailed
validation report also checks:

- null and blank values
- whether `TotalCharges` values can be converted to numeric values
- whether `tenure` is between 0 and 72 months
- whether `MonthlyCharges` and `TotalCharges` contain negative values
- whether `Churn` contains only `Yes` and `No` before encoding
- whether categorical columns contain unexpected values

## Data Cleaning and Preprocessing Decisions

- The raw dataset is not modified directly. `clean_data` creates a copy so the original values remain available for comparison and reproducibility.
- `TotalCharges` is stored as text in the raw CSV, so it is converted to numeric values for analysis and model use. Values that cannot be converted are treated as missing and filled with `0`. In this dataset, 11 blank `TotalCharges` values were observed; they correspond to customers with `tenure = 0`, so those customer records are kept rather than removed.
- `Churn` is mapped from `Yes`/`No` to `1`/`0` so it can be used as a binary classification target.
- `customerID` is excluded from the model features because it identifies a customer rather than describing customer behavior. It is returned separately so predictions can be associated with the corresponding customer.
- Categorical feature columns are one-hot encoded because their values are labels, not quantities with a meaningful numeric order. `handle_unknown="ignore"` allows the encoder to process categories not seen when it was fitted.
- Numerical feature columns are standardized so they are on comparable scales, which helps Logistic Regression converge.
- The `ColumnTransformer` applies one-hot encoding to categorical columns and standardization to numerical columns. To prevent data leakage, fit the preprocessor on the training data only, then use it to transform both training and test data.

## Model Features

The model uses 19 input features. `customerID` is retained only to associate
predictions with customers, and `Churn` is used as the target.

- **Demographic:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Charges:** `MonthlyCharges`, `TotalCharges`

## Train/Test Split

- The data is split into 80% training data and 20% test data (`test_size=0.20`).
- `random_state=42` is fixed so the same split can be reproduced across runs.
- The split is stratified by the `Churn` target (`stratify=target`) to preserve the churn-class proportions in both sets.

## Exploratory Data Analysis Findings

- Churn is imbalanced: 26.5% of customers churned, while 73.5% were retained.
- Month-to-month customers represent 55% of the dataset and have a 42.7% churn rate, compared with 11.3% for one-year and 2.8% for two-year contracts.
- Customers with 0–12 months of tenure have the highest churn rate at 47.4%; churn decreases across longer-tenure groups.
- Churned customers have higher average monthly charges ($74.44) than retained customers ($61.27).
- Customers without TechSupport have a 41.6% churn rate, compared with 15.2% for customers with TechSupport.
- Fiber optic customers have a 41.9% churn rate, compared with 19.0% for DSL customers and 7.4% for customers without internet service.

These findings describe associations in this dataset and do not establish
that any feature causes churn.

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

## Example Output

The training command writes aggregate metrics to `output/model_comparison.csv`
and customer-level predictions to `output/predictions.csv`. One prediction row
has the following form:

| customerID | actual_churn | predicted_churn | churn_probability | decision_tree_predicted_churn | decision_tree_churn_probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4376-KFVRS | 0 | 0 | 0.0459 | 0 | 0.0 |

## Limitations

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

Additional limitations include the use of one public dataset and one holdout
split, class imbalance, and no probability-threshold optimization. Filling the
11 blank `TotalCharges` values with zero is supported by their zero-tenure
records, but remains a modeling assumption.

## Future Improvements

- Use cross-validation and tune model hyperparameters, including Decision Tree depth.
- Select and calibrate a classification threshold using retention costs and churn risk.
- Validate the selected model on additional data and monitor data or performance drift.
- Evaluate additional feature engineering and interpretable model explanations.

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
python -m pip install -r requirements.txt
```

Python 3.11 or later is required.

## Run the Application

```powershell
python main.py --input data/raw/Telco-Customer-Churn.csv
```

The equivalent module command is:

```powershell
python -m telco_churn.train --input data/raw/Telco-Customer-Churn.csv
```

The application reports each loading, validation, cleaning, training,
evaluation, and output step. It writes `output/model_comparison.csv` and
`output/predictions.csv` after a successful run. The input must include the
`Churn` target because this command trains the models and evaluates them on a
held-out test split; it is not an inference command for unlabeled customers.

To print the detailed validation report without training models, run:

```powershell
python -m telco_churn.validation
```

## Run Tests

```powershell
python -m pytest
ruff check .
```

## Data Source

[IBM Telco Customer Churn dataset](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Notes

This README is intended as the formal project overview for repository users and reviewers. Detailed personal learning notes live in [Note.md](Note.md).
