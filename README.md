# Telco Customer Churn Pipeline

This project uses the IBM Telco Customer Churn sample dataset to build a reproducible, end-to-end binary classification pipeline that predicts whether a customer will churn.

## Project Goal

The final deliverables include data acquisition, validation, cleaning, exploratory data analysis (EDA), preprocessing, model training and comparison, evaluation, customer risk output, unit tests, GitHub Actions, and a final presentation.

## Project Roadmap

1. **Environment and data**: Create a virtual environment, install dependencies, and download the IBM CSV file.
2. **EDA**: Inspect the dataset shape, data types, missing and blank values, duplicates, summary statistics, category distributions, and relationships with churn.
3. **Data quality**: Define the schema, handle blank `TotalCharges` values, duplicate rows, and invalid values, and produce a validation report.
4. **Preprocessing**: Exclude identifier columns, scale numerical features, one-hot encode categorical features, and create a reproducible train/test split.
5. **Modeling and evaluation**: Establish a Logistic Regression baseline and compare it with a Decision Tree using precision, recall, F1, ROC AUC, and a confusion matrix.
6. **Risk output**: Produce churn probabilities and risk levels for each customer, then save the trained model and predictions.
7. **Automation**: Add pytest, linting, and GitHub Actions so validation can be reproduced for every change.
8. **Presentation**: Summarize the business problem, data findings, technical decisions, model results, limitations, and next steps.

## Phase One: Current Tasks

Activate the `.venv` created by VS Code and install the project:

```powershell
python -m pip install -e ".[dev]"
```

Download the dataset to `data/raw/Telco-Customer-Churn.csv`, open `notebooks/01_data_exploration.ipynb`, and complete the following tasks in order:

1. Display the row count, column count, and first five rows.
2. Inspect column data types and the number of unique values in each column.
3. Check for nulls, blank strings, duplicate rows, and duplicate `customerID` values.
4. Review numerical summary statistics and categorical value frequencies.
5. Calculate the overall churn rate.
6. Compare churn rates across `Contract`, `InternetService`, and `PaymentMethod` categories.
7. Compare the distributions of `tenure`, `MonthlyCharges`, and `TotalCharges` for churned and retained customers.
8. Record three to five data-quality or business findings. Do not train a model during this phase.

Data source: [IBM Telco Customer Churn repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv)

## Project Structure

```text
data/               Raw and processed datasets
notebooks/          Exploration and experiments
outputs/            Reports, predictions, and model artifacts
src/telco_churn/    Reusable production modules
tests/              Unit tests matching source modules
```

## Validation

```powershell
pytest
ruff check .
```
