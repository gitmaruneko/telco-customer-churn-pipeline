# Learning Notes

This file records the personal learning process for the Telco Customer Churn project.

## Purpose

This is not the project homepage. It is a study log for understanding the workflow, concepts, and practical steps involved in building the pipeline.

---

## 2026-09-25

### Project Overview

This project is a:

**Customer Churn Prediction and Data Quality Pipeline**

Goal:

- Predict whether a customer will `Churn` or `Not Churn`
- Build a complete end-to-end machine learning workflow

Main pipeline:

**CSV → Validation → Cleaning → EDA → Preprocessing → Model → Evaluation → Prediction → Tests → GitHub Actions**

### Learning Focus

The first priority is to understand the lifecycle of a machine learning pipeline, not just the final model.

---

## 2026-09-27

### EDA - Exploratory Data Analysis

**Goal:** Understand the dataset before building a machine learning model.

#### Main Steps

1. **Check Data Size**

   - Number of rows and columns
   - Example: `df.shape`
2. **Check Columns**

   - Column names
   - Example: `df.columns`
3. **Check Data Types**

   - Numeric vs. categorical
   - Example: `df.info()`
4. **Check Basic Statistics**

   - Mean, min, max, standard deviation
   - Example: `df.describe()`
5. **Check Value Distribution**

   - Unique values and category frequencies
   - Example: `value_counts()`
6. **Check Data Quality**

   - Missing values
   - Duplicates
   - Incorrect data types
7. **Visualize the Data**

   - Count plots
   - Histograms
   - Box plots
8. **Find Patterns**

   - Investigate relationships between features and the target
   - Example: which factors relate to `Churn`?

#### Simple Formula

**EDA = Understand Data + Find Problems + Find Patterns**

#### ML Workflow

`Data → EDA → Cleaning → Preprocessing → Model → Evaluation`

---

### Summary

This note is meant to track project understanding, learning progress, and concept review. The formal project information is kept in [README.md](README.md).
