# Employee Attrition Prediction

A machine learning project that predicts the likelihood of an employee leaving a company, comparing multiple supervised learning algorithms and deploying the best-performing model as an interactive Streamlit app.

## Overview

This project takes employee attributes - job satisfaction, income, overtime status, tenure, and more - and predicts whether an employee is at **High Risk** or **Low Risk** of attrition, along with a probability score.

Beyond just training a single model, this project focuses on a full ML workflow: handling class imbalance, comparing algorithms fairly, evaluating with the right metrics, and being transparent about what hyperparameter tuning did (and didn't) improve.

## Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset (Kaggle)
- **Rows:** 1,470 employee records
- **Target:** `Attrition` (Yes/No) - imbalanced at ~84% No / ~16% Yes
- **Features used:** Age, Gender, Department, JobRole, JobLevel, JobInvolvement, JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, OverTime, DistanceFromHome, BusinessTravel, StockOptionLevel, MonthlyIncome, TrainingTimesLastYear, YearsAtCompany, YearsWithCurrManager, YearsSinceLastPromotion

## Preprocessing

- Categorical features (`Department`, `JobRole`, `BusinessTravel`) one-hot encoded
- Binary features (`Gender`, `OverTime`) label-encoded
- Numerical features scaled with `StandardScaler` - **fit only on the training set** and applied to test data separately, to avoid data leakage
- Data split into train/test **before** scaling and resampling, to keep evaluation honest

## Handling Class Imbalance

The original dataset is imbalanced (~84/16), which makes raw accuracy a misleading metric. **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training set only, balancing it to 986/986 before model training. The test set was left untouched to reflect real-world class distribution during evaluation.

## Model Comparison

Four supervised learning algorithms were trained on the same SMOTE-balanced training data and evaluated on the same held-out test set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.755 | 0.329 | 0.511 | 0.400 | 0.656 |
| Decision Tree | 0.735 | 0.254 | 0.340 | 0.291 | 0.575 |
| Random Forest | 0.833 | 0.458 | 0.234 | 0.310 | 0.591 |
| **XGBoost** | **0.847** | **0.528** | **0.404** | **0.458** | **0.668** |

**XGBoost was selected as the final model**, achieving the best F1-score and ROC-AUC - the two most reliable metrics given the class imbalance (accuracy alone is misleading, since a model predicting "No attrition" for everyone would still score ~84%).

## Final Model Performance (XGBoost, default config)

```
              precision    recall  f1-score   support
           0       0.90      0.94      0.92       247
           1       0.57      0.45      0.50        47

    accuracy                           0.86       294
   macro avg       0.73      0.69      0.71       294
weighted avg       0.85      0.86      0.85       294

ROC-AUC: 0.668
```

The **macro average F1-score (0.71)** is the more representative metric here, since it weighs both classes equally rather than being dominated by the majority class.

## Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/pranavprajapati339-ops/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Project Structure

```
├── app.py                  # Streamlit app
├── model_xgb.pkl            # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
├── columns.pkl                # Expected input columns
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- scikit-learn (preprocessing, Logistic Regression, Decision Tree, Random Forest)
- XGBoost (final model)
- imbalanced-learn (SMOTE)
- pandas / numpy (data handling)
- Streamlit (web app / UI)
- joblib (model serialization)

## Note on Environment

This project pins `xgboost==3.2.0` in `requirements.txt`. Mismatched XGBoost versions between training and inference environments can cause feature-validation errors at prediction time, so make sure your local environment matches this version.

## Disclaimer

This tool is for educational and portfolio purposes only. It should not be used as the sole basis for real HR or employment decisions.

## Links

- **Live demo:** [Click here to see live demo.](https://pranavprajapati339-ops-employee-attrition-prediction-app-yy9vsk.streamlit.app/)
- **Training notebook:** [Colab Notebook](https://colab.research.google.com/drive/12uuN8lBfO4H8P87j3sgN7HNn-8p-O-iE?usp=sharing)

## Author

**Pranav** - Data Science, Machine Learning & AI
