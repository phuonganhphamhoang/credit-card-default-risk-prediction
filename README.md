# 💳 Credit Card Default Risk Prediction

This project aims to predict the likelihood of a credit card holder defaulting on their payment next month using supervised machine learning techniques. The dataset is based on customers in Taiwan and includes demographic, behavioral, and historical payment features.

---

## 🎯 Objectives

- Identify key factors influencing credit default risk.
- Apply various ML models and handle class imbalance using oversampling techniques.
- Evaluate and compare models to select the most effective approach.
- Save the best-performing model for future deployment.

---

## 📊 Dataset Overview

- **Source:** [UCI - Default of Credit Card Clients Dataset (Taiwan, 2005)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Size:** 30,000 records, 24 features + target
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)

---

## 🧼 Data Preprocessing

- Renamed variables for clarity (`PAY_0 → PAY_1`, `default.payment.next.month → default`)
- Grouped age into bins, created `AGE_BIN`
- Handled outliers using boxplot-based filtering
- Created `LIMIT_BAL` groupings
- Encoded categorical features
- Removed irrelevant columns (`ID`)

---

## 📊 Exploratory Analysis

- High correlation between delayed payments over consecutive months (PAY_1 ~ PAY_6)
- Default rate ≈ 22.2%
- Delays in PAY_1 and PAY_2 are the most impactful indicators of default
- Sociodemographic features (gender, marriage, education) have limited predictive power
- Visualized credit limits, age distributions, education levels, default rates by segment

---

## 🤖 Models Trained

We trained and evaluated 4 classifiers with **two oversampling strategies** (Random OverSampling & SMOTE):

| Model               | Oversampling     | Accuracy | F1 Score | Precision | Recall | ROC AUC |
|--------------------|------------------|----------|----------|-----------|--------|---------|
| Logistic Regression| SMOTE            | 72%      | 0.70     | 0.71      | 0.69   | 0.74    |
| K-Nearest Neighbors| SMOTE            | 77%      | 0.76     | 0.76      | 0.77   | 0.81    |
| XGBoost            | SMOTE            | 77%      | 0.74     | 0.73      | 0.81   | 0.79    |
| **Random Forest**  | **ROS** ✅        | **93%**  | **0.93** | **0.91**  | **0.95**| **0.98** ✅ |

> 📌 **Final Model**: Random Forest + Random OverSampling (best overall performance)

---

## 🔍 Feature Importance (Random Forest)

| Feature       | Importance |
|---------------|------------|
| PAY_1         | Highest    |
| LIMIT_BAL     | High       |
| PAY_2 ~ PAY_6 | Moderate   |
| EDUCATION, AGE, SEX | Low |

---

## 📈 ROC Curve

- AUC Score (Random Forest): **0.98**
- Model shows excellent discriminatory power between default and non-default classes.

---

## 💾 Model Export

```python
from joblib import dump
dump(final_model, 'models/BestModel_RF.joblib')

# Load later for inference
from joblib import load
model = load('models/BestModel_RF.joblib')
