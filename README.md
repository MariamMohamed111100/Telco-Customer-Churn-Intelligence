# 📊 Telco Customer Churn Intelligence Dashboard

An end-to-end Machine Learning project that predicts customer churn risk in a telecom company using advanced ML models and deploys an interactive business-ready dashboard using Streamlit.

---

## 🚀 Project Overview

Customer churn is a critical business problem in telecom.  
This project builds a high-performance churn prediction model and deploys it as an interactive executive dashboard.

The system provides:

- 🔮 Churn probability prediction
- 📈 Risk segmentation (Low / Medium / High)
- 🎯 Model confidence score
- 💰 Estimated revenue at risk
- 🔄 What-if simulation (contract upgrade impact)
- 📊 Feature importance
- 🧠 SHAP-based explainability

---

## 🧠 Machine Learning Pipeline

- Feature Engineering:
  - CLV Ratio
  - Contract Encoding
  - Risk Indicators
  - Service-based engineered features

- Preprocessing:
  - StandardScaler (Numerical)
  - OneHotEncoder (Categorical)
  - SMOTE (Class imbalance handling)

- Models Tested:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - CatBoost

- Final Model:
  - XGBoost (optimized with GridSearchCV)

### Final Performance:

| Metric | Score |
|--------|--------|
| ROC-AUC | 0.84 |
| Recall (Churn) | 0.75 |
| Accuracy | 0.75 |

---

## 📊 Dashboard Features

### Risk Classification
- Low Risk (<35%)
- Medium Risk (35–65%)
- High Risk (>65%)

### Business Intelligence Layer
- Estimated Customer Lifetime Value
- Expected Revenue at Risk
- Retention Recommendation Engine
- What-if Contract Simulation

### Explainable AI
- Top 3 SHAP Risk Drivers per customer
- Feature importance visualization

---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/your-username/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction

pip install -r requirements.txt

streamlit run app/app.py
