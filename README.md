# 🚀 Loan Default Prediction App

This is a Machine Learning-powered web app built using **Streamlit** that predicts whether a person is likely to default on a loan based on input features like age, gender, education, income, and employment history.

🔗 **Live App**: [Loan Default Prediction App](https://loan-default-prediction-app-53yc8ueok7wgpyb6wmelir.streamlit.app/)

---

## 📌 Problem Statement

Predict whether a loan applicant will **default** using demographic and financial information. This model can help financial institutions make better decisions and manage risk.

---
## 📊 Features

- User-friendly web interface (built with Streamlit)
- Predictive model using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Feature scaling
- Class imbalance handled using **SMOTE**
- Input fields:
  - person_age 
  - person_income 
  - person_home_ownership
  - person_emp_length
  - loan_intent
  - loan_grade
  - loan_amnt
  - loan_int_rate
  - loan_status
  - person_gender
  - previous_loan_defaults_on_file
---

## 📁 Project Structure

loan-default-prediction-app/
│
├── app.py # Streamlit app script
├── model.pkl # Trained ML model
├── scaler.pkl # Scaler object
├── smote.pkl # SMOTE object
├── requirements.txt # Python dependencies
└── README.md # Project documentation
---

## 🧠 Machine Learning Workflow

- **Data Preprocessing**:
  - Label Encoding (`person_gender`, `previous_loan_defaults_on_file`)
  - One-Hot Encoding: for multiple categorical columns
  - Feature Scaling (StandardScaler)
  - SMOTE for class balancing

- **Model Training**:
  - Logistic Regression, Random Forest, XGBoost, Gradient Boosting
  - Model saved using `joblib`

- **Deployment**:
  - App deployed on [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/soniyasaw/loan-default-prediction-app.git
cd loan-default-prediction-app
pip install -r requirements.txt
streamlit run app.py

📦 Requirements
Python 3.8+
streamlit
pandas
scikit-learn
xgboost
imbalanced-learn
joblib

Install all dependencies using:
pip install -r requirements.txt

📌 Author
👩‍💻 Soniya Saw
📬 LinkedIn: www.linkedin.com/in/soniya-saw
📫 Email: [soniyamandal697@gmail.com]
