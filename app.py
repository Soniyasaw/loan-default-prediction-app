import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model, scaler, and feature columns
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Loan Default Prediction", page_icon="💰")
st.title("📉 Loan Default Prediction App")

# User Inputs
age = st.slider("Person Age", 18, 70, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Associate", "Others"])
income = st.number_input("Annual Income", min_value=0)
employment_exp = st.slider("Years of Employment", 0, 40, 5)
home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amount = st.number_input("Loan Amount", min_value=100)
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0)
loan_percent_income = st.number_input("Loan % of Income", min_value=0.0)
credit_hist_length = st.slider("Credit History Length", 0, 30, 5)
credit_score = st.number_input("Credit Score", min_value=0)
defaults_on_file = st.selectbox("Previous Loan Defaults on File", ["Yes", "No"])

# Encode binary features
gender = 1 if gender == "Female" else 0
defaults_on_file = 1 if defaults_on_file == "Yes" else 0

# One-Hot Encoding for dropdowns
input_dict = {
    'person_age': age,
    'person_gender': gender,
    'person_income': income,
    'person_emp_exp': employment_exp,
    'loan_amnt': loan_amount,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': credit_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': defaults_on_file,
    f'person_education_{education}': 1,
    f'person_home_ownership_{home_ownership}': 1,
    f'loan_intent_{loan_intent}': 1
}

# Add missing dummy columns
for col in feature_columns:
    if col not in input_dict:
        input_dict[col] = 0

# Prepare DataFrame
input_df = pd.DataFrame([input_dict])[feature_columns]
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Likely to Default on Loan (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Loan Likely to be Repaid (Probability: {probability:.2%})")