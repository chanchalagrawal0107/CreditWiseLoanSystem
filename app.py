import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD MODEL ---------------- #
models = pickle.load(open("creditwise_models.pkl", "rb"))

st.set_page_config(page_title="CreditWise", layout="centered")
st.title("CreditWise – Loan Approval Prediction")

st.write("Fill in applicant details to predict loan approval.")

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model_name]


# ---------------- USER INPUTS ---------------- #

applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Co-applicant Income", min_value=0.0)
age = st.number_input("Age", min_value=18, max_value=100)
dependents = st.number_input("Dependents", min_value=0, max_value=10)

credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
existing_loans = st.number_input("Existing Loans", min_value=0)
dti_ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0)

savings = st.number_input("Savings Amount", min_value=0.0)
collateral_value = st.number_input("Collateral Value", min_value=0.0)

loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term (months)", min_value=1)

education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
employment_status = st.selectbox(
    "Employment Status", ["Salaried", "Self-employed", "Unemployed"]
)

marital_status = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox(
    "Loan Purpose", ["Car", "Education", "Home", "Personal"]
)

property_area = st.selectbox(
    "Property Area", ["Rural", "Semiurban", "Urban"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

employer_category = st.selectbox(
    "Employer Category", ["Government", "MNC", "Private", "Unemployed"]
)

# ---------------- FEATURE ENGINEERING ---------------- #

credit_score_sq = credit_score ** 2
dti_ratio_sq = dti_ratio ** 2

education_level = 1 if education_level == "Graduate" else 0
marital_single = 1 if marital_status == "Single" else 0
gender_male = 1 if gender == "Male" else 0

# Employment status (one-hot)
employment_salaried = 1 if employment_status == "Salaried" else 0
employment_self_employed = 1 if employment_status == "Self-employed" else 0
employment_unemployed = 1 if employment_status == "Unemployed" else 0

# Loan purpose (one-hot)
loan_purpose_car = 1 if loan_purpose == "Car" else 0
loan_purpose_education = 1 if loan_purpose == "Education" else 0
loan_purpose_home = 1 if loan_purpose == "Home" else 0
loan_purpose_personal = 1 if loan_purpose == "Personal" else 0

# Property area (one-hot)
property_semiurban = 1 if property_area == "Semiurban" else 0
property_urban = 1 if property_area == "Urban" else 0

# Employer category (one-hot)
employer_government = 1 if employer_category == "Government" else 0
employer_mnc = 1 if employer_category == "MNC" else 0
employer_private = 1 if employer_category == "Private" else 0
employer_unemployed = 1 if employer_category == "Unemployed" else 0

# ---------------- FINAL FEATURE VECTOR ---------------- #
features = np.array([[
    applicant_income,
    coapplicant_income,
    age,
    dependents,
    existing_loans,
    savings,
    collateral_value,
    loan_amount,
    loan_term,
    education_level,
    employment_salaried,
    employment_self_employed,
    employment_unemployed,
    marital_single,
    loan_purpose_car,
    loan_purpose_education,
    loan_purpose_home,
    loan_purpose_personal,
    property_semiurban,
    property_urban,
    gender_male,
    employer_government,
    employer_mnc,
    employer_private,
    employer_unemployed,
    dti_ratio_sq,
    credit_score_sq
]])

# ---------------- PREDICTION ---------------- #
if st.button("Predict Loan Approval"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
