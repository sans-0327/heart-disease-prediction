import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------------
# Load the trained model, scaler, and feature names
# ---------------------------------------------------------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict the likelihood of heart disease.")

st.header("Patient Information")

# ---------------------------------------------------------------
# Basic details
# ---------------------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    options=[("Male", 1), ("Female", 0)],
    format_func=lambda x: x[0]
)[1]

# ---------------------------------------------------------------
# Chest pain & vitals
# ---------------------------------------------------------------
cp = st.selectbox(
    "Chest Pain Type",
    options=[
        ("Typical angina", 0),
        ("Atypical angina", 1),
        ("Non-anginal pain", 2),
        ("Asymptomatic (no pain)", 3)
    ],
    format_func=lambda x: x[0]
)[1]
st.caption("The type of chest pain experienced, based on clinical description.")

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
st.caption("Blood pressure measured while at rest.")

chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
st.caption("Serum cholesterol level from a blood test.")

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl?",
    options=[("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]
st.caption("Whether fasting blood sugar is elevated — a marker linked to diabetes risk.")

restecg = st.selectbox(
    "Resting ECG Result",
    options=[
        ("Normal", 0),
        ("ST-T wave abnormality", 1),
        ("Left ventricular hypertrophy", 2)
    ],
    format_func=lambda x: x[0]
)[1]
st.caption("Results from a resting electrocardiogram (heart electrical activity test).")

# ---------------------------------------------------------------
# Exercise test results
# ---------------------------------------------------------------
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
st.caption("The highest heart rate reached during an exercise stress test.")

exang = st.selectbox(
    "Exercise-Induced Angina?",
    options=[("No", 0), ("Yes", 1)],
    format_func=lambda x: x[0]
)[1]
st.caption("Whether chest pain occurred during physical exercise.")

oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st.caption("A measurement from an ECG stress test — higher values may indicate reduced blood flow to the heart.")

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[
        ("Upsloping", 0),
        ("Flat", 1),
        ("Downsloping", 2)
    ],
    format_func=lambda x: x[0]
)[1]
st.caption("The shape of the ECG curve during peak exercise.")

# ---------------------------------------------------------------
# Advanced diagnostic tests
# ---------------------------------------------------------------
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
st.caption("Number of major blood vessels showing blockage, detected via a fluoroscopy scan. Higher values generally indicate more severe disease.")

thal = st.selectbox(
    "Thalassemia Test Result",
    options=[
        ("Normal", 1),
        ("Fixed defect", 2),
        ("Reversible defect", 3)
    ],
    format_func=lambda x: x[0]
)[1]
st.caption("Result from a thallium stress test, which checks blood flow to the heart muscle.")

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
st.divider()

if st.button("Predict"):
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]],
                                columns=feature_names)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ The model predicts this patient LIKELY HAS heart disease.")
    else:
        st.success("✅ The model predicts this patient likely does NOT have heart disease.")

    st.write(f"Confidence: {probability[prediction]*100:.1f}%")
    st.caption("This prediction is based on a machine learning model trained on a limited dataset and should not be used for actual diagnosis.")