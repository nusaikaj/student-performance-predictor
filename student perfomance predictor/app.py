# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained Random Forest model
rf_model = joblib.load("models/student_model.pkl")

# Risk level function
def risk_level(prob):
    if prob >= 0.75:
        return "Low Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "High Risk"

# Title
st.title("Student Performance Predictor")
st.write("Predict whether a student will PASS or FAIL and their risk level")

# Input fields
study_time = st.number_input("Study Time (hours per day)", min_value=0, max_value=24, value=3)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=70)
previous_marks = st.number_input("Previous Marks (%)", min_value=0, max_value=100, value=50)
internet = st.selectbox("Internet Access", ["Yes", "No"])
extra_classes = st.selectbox("Extra Classes", ["Yes", "No"])

# Convert categorical inputs to numeric
internet_num = 1 if internet == "Yes" else 0
extra_classes_num = 1 if extra_classes == "Yes" else 0

# Predict button
if st.button("Predict"):
    new_student = np.array([[study_time, attendance, previous_marks, internet_num, extra_classes_num]])
    prob = rf_model.predict_proba(new_student)[0][1]
    prediction = rf_model.predict(new_student)[0]

    st.write(f"**Pass Probability:** {round(prob, 2)}")
    st.write(f"**Risk Level:** {risk_level(prob)}")
    
    if prediction == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")
