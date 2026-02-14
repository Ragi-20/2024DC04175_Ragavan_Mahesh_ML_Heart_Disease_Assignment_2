import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Models and Scaler
# -----------------------------
log_reg = joblib.load("model/logistic_regression.pkl")
knn = joblib.load("model/knn.pkl")
nb = joblib.load("model/naive_bayes.pkl")
dt = joblib.load("model/decision_tree.pkl")
rf = joblib.load("model/random_forest.pkl")
xgb = joblib.load("model/xgboost.pkl")
scaler = joblib.load("model/scaler.pkl")

models = {
    "Logistic Regression": log_reg,
    "KNN": knn,
    "Naive Bayes": nb,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("This app predicts the likelihood of heart disease using multiple machine learning models.")

# Model selection
model_name = st.selectbox("Select a Model", list(models.keys()))
model = models[model_name]

# Feature names (must match training dataset)
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

st.subheader("Enter Patient Details")

user_input = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    user_input.append(value)

if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2f})")