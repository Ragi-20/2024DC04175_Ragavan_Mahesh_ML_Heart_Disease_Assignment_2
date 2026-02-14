import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------
# Load Models
# -------------------------------------------------
log_reg = joblib.load("model/logistic_regression.pkl")
decision_tree = joblib.load("model/decision_tree.pkl")
knn = joblib.load("model/knn.pkl")
naive_bayes = joblib.load("model/naive_bayes.pkl")
random_forest = joblib.load("model/random_forest.pkl")
xgboost_model = joblib.load("model/xgboost.pkl")
scaler = joblib.load("model/scaler.pkl")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("This application predicts heart disease using multiple machine learning models.")

# -------------------------------------------------
# Model Selection (Requirement b)
# -------------------------------------------------
model_name = st.selectbox(
    "Select a Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ),
)

if model_name == "Logistic Regression":
    model = log_reg
elif model_name == "Decision Tree":
    model = decision_tree
elif model_name == "KNN":
    model = knn
elif model_name == "Naive Bayes":
    model = naive_bayes
elif model_name == "Random Forest":
    model = random_forest
else:
    model = xgboost_model

# -------------------------------------------------
# Manual Input Prediction
# -------------------------------------------------
st.subheader("Manual Patient Input")

age = st.number_input("Age", 1, 120, 50)
sex = st.number_input("Sex (0 = Female, 1 = Male)", 0, 1, 1)
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.number_input("Fasting Blood Sugar (0 or 1)", 0, 1, 0)
restecg = st.number_input("Resting ECG (0-2)", 0, 2, 1)
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.number_input("Exercise Induced Angina (0 or 1)", 0, 1, 0)
oldpeak = st.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.number_input("Thalassemia (0-3)", 0, 3, 2)

if st.button("Predict Individual Result"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Probability of Heart Disease: {probability[0][1] * 100:.2f}%")
    st.write(f"Model Used: {model_name}")

# -------------------------------------------------
# CSV Upload Section (Requirement a)
# -------------------------------------------------
st.markdown("---")
st.subheader("Upload Test Dataset (CSV) for Evaluation")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("CSV must contain 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        # -------------------------------------------------
        # Evaluation Metrics (Requirement c)
        # -------------------------------------------------
        st.subheader("Evaluation Metrics")

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        st.write(f"Accuracy: {accuracy:.4f}")
        st.write(f"Precision: {precision:.4f}")
        st.write(f"Recall: {recall:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

        # -------------------------------------------------
        # Confusion Matrix (Requirement d)
        # -------------------------------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        ax.matshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{val}", ha='center', va='center')

        st.pyplot(fig)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
    ### 👨‍💻 Developed By  
    **Ragavan Mahesh**  
    Roll Number: 2024DC04175  
    Course: Machine Learning  
    """
)