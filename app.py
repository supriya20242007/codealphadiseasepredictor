import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    with open("heart_disease_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    # Fit scaler using the training dataset features exactly as in the notebook
    df = pd.read_csv("heart_cleveland_upload.csv")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    X = df.drop("condition", axis=1)
    
    scaler = StandardScaler()
    scaler.fit(X)
    
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model or dataset: {e}")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient clinical metrics below to predict the risk of heart disease.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=41)
    sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[
        ("Typical Angina (0)", 0),
        ("Atypical Angina (1)", 1),
        ("Non-anginal Pain (2)", 2),
        ("Asymptomatic (3)", 3)
    ], format_func=lambda x: x[0])[1]
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=130)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=204)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("False (0)", 0), ("True (1)", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG Results", options=[
        ("Normal (0)", 0),
        ("ST-T Wave Abnormality (1)", 1),
        ("Left Ventricular Hypertrophy (2)", 2)
    ], format_func=lambda x: x[0])[1]

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=230, value=172)
    exang = st.selectbox("Exercise Induced Angina", options=[("No (0)", 0), ("Yes (1)", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[
        ("Upsloping (0)", 0),
        ("Flat (1)", 1),
        ("Downsloping (2)", 2)
    ], format_func=lambda x: x[0])[1]
    ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    thal = st.selectbox("Thalassemia", options=[
        ("Normal (0)", 0),
        ("Fixed Defect (1)", 1),
        ("Reversible Defect (2)", 2)
    ], format_func=lambda x: x[0])[1]

st.markdown("---")

if st.button("Predict Heart Disease Risk", use_container_width=True):
    raw_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # Apply standard scaling identical to training notebook
    scaled_features = scaler.transform(raw_features)
    
    try:
        prediction = model.predict(scaled_features)
        
        # Notebook Mapping: 0 = No Disease (Low Risk), 1 = Disease (High Risk)
        if prediction[0] == 1:
            st.error("⚠️ High Risk: The model predicts the presence of heart disease.")
        else:
            st.success("✅ Low Risk: The model predicts no presence of heart disease.")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")