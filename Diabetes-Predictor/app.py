import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model/diabetes_model.pkl")

# Page title
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the details below to predict whether the person is diabetic or not.")

# 👉 Sidebar layout
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3875/3875186.png", width=100)
    st.title("Input Features")

    pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
    age = st.slider("Age", 10, 100, 30)

# 🎯 Prediction logic
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result:")
    st.success(f"The person is **{result}**")
    st.info(f"Model confidence: {proba:.2%}")
