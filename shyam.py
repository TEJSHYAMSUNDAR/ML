import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
model_filename = "C:/Users/nirma/Downloads/archive (1)/logistic_regression_medical_conditions_model.pkl"
model = joblib.load(model_filename)

# Load scaler (ensure you save and load the same scaler used for training)
scaler = StandardScaler()

# Streamlit App Title
st.title("Medical Condition Prediction")

# Input features from the user
age = st.number_input("Age", min_value=0, max_value=120, value=25)
gender = st.selectbox("Gender", ("Male", "Female"))
smoking_status = st.selectbox("Smoking Status", ("Smoker", "Non-Smoker"))
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=120.0)
glucose_levels = st.number_input("Glucose Levels", min_value=0.0, max_value=500.0, value=100.0)

# Encoding categorical variables (the same way it was done in the training)
gender_encoded = 1 if gender == "Male" else 0
smoking_status_encoded = 0 if smoking_status == "Non-Smoker" else 1

# Prepare the input data for the model (ensure it's in the same format as the training data)
input_data = np.array([[age, gender_encoded, smoking_status_encoded, bmi, blood_pressure, glucose_levels]])

# Apply scaling to the input data
input_data_scaled = scaler.fit_transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    condition = "Diabetic" if prediction == 1 else "Pneumonia"
    
    st.write(f"The predicted medical condition is: {condition}")
