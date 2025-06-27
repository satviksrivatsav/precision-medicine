import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Precision Medicine Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Asset Loading (Handles multiple models) ---
@st.cache_data
def load_assets(disease_name):
    """Load the model, scaler, and explainer for the selected disease."""
    # Construct paths relative to the script location
    base_path = os.path.join("models", disease_name)
    model_path = os.path.join(base_path, f"best_{disease_name}_classifier.joblib")
    scaler_path = os.path.join(base_path, f"{disease_name}_scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, explainer

# --- Main App Logic ---

# Title for the whole app
st.title("🔬 Precision Medicine Prediction Suite")

# --- Sidebar Model Selection ---
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Heart Disease", "Diabetes")
)

# --- Dynamic UI based on Model Choice ---

if model_choice == 'Heart Disease':
    st.header("❤️ Heart Disease Mortality Prediction")
    
    # Load Heart Disease Assets
    model, scaler, explainer = load_assets('heart_disease')
    
    # Sidebar Inputs for Heart Disease
    st.sidebar.header("Patient Information")
    age = st.sidebar.slider('Age', 40, 95, 60)
    anaemia = st.sidebar.selectbox('Anaemia', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    creatinine_phosphokinase = st.sidebar.number_input('Creatinine Phosphokinase (mcg/L)', 23, 7861, 582)
    diabetes_h = st.sidebar.selectbox('Diabetes', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 14, 80, 38)
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    platelets = st.sidebar.number_input('Platelets (kiloplatelets/mL)', 25100, 850000, 263358)
    serum_creatinine = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.4)
    serum_sodium = st.sidebar.slider('Serum Sodium (mEq/L)', 113, 148, 137)
    sex = st.sidebar.selectbox('Sex', (0, 1), format_func=lambda x: 'Male' if x == 1 else 'Female')
    smoking = st.sidebar.selectbox('Smoking', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    time = st.sidebar.slider('Follow-up Period (days)', 4, 285, 130)

    # Create the input DataFrame
    data = {'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase,
            'diabetes': diabetes_h, 'ejection_fraction': ejection_fraction,
            'high_blood_pressure': high_blood_pressure, 'platelets': platelets,
            'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium, 'sex': sex,
            'smoking': smoking, 'time': time}
    input_df = pd.DataFrame(data, index=[0])
    
    # Define feature order for the model
    feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

elif model_choice == 'Diabetes':
    st.header("🩸 Diabetes Risk Prediction")
    
    # Add a disclaimer about the dataset's limitations
    st.info("ℹ️ **Note:** This model is trained on the PIMA Indians Diabetes Dataset, which includes only female patients of Pima Indian heritage aged 21 and older.")

    # Load Diabetes Assets
    model, scaler, explainer = load_assets('diabetes')

    # Sidebar Inputs for Diabetes
    st.sidebar.header("Patient Information")
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BloodPressure = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    SkinThickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 30)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0, step=0.1)
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

    # Create the input DataFrame
    data = {'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}
    input_df = pd.DataFrame(data, index=[0])

    # Define feature order for the model
    feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                     'BMI', 'DiabetesPedigreeFunction', 'Age']

# --- Common UI elements (display input and button) ---

st.subheader("Patient Data Summary")

# Dynamically display the input data in a user-friendly format
col1, col2 = st.columns(2)
patient_data = input_df.iloc[0]

# Split the features between the two columns
features_list = feature_order
mid_point = len(features_list) // 2 + len(features_list) % 2
features_col1 = features_list[:mid_point]
features_col2 = features_list[mid_point:]

with col1:
    for feature in features_col1:
        st.markdown(f"**{feature.replace('_', ' ').title()}:** `{patient_data[feature]}`")

with col2:
    for feature in features_col2:
        st.markdown(f"**{feature.replace('_', ' ').title()}:** `{patient_data[feature]}`")
        
st.divider()

if st.button(f"🔍 Predict {model_choice} Risk"):
    # Ensure columns are in the correct order
    input_df_ordered = input_df[feature_order]
    
    # Scale the input
    scaled_input_np = scaler.transform(input_df_ordered)
    scaled_input_df = pd.DataFrame(scaled_input_np, columns=feature_order)
    
    # Make prediction
    prediction = model.predict(scaled_input_df)
    prediction_proba = model.predict_proba(scaled_input_df)
    
    # Display prediction result (dynamically)
    st.subheader("Prediction Result")
    positive_class_text = "mortality from heart failure" if model_choice == 'Heart Disease' else "diabetes"
    
    if prediction[0] == 0:
        st.success(f"The model predicts a **LOW** risk of {positive_class_text}.")
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
    else:
        st.error(f"The model predicts a **HIGH** risk of {positive_class_text}.")
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")

    st.divider()

    # SHAP Explanation
    st.subheader("Prediction Explanation")
    st.markdown("The waterfall plot below shows how each feature contributed to the final prediction.")
    
    explanation = explainer(scaled_input_df)
    
    # Handle multi-class vs single-output from SHAP explainer
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        explanation_to_plot = explanation[0, :, 1]
    else:
        explanation_to_plot = explanation[0]

    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation_to_plot, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)