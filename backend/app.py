import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching for Performance ---
@st.cache_data
def load_assets():
    """Load the saved model, scaler, and create a SHAP explainer."""
    model_path = os.path.join(r"D:\Satvik\Projects\College\Minor\Code\backend", "models", "best_heart_disease_classifier.joblib")
    scaler_path = os.path.join(r"D:\Satvik\Projects\College\Minor\Code\backend", "models", "scaler.joblib")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    explainer = shap.TreeExplainer(model)
    
    return model, scaler, explainer

# Load all assets
model, scaler, explainer = load_assets()

# --- Application Title and Description ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This app predicts the likelihood of mortality from heart failure based on patient clinical data.
Please enter the patient's information in the sidebar to get a prediction and a detailed explanation.
This tool is for educational purposes only and is not a substitute for professional medical advice.
""")

st.divider()

# --- Sidebar for User Input ---
st.sidebar.header("Patient Information")

def get_user_input():
    """Get user input from the sidebar."""
    age = st.sidebar.slider('Age', 40, 95, 60)
    anaemia = st.sidebar.selectbox('Anaemia', ('No', 'Yes'))
    creatinine_phosphokinase = st.sidebar.number_input('Creatinine Phosphokinase (mcg/L)', min_value=23, max_value=7861, value=582)
    diabetes = st.sidebar.selectbox('Diabetes', ('No', 'Yes'))
    ejection_fraction = st.sidebar.slider('Ejection Fraction (%)', 14, 80, 38)
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', ('No', 'Yes'))
    platelets = st.sidebar.number_input('Platelets (kiloplatelets/mL)', min_value=25100, max_value=850000, value=263358)
    serum_creatinine = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.4)
    serum_sodium = st.sidebar.slider('Serum Sodium (mEq/L)', 113, 148, 137)
    sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
    smoking = st.sidebar.selectbox('Smoking', ('No', 'Yes'))
    time = st.sidebar.slider('Follow-up Period (days)', 4, 285, 130)

    data = {
        'age': age,
        'anaemia': 1 if anaemia == 'Yes' else 0,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': 1 if high_blood_pressure == 'Yes' else 0,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': 1 if sex == 'Male' else 0,
        'smoking': 1 if smoking == 'Yes' else 0,
        'time': time
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get the user input
input_df = get_user_input()

# --- Display User Input ---
st.header("Patient Data Summary")
col1, col2 = st.columns(2)
patient_data = input_df.iloc[0]
display_labels = {
    'age': 'Age', 'anaemia': 'Anaemia', 'creatinine_phosphokinase': 'Creatinine Phosphokinase (mcg/L)',
    'diabetes': 'Diabetes', 'ejection_fraction': 'Ejection Fraction (%)', 'high_blood_pressure': 'High Blood Pressure',
    'platelets': 'Platelets (kiloplatelets/mL)', 'serum_creatinine': 'Serum Creatinine (mg/dL)',
    'serum_sodium': 'Serum Sodium (mEq/L)', 'sex': 'Sex', 'smoking': 'Smoking', 'time': 'Follow-up Period (days)'
}
with col1:
    st.markdown(f"**{display_labels['age']}:** `{patient_data['age']}`")
    st.markdown(f"**{display_labels['sex']}:** `{'Male' if patient_data['sex'] == 1 else 'Female'}`")
    st.markdown(f"**{display_labels['ejection_fraction']}:** `{patient_data['ejection_fraction']}` %")
    st.markdown(f"**{display_labels['serum_creatinine']}:** `{patient_data['serum_creatinine']}` mg/dL")
    st.markdown(f"**{display_labels['serum_sodium']}:** `{patient_data['serum_sodium']}` mEq/L")
    st.markdown(f"**{display_labels['time']}:** `{patient_data['time']}` days")
with col2:
    st.markdown(f"**{display_labels['anaemia']}:** `{'Yes' if patient_data['anaemia'] == 1 else 'No'}`")
    st.markdown(f"**{display_labels['diabetes']}:** `{'Yes' if patient_data['diabetes'] == 1 else 'No'}`")
    st.markdown(f"**{display_labels['high_blood_pressure']}:** `{'Yes' if patient_data['high_blood_pressure'] == 1 else 'No'}`")
    st.markdown(f"**{display_labels['smoking']}:** `{'Yes' if patient_data['smoking'] == 1 else 'No'}`")
    st.markdown(f"**{display_labels['creatinine_phosphokinase']}:** `{patient_data['creatinine_phosphokinase']}` mcg/L")
    st.markdown(f"**{display_labels['platelets']}:** `{int(patient_data['platelets'])}`")

st.divider()

# --- SHAP Plot Display Function ---
def st_shap(plot, height=None):
    """A wrapper to display SHAP plots in Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# --- Prediction Logic and SHAP Explanation ---
# This entire block should replace your existing one.
if st.button("🔍 Predict Heart Disease Risk"):
    # Define feature order (must match training)
    feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
                     'ejection_fraction', 'high_blood_pressure', 'platelets', 
                     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
    input_df_ordered = input_df[feature_order]
    
    # Scale the user input and convert back to DataFrame
    scaled_input_np = scaler.transform(input_df_ordered)
    scaled_input_df = pd.DataFrame(scaled_input_np, columns=feature_order)
    
    # --- Make Prediction ---
    prediction = model.predict(scaled_input_df)
    prediction_proba = model.predict_proba(scaled_input_df)
    
    # --- Display Prediction ---
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.success("The model predicts a **LOW** risk of mortality from heart failure.")
        st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
    else:
        st.error("The model predicts a **HIGH** risk of mortality from heart failure.")
        st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")

    st.divider()

    # --- SHAP Explanation ---
    st.subheader("Prediction Explanation")
    st.markdown("The plot below shows how each feature contributed to the final prediction.")
    
    # --- Using the Waterfall Plot (More Robust) ---
    
    # 1. Use the explainer to get the Explanation object.
    explanation = explainer(scaled_input_df)
    
    # 2. Select the explanation for our first sample and the positive class (High Risk).
    explanation_for_class_1 = explanation[0, :, 1]

    # 3. Create a waterfall plot. This is a native Matplotlib plot and is very reliable.
    #    We create a figure first to display it in Streamlit.
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation_for_class_1, show=False)
    plt.tight_layout() # Adjust layout to make sure everything fits
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

st.sidebar.divider()
st.sidebar.info(
    "This is a demo application. The underlying model is a tuned Random Forest classifier. "
    "Model explainability is provided by SHAP."
)