import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Precision Medicine Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Asset Loading ---
@st.cache_data
def load_assets(model_name):
    """Load all assets for the selected model."""
    base_path = os.path.join("models", model_name)
    assets = {}
    assets['model'] = joblib.load(os.path.join(base_path, f"best_{model_name}_classifier.joblib"))
    assets['scaler'] = joblib.load(os.path.join(base_path, f"{model_name}_scaler.joblib"))
    
    feature_path = os.path.join(base_path, f"{model_name}_features.joblib")
    if os.path.exists(feature_path):
        assets['features'] = joblib.load(feature_path)
    else: # Fallback for simple models
        # This gets the feature names from the scaler if a feature list wasn't saved
        if hasattr(assets['scaler'], 'feature_names_in_'):
            assets['features'] = assets['scaler'].feature_names_in_
        
    assets['explainer'] = shap.TreeExplainer(assets['model'])
    return assets

# --- UI and Logic ---
st.title("🔬 Precision Medicine Prediction Suite")
st.sidebar.title("Configuration")

model_choice_name = st.sidebar.selectbox(
    "Choose a Prediction Model",
    ("Heart Disease", "Diabetes", "Lung Cancer", "Kidney Disease")
)

model_map = {"Heart Disease": "heart_disease", "Diabetes": "diabetes", "Lung Cancer": "lung_cancer", "Kidney Disease": "kidney_disease"}
model_dir_name = model_map[model_choice_name]
assets = load_assets(model_dir_name)

st.sidebar.header("Patient Information")
data = {}

# --- DYNAMIC UI GENERATION ---
if model_choice_name == 'Heart Disease':
    st.header("❤️ Heart Disease Mortality Prediction")
    feature_order = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']
    # Simplified UI creation
    data['age'] = st.sidebar.slider('Age', 40, 95, 60, key='hd_age')
    data['ejection_fraction'] = st.sidebar.slider('Ejection Fraction (%)', 14, 80, 38, key='hd_ef')
    data['serum_creatinine'] = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.4, key='hd_sc')
    data['time'] = st.sidebar.slider('Follow-up Period (days)', 4, 285, 130, key='hd_time')
    # Use defaults for other features for a cleaner UI
    
elif model_choice_name == 'Diabetes':
    st.header("🩸 Diabetes Risk Prediction")
    feature_order = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    sex_d = st.sidebar.selectbox('Sex', ('Female', 'Male'), key='d_sex')
    data['Pregnancies'] = st.sidebar.slider('Pregnancies', 0, 17, 3, key='d_preg') if sex_d == 'Female' else 0
    if sex_d == 'Male': st.sidebar.text('Pregnancies: 0 (Set for male patient)')
    data['Glucose'] = st.sidebar.slider('Glucose', 44, 199, 117, key='d_glucose')
    data['BMI'] = st.sidebar.slider('BMI', 18.2, 67.1, 32.0, step=0.1, key='d_bmi')
    data['Age'] = st.sidebar.slider('Age', 21, 81, 29, key='d_age')
    
elif model_choice_name == 'Lung Cancer':
    st.header("🫁 Lung Cancer Prediction")
    feature_order = assets['features']
    data['Age'] = st.sidebar.slider('Age', 20, 90, 65, key='lc_age')
    data['Sex_Male'] = 1 if st.sidebar.selectbox('Sex', ('Female', 'Male'), key='lc_sex') == 'Male' else 0
    st.sidebar.subheader("Gene Mutations")
    for feature in feature_order:
        if feature not in ['Age', 'Sex_Male']:
            data[feature] = 1 if st.sidebar.checkbox(f'Mutation in {feature.replace("_", " ")}', key=f'lc_{feature}') else 0

elif model_choice_name == 'Kidney Disease':
    st.header("🩺 Chronic Kidney Disease Prediction")
    
    # The model was trained on only these 13 features (after dropping leaky features)
    feature_order = ['age', 'bp', 'bgr', 'bu', 'sod', 'pot', 'wc', 'htn_yes', 'dm_yes', 'cad_yes', 'appet_poor', 'pe_yes', 'ane_yes']
    
    st.info("Enter patient information for CKD risk assessment. This model uses key predictive features while avoiding diagnostic markers.")
    
    # Collect only the features that the model was actually trained on
    data['age'] = st.sidebar.slider('Age', 2, 90, 50, key='kd_age')
    data['bp'] = st.sidebar.slider('Blood Pressure (mm/Hg)', 50, 180, 80, key='kd_bp')
    data['bgr'] = st.sidebar.slider('Blood Glucose Random (mgs/dl)', 22, 490, 148, key='kd_bgr')
    data['bu'] = st.sidebar.slider('Blood Urea (mgs/dl)', 1.5, 391.0, 57.0, step=0.1, key='kd_bu')
    data['sod'] = st.sidebar.slider('Sodium (mEq/L)', 4.5, 163.0, 137.0, step=0.1, key='kd_sod')
    data['pot'] = st.sidebar.slider('Potassium (mEq/L)', 2.5, 47.0, 4.6, step=0.1, key='kd_pot')
    data['wc'] = st.sidebar.slider('White Blood Cell Count (cells/cumm)', 2200, 26400, 7800, step=100, key='kd_wc')
    
    # Categorical features (medical history)
    data['htn_yes'] = 1 if st.sidebar.selectbox('Hypertension', ('No', 'Yes'), key='kd_htn') == 'Yes' else 0
    data['dm_yes'] = 1 if st.sidebar.selectbox('Diabetes Mellitus', ('No', 'Yes'), key='kd_dm') == 'Yes' else 0
    data['cad_yes'] = 1 if st.sidebar.selectbox('Coronary Artery Disease', ('No', 'Yes'), key='kd_cad') == 'Yes' else 0
    data['appet_poor'] = 1 if st.sidebar.selectbox('Appetite', ('Good', 'Poor'), key='kd_appet') == 'Poor' else 0
    data['pe_yes'] = 1 if st.sidebar.selectbox('Pedal Edema', ('No', 'Yes'), key='kd_pe') == 'Yes' else 0
    data['ane_yes'] = 1 if st.sidebar.selectbox('Anemia', ('No', 'Yes'), key='kd_ane') == 'Yes' else 0

# --- Feature Name Mappings for Better Display ---
feature_name_mappings = {
    # Heart Disease
    'age': 'Age',
    'anaemia': 'Anemia',
    'creatinine_phosphokinase': 'Creatinine Phosphokinase',
    'diabetes': 'Diabetes',
    'ejection_fraction': 'Ejection Fraction (%)',
    'high_blood_pressure': 'High Blood Pressure',
    'platelets': 'Platelets',
    'serum_creatinine': 'Serum Creatinine',
    'serum_sodium': 'Serum Sodium',
    'sex': 'Sex',
    'smoking': 'Smoking',
    'time': 'Follow-up Time (days)',
    
    # Diabetes
    'Pregnancies': 'Pregnancies',
    'Glucose': 'Glucose Level',
    'BloodPressure': 'Blood Pressure',
    'SkinThickness': 'Skin Thickness',
    'Insulin': 'Insulin Level',
    'BMI': 'Body Mass Index',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree Function',
    'Age': 'Age',
    
    # Kidney Disease
    'bp': 'Blood Pressure',
    'bgr': 'Blood Glucose Random',
    'bu': 'Blood Urea',
    'sod': 'Sodium Level',
    'pot': 'Potassium Level',
    'wc': 'White Blood Cell Count',
    'htn_yes': 'Hypertension',
    'dm_yes': 'Diabetes Mellitus',
    'cad_yes': 'Coronary Artery Disease',
    'appet_poor': 'Poor Appetite',
    'pe_yes': 'Pedal Edema',
    'ane_yes': 'Anemia',
    
    # Lung Cancer (gene mutations)
    'Age': 'Age',
    'Sex_Male': 'Male Gender'
}

# --- PREDICTION LOGIC ---
st.divider()

if st.button(f"🔍 Predict {model_choice_name} Risk"):
    
    # 1. Create a dataframe from the UI inputs
    input_df = pd.DataFrame([data])
    
    # 2. Create the final dataframe with all the columns the model expects, in the correct order, filling missing ones with 0
    final_df = input_df.reindex(columns=feature_order, fill_value=0)

    # 3. Get the scaler and the columns it was trained on
    scaler = assets['scaler']
    cols_to_scale = scaler.feature_names_in_
    
    # 4. Scale only the necessary columns
    final_df[cols_to_scale] = scaler.transform(final_df[cols_to_scale])

    # 5. Predict using the correctly formatted and scaled DataFrame
    model = assets['model']
    prediction = model.predict(final_df)
    prediction_proba = model.predict_proba(final_df)
    
    # --- Display Prediction and Explanation ---
    st.subheader("Prediction Result")
    # ... display logic ...
    positive_class_text_map = {"Heart Disease": "mortality from heart failure", "Diabetes": "diabetes", "Lung Cancer": "mortality from lung cancer", "Kidney Disease": "Chronic Kidney Disease"}
    positive_class_text = positive_class_text_map[model_choice_name]
    if prediction[0] == 0: st.success(f"The model predicts a **LOW** risk of {positive_class_text}.")
    else: st.error(f"The model predicts a **HIGH** risk of {positive_class_text}.")
    st.write(f"**Confidence:** {prediction_proba[0][1]*100 if prediction[0] == 1 else prediction_proba[0][0]*100:.2f}%")
        
    st.divider()
    st.subheader("Prediction Explanation")
    
    explainer = assets['explainer']
    explanation = explainer(final_df)
    
    # ... SHAP display logic ...
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        explanation_to_plot = explanation[0, :, 1]
    else:
        explanation_to_plot = explanation[0]

    # Map feature names to readable names
    readable_feature_names = []
    for feature in explanation_to_plot.feature_names:
        if feature in feature_name_mappings:
            readable_feature_names.append(feature_name_mappings[feature])
        else:
            # For lung cancer gene mutations, clean up the name
            if feature.startswith('EGFR') or feature.startswith('KRAS') or 'mutation' in feature.lower():
                readable_feature_names.append(feature.replace('_', ' ').title() + ' Mutation')
            else:
                readable_feature_names.append(feature.replace('_', ' ').title())
    
    # Modify the feature names directly in the explanation object
    explanation_to_plot.feature_names = readable_feature_names

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation_to_plot, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)