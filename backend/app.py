import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from transformers import pipeline
import textwrap

# --- Page Configuration ---
st.set_page_config(
    page_title="Precision Medicine Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. ASSET LOADING (CACHED FOR PERFORMANCE)
# ==============================================================================

@st.cache_data
def load_prediction_assets(model_name):
    """Load all assets for a specific prediction model."""
    base_path = os.path.join("models", model_name)
    assets = {}
    assets['model'] = joblib.load(os.path.join(base_path, f"best_{model_name}_classifier.joblib"))
    assets['scaler'] = joblib.load(os.path.join(base_path, f"{model_name}_scaler.joblib"))
    feature_path = os.path.join(base_path, f"{model_name}_features.joblib")
    if os.path.exists(feature_path):
        assets['features'] = joblib.load(feature_path)
    assets['explainer'] = shap.TreeExplainer(assets['model'])
    return assets

@st.cache_resource
def load_qa_pipeline():
    """Load the Hugging Face Question Answering pipeline."""
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_data
def load_context(topic_filename):
    """Load the knowledge base text from a file."""
    path = os.path.join("knowledgeBase", topic_filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Context file '{path}' not found. Please create it."

# ==============================================================================
# 2. MAIN APP LAYOUT
# ==============================================================================

st.title("🔬 Precision Medicine & Healthcare AI Suite")

st.sidebar.title("Configuration")
tool_choice = st.sidebar.selectbox(
    "Select a Tool",
    ("Disease Prediction Suite", "Help Chatbot")
)

# ==============================================================================
# MODE 1: DISEASE PREDICTION SUITE
# ==============================================================================
if tool_choice == "Disease Prediction Suite":
    st.sidebar.divider()
    model_choice_name = st.sidebar.selectbox(
        "Choose a Prediction Model",
        ("Heart Disease", "Diabetes", "Lung Cancer", "Kidney Disease")
    )
    model_map = {
        "Heart Disease": "heart_disease", "Diabetes": "diabetes",
        "Lung Cancer": "lung_cancer", "Kidney Disease": "kidney_disease"
    }
    model_dir_name = model_map[model_choice_name]
    assets = load_prediction_assets(model_dir_name)
    st.sidebar.header("Patient Information")
    data = {}

    # --- UI Generation for Predictions ---
    # ... (This is your working UI code, with unique keys for widgets) ...
    if model_choice_name == 'Heart Disease':
        st.header("❤️ Heart Disease Mortality Prediction")
        feature_order = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
        data['age'] = st.sidebar.slider('Age', 40, 95, 60, key='hd_age')
        data['anaemia'] = st.sidebar.selectbox('Anaemia', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='hd_anaemia')
        # ... (and so on for all heart disease inputs)
        data['creatinine_phosphokinase'] = st.sidebar.number_input('Creatinine Phosphokinase (mcg/L)', 23, 7861, 582, key='hd_cpk')
        data['diabetes'] = st.sidebar.selectbox('Diabetes', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='hd_diabetes')
        data['ejection_fraction'] = st.sidebar.slider('Ejection Fraction (%)', 14, 80, 38, key='hd_ef')
        data['high_blood_pressure'] = st.sidebar.selectbox('High Blood Pressure', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='hd_hbp')
        data['platelets'] = st.sidebar.number_input('Platelets (kiloplatelets/mL)', 25100, 850000, 263358, key='hd_platelets')
        data['serum_creatinine'] = st.sidebar.slider('Serum Creatinine (mg/dL)', 0.5, 9.4, 1.4, key='hd_sc')
        data['serum_sodium'] = st.sidebar.slider('Serum Sodium (mEq/L)', 113, 148, 137, key='hd_ss')
        data['sex'] = st.sidebar.selectbox('Sex', (0, 1), format_func=lambda x: 'Male' if x == 1 else 'Female', key='hd_sex')
        data['smoking'] = st.sidebar.selectbox('Smoking', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='hd_smoking')
        data['time'] = st.sidebar.slider('Follow-up Period (days)', 4, 285, 130, key='hd_time')

    elif model_choice_name == 'Diabetes':
        st.header("🩸 Diabetes Risk Prediction")
        feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        sex_d = st.sidebar.selectbox('Sex', ('Female', 'Male'), key='d_sex')
        data['Pregnancies'] = st.sidebar.slider('Pregnancies', 0, 17, 3, key='d_preg') if sex_d == 'Female' else 0
        if sex_d == 'Male': st.sidebar.text('Pregnancies: 0 (for male patient)')
        data['Glucose'] = st.sidebar.slider('Glucose', 0, 199, 117, key='d_glucose')
        data['BloodPressure'] = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72, key='d_bp')
        data['SkinThickness'] = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 23, key='d_skin')
        data['Insulin'] = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 30, key='d_insulin')
        data['BMI'] = st.sidebar.slider('BMI', 0.0, 67.1, 32.0, step=0.1, key='d_bmi')
        data['DiabetesPedigreeFunction'] = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, key='d_dpf')
        data['Age'] = st.sidebar.slider('Age', 21, 81, 29, key='d_age')

    elif model_choice_name in ['Lung Cancer', 'Kidney Disease']:
        header_text = "🫁 Lung Cancer Survival Prediction" if model_choice_name == 'Lung Cancer' else "երի Chronic Kidney Disease (CKD) Prediction"
        st.header(header_text)
        feature_order = assets['features']
        if model_choice_name == 'Lung Cancer':
            data['Age'] = st.sidebar.slider('Age', 20, 90, 65, key='lc_age')
            data['Sex_Male'] = 1 if st.sidebar.selectbox('Sex', ('Female', 'Male'), key='lc_sex') == 'Male' else 0
            st.sidebar.subheader("Gene Mutations")
            for feature in feature_order:
                if feature not in ['Age', 'Sex_Male']:
                    data[feature] = 1 if st.sidebar.checkbox(f'Mutation in {feature.replace("_", " ")}', key=f'lc_{feature}') else 0
        else: # Kidney Disease
            st.info("Showing a subset of key features for this demo.")
            data['age'] = st.sidebar.slider('Age', 2, 90, 55, key='kd_age')
            data['bp'] = st.sidebar.slider('Blood Pressure (mm/Hg)', 50, 180, 80, key='kd_bp')
            data['bgr'] = st.sidebar.slider('Blood Glucose Random (mgs/dl)', 22, 490, 148, key='kd_bgr')
            data['bu'] = st.sidebar.slider('Blood Urea (mgs/dl)', 1.5, 391.0, 57.0, step=0.1, key='kd_bu')
            data['htn_yes'] = 1 if st.sidebar.selectbox('Hypertension', ('No', 'Yes'), key='kd_htn') == 'Yes' else 0
            data['dm_yes'] = 1 if st.sidebar.selectbox('Diabetes Mellitus', ('No', 'Yes'), key='kd_dm') == 'Yes' else 0

    input_df = pd.DataFrame(data, index=[0])

    # --- Common Prediction UI elements ---
    st.subheader("Patient Data Summary")
    # ... (Dynamic 2-column display) ...
    col1, col2 = st.columns(2)
    display_features = list(data.keys())
    mid_point = len(display_features) // 2 + len(display_features) % 2
    with col1:
        for feature in display_features[:mid_point]: st.markdown(f"**{feature.replace('_', ' ').title()}:** `{input_df[feature].iloc[0]}`")
    with col2:
        for feature in display_features[mid_point:]: st.markdown(f"**{feature.replace('_', ' ').title()}:** `{input_df[feature].iloc[0]}`")
    
    st.divider()

    if st.button(f"🔍 Predict {model_choice_name} Risk"):
        # ... (This is your working prediction logic) ...
        input_df_ordered = input_df.reindex(columns=feature_order, fill_value=0)
        scaler = assets['scaler']
        cols_to_scale = []
        if hasattr(scaler, 'feature_names_in_'): cols_to_scale = scaler.feature_names_in_
        scaled_input_df = input_df_ordered.copy()
        if len(cols_to_scale) > 0:
            cols_that_exist = [col for col in cols_to_scale if col in scaled_input_df.columns]
            scaled_input_df[cols_that_exist] = scaler.transform(scaled_input_df[cols_that_exist])
        
        prediction = assets['model'].predict(scaled_input_df)
        prediction_proba = assets['model'].predict_proba(scaled_input_df)
        
        st.subheader("Prediction Result")
        text_map = {"Heart Disease": "mortality from heart failure", "Diabetes": "diabetes", "Lung Cancer": "mortality from lung cancer", "Kidney Disease": "Chronic Kidney Disease"}
        positive_text = text_map[model_choice_name]
        if prediction[0] == 0:
            st.success(f"The model predicts a **LOW** risk of {positive_text}.")
            st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
        else:
            st.error(f"The model predicts a **HIGH** risk of {positive_text}.")
            st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
        
        st.divider()
        st.subheader("Prediction Explanation")
        explanation = assets['explainer'](scaled_input_df)
        if isinstance(assets['explainer'].expected_value, (list, np.ndarray)) and len(assets['explainer'].expected_value) > 1:
            explanation_to_plot = explanation[0, :, 1]
        else:
            explanation_to_plot = explanation[0]
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation_to_plot, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==============================================================================
# MODE 2: HELP CHATBOT
# ==============================================================================
elif tool_choice == "Help Chatbot":
    st.header("🤖 Help & Information Q&A Bot")
    st.warning("⚠️ **Disclaimer:** This bot provides information from pre-loaded texts. It is not a live medical professional.")
    
    qa_pipeline = load_qa_pipeline()

    st.sidebar.divider()
    st.sidebar.header("Help Topics")
    topic = st.sidebar.selectbox(
        "Choose a knowledge base to query:",
        ("Site Navigation", "Heart Disease", "Diabetes", "Lung Cancer", "Kidney Disease")
    )
    
    topic_map = {
        "Site Navigation": "site_navigation.txt", "Heart Disease": "heart_disease.txt",
        "Diabetes": "diabetes.txt", "Lung Cancer": "lung_cancer.txt", "Kidney Disease": "kidney_disease.txt"
    }
    context = load_context(topic_map[topic])
    
    with st.expander(f"📚 See the knowledge base for '{topic}'"):
        st.markdown(f'<div style="height:300px;overflow-y:scroll;padding:10px;border:1px solid gray;">{context}</div>', unsafe_allow_html=True)

    st.divider()
    st.header("Ask a Question")
    user_question = st.text_input("Type your question about the selected topic here:")
    
    if user_question:
        with st.spinner("🧠 Searching for the answer..."):
            result = qa_pipeline(question=user_question, context=context)
            answer_text = result['answer']
            confidence_score = result['score']
            wrapped_text = textwrap.fill(answer_text, width=100)
            
            st.subheader("💡 Answer")
            if confidence_score < 0.3:
                st.warning(f"**Answer (Low Confidence):** {wrapped_text}")
            else:
                st.success(f"**Answer:** {wrapped_text}")
            st.write(f"**Confidence Score:** {confidence_score:.2%}")