# File: backend/api.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import os
import shap
import numpy as np

# --- 1. App Initialization ---
app = FastAPI(
    title="Precision Medicine API Suite",
    description="API for predicting Heart Disease, Diabetes, Lung Cancer, and Kidney Disease.",
    version="1.0"
)

# --- 2. CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Asset Loading ---
ASSETS = {}

def load_prediction_assets(model_name):
    # ... (This function remains exactly the same) ...
    assets = {}
    base_path = os.path.join("models", model_name)
    assets['model'] = joblib.load(os.path.join(base_path, f"best_{model_name}_classifier.joblib"))
    assets['scaler'] = joblib.load(os.path.join(base_path, f"{model_name}_scaler.joblib"))
    feature_path = os.path.join(base_path, f"{model_name}_features.joblib")
    if os.path.exists(feature_path):
        assets['features'] = joblib.load(feature_path)
    assets['explainer'] = shap.TreeExplainer(assets['model'])
    return assets

@app.on_event("startup")
def startup_event():
    """Load all models into memory when the API starts up."""
    print("--- Loading all model assets... ---")
    ASSETS['heart_disease'] = load_prediction_assets('heart_disease')
    ASSETS['diabetes'] = load_prediction_assets('diabetes')
    ASSETS['lung_cancer'] = load_prediction_assets('lung_cancer')
    ASSETS['kidney_disease'] = load_prediction_assets('kidney_disease')
    print("--- All assets loaded successfully! ---")

# --- 4. Pydantic Models (Input Schemas) ---

# Static models for simple cases
class HeartDiseaseInput(BaseModel):
    age: int; anaemia: int; creatinine_phosphokinase: int; diabetes: int
    ejection_fraction: int; high_blood_pressure: int; platelets: float
    serum_creatinine: float; serum_sodium: int; sex: int; smoking: int; time: int

class DiabetesInput(BaseModel):
    Pregnancies: int; Glucose: int; BloodPressure: int; SkinThickness: int
    Insulin: int; BMI: float; DiabetesPedigreeFunction: float; Age: int

# --- THIS IS THE FIX ---
# Load feature lists first to dynamically create Pydantic models statically
try:
    lung_cancer_features = joblib.load(os.path.join("models", "lung_cancer", "lung_cancer_features.joblib"))
    LungCancerInput = create_model('LungCancerInput', **{f: (int, ...) for f in lung_cancer_features})
except FileNotFoundError:
    # Define a placeholder if the file doesn't exist, to prevent crashes
    class LungCancerInput(BaseModel):
        error: str = "Model assets not found"

try:
    kidney_disease_features = joblib.load(os.path.join("models", "kidney_disease", "kidney_disease_features.joblib"))
    KidneyDiseaseInput = create_model('KidneyDiseaseInput', **{f: (float, ...) for f in kidney_disease_features})
except FileNotFoundError:
    class KidneyDiseaseInput(BaseModel):
        error: str = "Model assets not found"


# --- 5. Reusable Prediction Service ---
def perform_prediction(model_name: str, input_df: pd.DataFrame):
    # ... (This function remains exactly the same) ...
    assets = ASSETS[model_name]
    model, scaler, explainer = assets['model'], assets['scaler'], assets['explainer']
    feature_order = assets.get('features', list(input_df.columns))
    input_df_ordered = input_df.reindex(columns=feature_order, fill_value=0)
    
    scaled_input_df = input_df_ordered.copy()
    if hasattr(scaler, 'feature_names_in_'):
        cols_to_scale = scaler.feature_names_in_
        cols_that_exist = [col for col in cols_to_scale if col in scaled_input_df.columns]
        if cols_that_exist:
            scaled_input_df[cols_that_exist] = scaler.transform(scaled_input_df[cols_that_exist])
    
    prediction = model.predict(scaled_input_df)
    prediction_proba = model.predict_proba(scaled_input_df)
    explanation = explainer(scaled_input_df)
    if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
        explanation_to_plot = explanation[0, :, 1]
    else:
        explanation_to_plot = explanation[0]
    return {
        "prediction": int(prediction[0]),
        "prediction_probability_high_risk": float(prediction_proba[0][1]),
        "explanation": {
            "base_value": float(explanation_to_plot.base_values),
            "shap_values": explanation_to_plot.values.tolist(),
            "feature_names": scaled_input_df.columns.tolist(),
        }
    }

# --- 6. API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Precision Medicine API is running."}

@app.post("/predict/heart_disease")
def predict_heart_disease_endpoint(input_data: HeartDiseaseInput):
    return perform_prediction('heart_disease', pd.DataFrame([input_data.dict()]))

@app.post("/predict/diabetes")
def predict_diabetes_endpoint(input_data: DiabetesInput):
    return perform_prediction('diabetes', pd.DataFrame([input_data.dict()]))

@app.post("/predict/lung_cancer")
def predict_lung_cancer_endpoint(input_data: LungCancerInput):
    return perform_prediction('lung_cancer', pd.DataFrame([input_data.dict()]))

@app.post("/predict/kidney_disease")
def predict_kidney_disease_endpoint(input_data: KidneyDiseaseInput):
    return perform_prediction('kidney_disease', pd.DataFrame([input_data.dict()]))