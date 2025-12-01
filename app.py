from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Byte 2 Beat Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    age: int
    gender: int          # 1 = male, 2 = female
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int     # 1,2,3
    gluc: int            # 1,2,3
    smoke: int
    alco: int
    active: int

@app.get("/")
def home():
    return {"message": "Byte 2 Beat API is LIVE", "features": model.n_features_in_}

@app.post("/predict")
def predict(data: PatientData):
    # Feature engineering
    bmi = data.weight / ((data.height / 100) ** 2)
    pulse_pressure = data.ap_hi - data.ap_lo
    map_val = data.ap_lo + (data.ap_hi - data.ap_lo) / 3
    age_bmi = data.age * bmi
    age_ap_hi = data.age * data.ap_hi
    bmi_ap_hi = bmi * data.ap_hi

    # Create full row exactly like training
    df = pd.DataFrame([{
        'age': data.age,
        'gender': data.gender,
        'ap_hi': data.ap_hi,
        'ap_lo': data.ap_lo,
        'cholesterol': data.cholesterol,
        'gluc': data.gluc,
        'smoke': data.smoke,
        'alco': data.alco,
        'active': data.active,
        'bmi': bmi,
        'pulse_pressure': pulse_pressure,
        'map': map_val,
        'age_bmi': age_bmi,
        'age_ap_hi': age_ap_hi,
        'bmi_ap_hi': bmi_ap_hi,
        'max_hr': 150,        # default fallback (most common in dataset)
        'oldpeak': 0.0,       # default fallback
        'cp': 0,              # assume no chest pain
        'exang': 0,
        'slope': 2,
        'ca': 0,
        'thal': 2,
        'restecg': 0,
        'fbs': 0
    }])

    # One-hot encode exactly like training
    cat_cols = ['gender', 'cholesterol', 'gluc', 'cp', 'slope', 'ca', 'thal', 'restecg']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add back any missing one-hot columns with 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training exactly
    df = df[model.feature_names_in_]

    # Scale numerical columns
    num_cols = ['age', 'ap_hi', 'ap_lo', 'bmi', 'max_hr', 'oldpeak',
                'pulse_pressure', 'map', 'age_bmi', 'age_ap_hi', 'bmi_ap_hi']
    df[num_cols] = scaler.transform(df[num_cols])

    # Predict
    prob = float(model.predict_proba(df.values)[0, 1])
    risk = "High Risk" if prob >= 0.5 else "Low Risk"

    return {
        "risk": risk,
        "probability": round(prob, 3),
        "status": "Success"
    }

