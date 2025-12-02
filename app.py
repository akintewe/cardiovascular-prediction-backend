from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import pandas as pd

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Byte 2 Beat – Final Submission")

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
    smoke: int           # 0/1
    alco: int            # 0/1
    active: int          # 0/1

    # ============ INPUT VALIDATION (PREVENTS CRASHES) ============
    @validator('height')
    def height_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Height must be greater than 0 cm')
        if v < 100 or v > 250:
            raise ValueError('Height must be between 100 and 250 cm')
        return v

    @validator('weight')
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Weight must be greater than 0 kg')
        return v

    @validator('ap_hi')
    def ap_hi_valid(cls, v):
        if v < 70 or v > 250:
            raise ValueError('Systolic BP (ap_hi) must be 70–250 mmHg')
        return v

    @validator('ap_lo')
    def ap_lo_valid(cls, v):
        if v < 40 or v > 150:
            raise ValueError('Diastolic BP (ap_lo) must be 40–150 mmHg')
        return v

    @validator('age')
    def age_valid(cls, v):
        if v < 20 or v > 100:
            raise ValueError('Age must be 20–100 years')
        return v

@app.get("/")
def home():
    return {"message": "Byte 2 Beat API – LIVE & BULLETPROOF", "status": "ready"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Feature engineering (now safe)
        bmi = data.weight / ((data.height / 100) ** 2)
        pulse_pressure = data.ap_hi - data.ap_lo
        map_val = data.ap_lo + (data.ap_hi - data.ap_lo) / 3
        age_bmi = data.age * bmi
        age_ap_hi = data.age * data.ap_hi
        bmi_ap_hi = bmi * data.ap_hi

        # Full dataframe (exact match to training)
        df = pd.DataFrame([{
            'age': data.age, 'gender': data.gender, 'ap_hi': data.ap_hi, 'ap_lo': data.ap_lo,
            'cholesterol': data.cholesterol, 'gluc': data.gluc, 'smoke': data.smoke,
            'alco': data.alco, 'active': data.active, 'bmi': bmi,
            'pulse_pressure': pulse_pressure, 'map': map_val,
            'age_bmi': age_bmi, 'age_ap_hi': age_ap_hi, 'bmi_ap_hi': bmi_ap_hi,
            'max_hr': 150, 'oldpeak': 0.0, 'cp': 0, 'exang': 0, 'slope': 2,
            'ca': 0, 'thal': 2, 'restecg': 0, 'fbs': 0
        }])

        # One-hot + reindex exactly like training
        cat_cols = ['gender', 'cholesterol', 'gluc', 'cp', 'slope', 'ca', 'thal', 'restecg']
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
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
            "bmi": round(bmi, 1),
            "message": "Prediction successful!"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")