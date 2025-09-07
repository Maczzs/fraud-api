from typing import Optional, Dict, List
import os
import json
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------- Load model & preprocessing ----------
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model_rf.pkl")
SCALER_JSON = os.getenv("SCALER_JSON", "scaler_params.json")

# scaler_params.json must contain:
# {
#   "amount_mean": float,
#   "amount_scale": float,
#   "time_mean": float,
#   "time_scale": float,
#   "feature_order": ["Time", "V1", ..., "V28", "Amount"]
# }
with open(SCALER_JSON, "r") as f:
    sp = json.load(f)

FEATURE_ORDER: List[str] = sp["feature_order"]
AMOUNT_MEAN: float  = float(sp["amount_mean"])
AMOUNT_SCALE: float = float(sp["amount_scale"]) if float(sp["amount_scale"]) != 0 else 1.0
TIME_MEAN: float    = float(sp["time_mean"])
TIME_SCALE: float   = float(sp["time_scale"]) if float(sp["time_scale"]) != 0 else 1.0

model = joblib.load(MODEL_PATH)

# ---------- FastAPI ----------
app = FastAPI(title="Credit Card Fraud API (Docker)", version="1.0.0")

# CORS: in production, set your website domain instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    # Either send "features" as a dict by feature name,
    # OR send "values" as a list matching FEATURE_ORDER exactly.
    features: Optional[Dict[str, float]] = None
    values:   Optional[List[float]] = None

class PredictResponse(BaseModel):
    prediction: int
    label: str
    fraud_probability: float

@app.get("/")
def root():
    return {
        "message": "OK. Visit /docs for interactive API, /health for basic check.",
        "endpoints": ["/health", "/predict", "/docs", "/openapi.json"],
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "feature_count": len(FEATURE_ORDER),
        "first_features": FEATURE_ORDER[:5]
    }

def _standardize_row(feats: Dict[str, float]) -> np.ndarray:
    # Standardize Time & Amount using training stats
    if "Amount" in feats:
        feats["Amount"] = (float(feats["Amount"]) - AMOUNT_MEAN) / AMOUNT_SCALE
    if "Time" in feats:
        feats["Time"] = (float(feats["Time"]) - TIME_MEAN) / TIME_SCALE
    row = [float(feats[name]) for name in FEATURE_ORDER]
    return np.array(row, dtype=float).reshape(1, -1)

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if payload.features is not None:
        feats = {k: float(v) for k, v in payload.features.items()}
        missing = [f for f in FEATURE_ORDER if f not in feats]
        if missing:
            return {"prediction": 0, "label": "Invalid Input (missing features)", "fraud_probability": 0.0}
        X = _standardize_row(feats)
    elif payload.values is not None:
        if len(payload.values) != len(FEATURE_ORDER):
            return {"prediction": 0, "label": "Invalid Input (wrong values length)", "fraud_probability": 0.0}
        feats = dict(zip(FEATURE_ORDER, map(float, payload.values)))
        X = _standardize_row(feats)
    else:
        return {"prediction": 0, "label": "Invalid Input (no features)", "fraud_probability": 0.0}

    # Predict using sklearn RF (has predict_proba)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "label": "Fraud" if pred else "Not Fraud",
        "fraud_probability": round(prob, 6)
    }