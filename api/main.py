from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Dual-Engine Predictive Maintenance API",
    description="Handles both baseline (FD001) and regime-normalized (FD004) telemetry.",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. LOAD FD001 (BASELINE MODEL)

model_fd001 = xgb.XGBRegressor()
model_fd001.load_model(os.path.join("models", "xgboost_rul_model.json"))

class FD001Features(BaseModel):
    time_cycle: float; op_setting_1: float; op_setting_2: float
    sensor_2: float; sensor_3: float; sensor_4: float; sensor_5: float
    sensor_6: float; sensor_7: float; sensor_8: float; sensor_9: float
    sensor_11: float; sensor_12: float; sensor_13: float; sensor_14: float
    sensor_15: float; sensor_16: float; sensor_17: float; sensor_20: float; sensor_21: float

# 2. LOAD FD004 (ADVANCED MODEL + PREPROCESSORS)

model_fd004 = xgb.XGBRegressor()
model_fd004.load_model(os.path.join("models", "xgboost_fd004_model.json"))
kmeans_fd004 = joblib.load(os.path.join("models", "kmeans_fd004.joblib"))
scaler_fd004 = joblib.load(os.path.join("models", "scaler_fd004.joblib"))

class FD004Features(BaseModel):
    time_cycle: float; op_setting_1: float; op_setting_2: float; op_setting_3: float
    sensor_1: float; sensor_2: float; sensor_3: float; sensor_4: float; sensor_5: float
    sensor_6: float; sensor_7: float; sensor_8: float; sensor_9: float; sensor_10: float
    sensor_11: float; sensor_12: float; sensor_13: float; sensor_14: float; sensor_15: float
    sensor_16: float; sensor_17: float; sensor_18: float; sensor_19: float; sensor_20: float; sensor_21: float

# 3. ROUTING ENDPOINTS

@app.post("/predict")
def predict_fd001(data: FD001Features):
    """Original Endpoint for FD001 Data"""
    df = pd.DataFrame([data.model_dump()])
    rul = max(0, float(model_fd001.predict(df)[0]))
    return {
        "status": "success", "engine_type": "Baseline (FD001)",
        "predicted_rul_cycles": round(rul),
        "health_status": "CRITICAL" if rul < 30 else "WARNING" if rul < 75 else "HEALTHY"
    }

@app.post("/predict/fd004")
def predict_fd004(data: FD004Features):
    """Advanced Endpoint with Regime Normalization for FD004 Data"""
    df = pd.DataFrame([data.model_dump()])
    
    # Apply K-Means Clustering
    op_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    df['flight_regime'] = kmeans_fd004.predict(df[op_settings])
    
    # Apply StandardScaler
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    features_to_scale = op_settings + sensor_cols
    df[features_to_scale] = scaler_fd004.transform(df[features_to_scale])
    
    # Predict
    features = features_to_scale + ['flight_regime']
    rul = max(0, float(model_fd004.predict(df[features])[0]))
    
    return {
        "status": "success", "engine_type": "Advanced Multi-Regime (FD004)",
        "predicted_rul_cycles": round(rul),
        "health_status": "CRITICAL" if rul < 30 else "WARNING" if rul < 75 else "HEALTHY"
    }