from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import random
from fastapi.middleware.cors import CORSMiddleware

# --------------------
# CONFIG
# --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CPU_IDLE_THRESHOLD = 10
CPU_SCALE_THRESHOLD = 80
INTERVAL_MINUTES = 5
INTERVAL_HOURS = INTERVAL_MINUTES / 60

# --------------------
# LOAD MODELS & DATA
# --------------------
forecast_model = joblib.load("models/forecast_model.joblib")
anomaly_model = joblib.load("models/anomaly_model.joblib")
kmeans = joblib.load("models/kmeans_model.joblib")
scaler = joblib.load("models/scaler.joblib")

dc = pd.read_parquet("data/datacenter_timeseries.parquet")
vm_features = pd.read_parquet("data/vm_features.parquet")
df = pd.read_parquet("data/vm_level_data.parquet")

app = FastAPI(title="Energy Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# HELPERS
# --------------------
def fake_realtime_dc_power():
    """Simulate incoming real-time power"""
    last_power = dc["dc_power_w"].iloc[-1]
    noise = np.random.normal(0, last_power * 0.01)
    return max(last_power + noise, 0)

def fake_vm_sample():
    """Simulate a VM snapshot"""
    return {
        "cpu_avg": round(random.uniform(0, 100), 2),
        "core_count": random.choice([2, 4, 8, 16, 32, 64])
    }

# --------------------
# API ENDPOINTS
# --------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/realtime/power")
def realtime_power():
    power = fake_realtime_dc_power()
    anomaly = anomaly_model.predict([[power]])[0]

    return {
        "dc_power_w": power,
        "anomaly": bool(anomaly == -1)
    }

@app.get("/realtime/predict")
def predict_power():
    last = dc.tail(3)["dc_power_w"].values
    features = np.array([[last[2], last[1], last.mean()]])
    pred = forecast_model.predict(features)[0]

    return {
        "predicted_power_w": float(pred)
    }

@app.get("/realtime/vm")
def vm_inference():
    vm = fake_vm_sample()

    # Estimate power
    est_power = vm["core_count"] * (60 + (vm["cpu_avg"] / 100) * (250 - 60))

    # Cluster
    X = scaler.transform([[vm["cpu_avg"], vm["core_count"], est_power]])
    cluster = int(kmeans.predict(X)[0])

    # Recommendation
    if vm["cpu_avg"] < CPU_IDLE_THRESHOLD:
        rec = "Downsize or shut down VM"
    elif vm["cpu_avg"] > CPU_SCALE_THRESHOLD:
        rec = "Scale out / add capacity"
    else:
        rec = "Operating normally"

    return {
        "cpu_avg": vm["cpu_avg"],
        "core_count": vm["core_count"],
        "estimated_power_w": est_power,
        "cluster": cluster,
        "recommendation": rec
    }