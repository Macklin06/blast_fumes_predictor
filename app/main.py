from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import os
import math

from app.schemas import BlastInput, PredictionOutput

# ── 1. Load model at startup, not per request ──────────────────
# We use lifespan so the model loads once when the server starts
# and stays in memory. Loading it per-request would be very slow.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
model = None
DGMS_MIN_WAIT_MINS = int(os.getenv("DGMS_MIN_WAIT_MINS", "30"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


def compute_dynamic_mandatory_wait(payload: BlastInput) -> int:
    """Compute an internal mandatory minimum wait based on risk factors."""
    risk_points = 0

    if payload.explosive_kg >= 120:
        risk_points += 10
    elif payload.explosive_kg >= 100:
        risk_points += 6

    if payload.fan_capacity_m3s <= 8:
        risk_points += 8
    elif payload.fan_capacity_m3s <= 10:
        risk_points += 4

    if payload.duct_distance_from_face_m >= 20:
        risk_points += 6
    elif payload.duct_distance_from_face_m >= 16:
        risk_points += 3

    if payload.heading_length_m >= 120:
        risk_points += 4
    elif payload.heading_length_m >= 90:
        risk_points += 2

    if payload.temperature_c >= 34:
        risk_points += 3

    if payload.humidity_pct >= 85:
        risk_points += 3

    return DGMS_MIN_WAIT_MINS + risk_points

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded and ready")
    yield
    print("🛑 Shutting down")

app = FastAPI(
    title="Blast Fumes Clearance Predictor",
    description="Predicts safe re-entry time after a mine blast. Built for DGMS-compliant Indian metal mines.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_web_app():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ── 2. Health check endpoint ───────────────────────────────────
# Always include this. It lets you verify the service is alive
# without making a real prediction. Docker and load balancers
# use this to know if the container is healthy.

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# ── 3. Prediction endpoint ─────────────────────────────────────

@app.post("/predict_clearance", response_model=PredictionOutput)
def predict_clearance(payload: BlastInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    heading_volume = payload.heading_area_m2 * payload.heading_length_m

    # Build input array in exact same column order as training
    features = np.array([[
        payload.heading_area_m2,
        payload.heading_length_m,
        heading_volume,
        payload.explosive_kg,
        payload.fan_capacity_m3s,
        payload.duct_distance_from_face_m,
        payload.temperature_c,
        payload.humidity_pct,
    ]])

    prediction = float(model.predict(features)[0])
    model_time = math.ceil(prediction)
    dynamic_policy_wait = compute_dynamic_mandatory_wait(payload)
    enforced_wait = max(model_time, dynamic_policy_wait)

    # Safety-first: always round UP, never down
    # Under-predicting clearance time has direct safety consequences
    recommendation = (
        f"WAIT — model suggests {model_time} minutes; enforce {enforced_wait} minutes "
        f"(mandatory dynamic minimum: {dynamic_policy_wait}, base floor: {DGMS_MIN_WAIT_MINS})."
    )

    return PredictionOutput(
        predicted_clearance_time_mins=enforced_wait,
        model_predicted_time_mins=model_time,
        regulatory_min_wait_mins=DGMS_MIN_WAIT_MINS,
        dynamic_mandatory_wait_mins=dynamic_policy_wait,
        safety_recommendation=recommendation,
    )