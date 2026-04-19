import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_ventilation_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

df = pd.read_csv(DATA_PATH)

# Derived geometric signal: volume = area x length.
# This is available before blast and avoids leaking post-blast quantities.
df["heading_volume_m3"] = df["heading_area_m2"] * df["heading_length_m"]

# ── 2. Define features (X) and target (y) ─────────────────────
# These are ONLY the columns an engineer knows before re-entry
# We deliberately exclude initial_co_ppm and initial_nox_ppm
# because those are intermediate physics calculations, not
# something an engineer measures before a blast

FEATURES = [
    "heading_area_m2",
    "heading_length_m",
    "heading_volume_m3",
    "explosive_kg",
    "fan_capacity_m3s",
    "duct_distance_from_face_m",
    "temperature_c",
    "humidity_pct",
]

TARGET = "safe_reentry_time_mins"

X = df[FEATURES]
y = df[TARGET]

# ── 3. Split into train and test sets ─────────────────────────
# 80% of data trains the model, 20% is held back to evaluate it
# random_state=42 means you get the same split every time you run it
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train XGBoost model ────────────────────────────────────
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    verbosity=0,
)

model.fit(X_train, y_train)

# ── 5. Evaluate on the held-back test set ─────────────────────
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

# Fixed-wait baselines to compare with a regulation-style constant wait.
baseline_30 = np.full_like(y_test.to_numpy(), 30.0)
baseline_15 = np.full_like(y_test.to_numpy(), 15.0)
mae_30 = mean_absolute_error(y_test, baseline_30)
mae_15 = mean_absolute_error(y_test, baseline_15)

print(f"✅ Model trained successfully")
print(f"   MAE : {mae:.2f} minutes")
print(f"   R²  : {r2:.4f}")
print(f"   Baseline MAE (fixed 30 min): {mae_30:.2f} minutes")
print(f"   Baseline MAE (fixed 15 min): {mae_15:.2f} minutes")

# ── 6. Save the trained model ──────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"   Model saved to {MODEL_PATH}")