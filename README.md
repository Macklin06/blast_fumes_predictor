# Blast Fumes Clearance Predictor

FastAPI app + built-in web form for post-blast re-entry wait prediction.

## What this platform does

- Accepts 7 pre-blast operational inputs.
- Predicts clearance wait time using XGBoost.
- Enforces a conservative DGMS minimum wait floor (`DGMS_MIN_WAIT_MINS`, default `30`).
- Enforces a mandatory dynamic minimum wait based on risk inputs (explosive load, ventilation strength, duct distance, heading length, heat/humidity).
- Shows both:
  - model output predicted wait
  - enforced final wait used for recommendation

## Safety positioning

This is **decision support**, not a replacement for DGMS compliance.

- Always follow mandatory inspection, gas checks, and local statutory procedures.
- The app never recommends below configured regulatory floor.
- The app never recommends below its mandatory dynamic minimum policy.

## Local run

```bash
./venv/bin/python model/train.py
./venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

- Web form: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## API test (curl)

```bash
curl -s -X POST http://127.0.0.1:8000/predict_clearance \
  -H 'Content-Type: application/json' \
  -d '{
    "heading_area_m2":15.0,
    "heading_length_m":92.3,
    "explosive_kg":110.7,
    "fan_capacity_m3s":14.0,
    "duct_distance_from_face_m":18.0,
    "temperature_c":32.1,
    "humidity_pct":92.0
  }'
```

## Current benchmark (latest training)

- Model MAE: `7.41 minutes`
- Model R2: `0.7425`
- Fixed 30-minute baseline MAE: `14.91 minutes`
- Fixed 15-minute baseline MAE: `17.32 minutes`

Interpretation: on this dataset split, model error is substantially lower than fixed-wait baselines.


