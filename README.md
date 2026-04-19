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

## Profit estimate vs fixed downtime

If your site currently applies fixed 30 or 45 minute waits, run:

```bash
./venv/bin/python model/profit_analysis.py --dgms-floor 30 --value-per-hour 100000 --blasts-per-day 2 --days-per-month 26
```

What it reports:

- Average and median time delta per blast versus fixed 30 and fixed 45 methods.
- Share of blasts where time is saved.
- Net delta (can be negative on some blasts for safety).
- Estimated monthly net value using your economics assumptions.

Important:

- Do not claim savings by going below statutory minimum wait.
- Use `net delta` for honest value estimation, not only positive cases.

## Deploy fastest path (Render)

This repo includes `render.yaml` and a Docker image setup.

1. Push repository to GitHub.
2. In Render, create new Web Service from repo.
3. Render auto-detects `render.yaml`.
4. Set environment variable if needed:
   - `DGMS_MIN_WAIT_MINS=30`
  - `ALLOWED_ORIGINS=*`
5. Deploy and test:
   - `GET /health`
   - open `/` web form

## Docker notes

`Dockerfile` uses:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

This works locally and on hosted platforms that inject `PORT`.

## Publish frontend on GitHub Pages

This repo now includes [`.github/workflows/pages.yml`](.github/workflows/pages.yml) to publish the slider frontend from `app/static`.

1. Push to `main`.
2. In GitHub repo settings, open **Settings -> Pages** and set Source to **GitHub Actions**.
3. Workflow runs and publishes to:
  - `https://<your-username>.github.io/<your-repo>/`

Notes:

- This GitHub Pages version is fully static and runs prediction in-browser.
- No backend or CORS setup is required for the published page.
- The browser model is a lightweight surrogate estimate for demo/accessibility.

## Commit and push these changes

```bash
git add app/static/index.html .github/workflows/pages.yml README.md
git commit -m "Make frontend GitHub Pages-only with browser prediction"
git push origin main
```
