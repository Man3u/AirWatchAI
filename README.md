# AirWatchAI

A deep-learning air quality forecasting system for the public and policymakers — built from satellite data, deployed as a live interactive app, not a static report.

## Why this project is different from the rest of this portfolio

Every earlier project (crop stress, reservoir stress, flood risk, glacier retreat) used hand-tuned rules and per-region thresholds — a VV backscatter cutoff here, a rainfall-gate there, a region excluded there because a rule didn't generalize to it. Those were honest engineering calls at the time, but they don't scale, and they're not what "the model learns patterns from data" is supposed to mean.

This project is built to a different, explicit discipline instead:

1. **No region gets special-cased.** Every city runs through the exact same extraction, cleaning, and modeling pipeline. If a city behaves differently, that has to show up as something the model itself learns to handle — not as a hardcoded exception in the code.
2. **Missing or low-quality data gets handled, not dropped.** Satellite retrievals have real gaps (clouds, sensor artifacts, quality-flag failures). Earlier projects sometimes resolved this by excluding a whole region. Here, gaps are interpolated and explicitly flagged with a confidence/coverage value carried alongside every prediction — the pipeline is built to survive imperfect data by design, not by exclusion.
3. **Generalization is proven, not assumed.** The model is validated on cities it never saw during training. If it can't forecast a held-out city reasonably, that's a real result to report, not something to quietly work around.
4. **Uncertainty is shown, not hidden.** Where the model is less confident (sparse data, an unprecedented situation), the app says so, rather than presenting every forecast with false confidence.
5. **The deliverable is a live, working app.** Not a folder of PNGs and a Word doc — an interactive, deployed dashboard anyone can open and use.

## What it does

Forecasts near-term air pollution (NO2, CO, aerosol index) for major world cities from Sentinel-5P satellite data + rainfall, using an LSTM trained jointly across many cities. An interactive Streamlit + pydeck app lets anyone click a city and see its current pollution level, recent trend, and forecast — in plain health-risk language, with honest confidence framing.

## Repository structure

```
scripts/    Data extraction (Earth Engine) and model training code
data/       Static reference data (city coordinates, metadata)
outputs/    Extracted time series, evaluation plots, metrics
models/     Trained model checkpoints + scalers
app/        Streamlit application source
reports/    Any supporting write-up
```

## Cities

Training: Delhi, Beijing, Los Angeles, Mexico City, Cairo, Lagos, Jakarta, London
Held-out test (never seen in training): Bangkok, Tehran, Karachi, Johannesburg

## Data sources

Sentinel-5P TROPOMI (NO2 tropospheric column, CO total column, UV Aerosol Index), CHIRPS Daily precipitation — accessed via Google Earth Engine.

## Honest limitation, stated up front

Satellite NO2/CO/aerosol columns measure total atmospheric column content, not ground-level PM2.5 concentration the way a phone AQI app does. This system tracks and forecasts column pollution trends — genuinely useful for emissions/policy tracking — but is not a certified local air-quality-index reading, and the app says so.
