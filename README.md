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

Forecasts NO2 4 weeks ahead for 12 major world cities from Sentinel-5P satellite data + ERA5-Land rainfall, using a single-layer LSTM trained jointly across 8 cities and evaluated on 4 it never saw during training at all. An interactive Streamlit + pydeck app lets anyone click a city on a world map and see its actual pollution history, the model's forecast stitched through that whole history, and a live forecast for the next 4 weeks — with the model's own generalization skill on that specific city shown directly on the map, not hidden in an appendix.

## Results (see `models/training_metrics.json` and `models/city_skill_summary.json` for full numbers)

- Validation skill vs. a naive "repeat last week" baseline: **+48.9%**
- Held-out, entirely-unseen-city skill: **+7.8%** overall; 3 of the 4 unseen cities individually beat the baseline (Bangkok +33.8%, Karachi +34.6%, Tehran +15.3%); Johannesburg does not (-11.0%) — reported, not hidden.
- Checked against real, independently documented events: the COVID-19 lockdown NO2 drop (Delhi -50%, LA -58%, Tehran -62%, matching the published NASA/ESA TROPOMI finding), the Jan 2025 LA wildfires (aerosol index spikes the exact week the fires ignited), and Delhi's recurring Oct-Nov pollution season (6 of 7 years, and the model beats persistence by +45.4% on the most recent, held-out instance of it). See `outputs/event_evaluation_summary.json` and the `event_*.png` plots.

## Repository structure

```
app.py          Streamlit application entry point (deploy target)
scripts/        Data extraction (Earth Engine), model training, QGIS export, event evaluation
outputs/        Extracted time series, evaluation plots, QGIS GeoPackage, event-check plots
models/         Trained model (.pt), NumPy-exported weights (.npz), scalers, metrics
requirements.txt   Dependencies for the deployed app (no torch — see below)
.streamlit/     App theme config
```

## Cities

Training: Delhi, Beijing, Los Angeles, Mexico City, Cairo, Lagos, Jakarta, London
Held-out test (never seen in training at all): Bangkok, Tehran, Karachi, Johannesburg

## Data sources

Sentinel-5P TROPOMI (NO2 tropospheric column, CO total column, UV Aerosol Index) and ECMWF ERA5-Land Daily Aggregated (total precipitation) — both accessed via Google Earth Engine. ERA5-Land was chosen over the initially-used CHIRPS specifically because CHIRPS only covers 50°S-50°N and silently returned null precipitation for London (51.5°N); rather than exclude London, every city was switched to a source with genuine global coverage.

## Why the deployed app has no PyTorch dependency

The model was trained with PyTorch, but the app runs inference through a hand-written NumPy re-implementation of the exact same LSTM forward pass (`scripts/aq_step4_export_weights_numpy.py`), numerically verified against the live torch model to a maximum absolute difference of 2.4e-07. This keeps the deployed app's dependency list to five small, fast-installing packages instead of a multi-hundred-MB torch wheel — meaningfully faster cold starts on free-tier hosting, with no accuracy cost.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Honest limitations, stated up front

- Satellite NO2/CO/aerosol columns measure total atmospheric column content (mol/m²), not ground-level PM2.5 concentration the way a phone AQI app does. This system tracks and forecasts column pollution trends — genuinely useful for emissions/policy tracking — but is not a certified local air-quality-index reading, and the app says so.
- The model has no input for sudden, unprecedented events (wildfires, policy shocks, industrial accidents) — it forecasts from 12 weeks of a city's own recent trajectory only, and will not anticipate a shock before it appears in the data. Checked explicitly against the Jan 2025 LA wildfires and COVID-19 lockdowns.
- Johannesburg currently loses to the naive persistence baseline (-11% skill) on its held-out weeks. Shown in red on the app's map, not excluded from the results.
